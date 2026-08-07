"""
entropy/nlp.py
==============
Shared spaCy plumbing for the Entropy pipeline.

This module is the single place where the project decides:
  * which spaCy model to load and how to configure it,
  * how document metadata (doc_id / title / dct / categories) rides along
    with a spaCy ``Doc``,
  * how annotated documents are written to and read back from the on-disk
    cache in ``data/corpus/cache/``.

Stage 2 (scripts/preprocess.py) *writes* the cache.
Stages 3-7 should only ever *read* it, via :func:`load_doc` /
:func:`iter_cached_docs`.  Nothing downstream should call ``nlp(text)`` on
corpus text again.

Cache format
------------
One ``DocBin`` per document at ``data/corpus/cache/<doc_id>.spacy``.
``DocBin`` preserves the complete ``Doc``: tokens, sentence boundaries,
POS/TAG, lemmas, dependency arcs and NER spans, plus ``doc.user_data``
(which is where the ``._.`` extensions live).

Reading the cache does NOT require the full model.  A blank English vocab is
enough, because ``DocBin`` serialises the strings it needs.  That matters for
Streamlit Community Cloud, where loading ``en_core_web_sm`` costs RAM we would
rather spend elsewhere::

    from entropy.nlp import load_doc
    doc = load_doc("wn-0042")          # ~no model load, blank vocab
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import spacy
from spacy.language import Language
from spacy.tokens import Doc, DocBin

from entropy import config
from entropy.gazetteer import add_gazetteer

# --------------------------------------------------------------------------
# Paths and settings, resolved from config.py with sane fallbacks.
#
# config.py is the tuning surface for the whole project, so anything defined
# there wins.  The getattr() fallbacks exist so this module still imports
# cleanly before you have added the Stage 2 settings to config.py.
# --------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(config.PROJECT_ROOT)
DATA_DIR: Path = Path(getattr(config, "DATA_DIR", PROJECT_ROOT / "data"))
CORPUS_DIR: Path = Path(getattr(config, "CORPUS_DIR", DATA_DIR / "corpus"))
CACHE_DIR: Path = Path(getattr(config, "CACHE_DIR", CORPUS_DIR / "cache"))
OUTPUTS_DIR: Path = Path(
    getattr(config, "OUTPUTS_DIR", getattr(config, "OUTPUT_DIR", PROJECT_ROOT / "outputs"))
)

SPACY_MODEL: str = getattr(config, "SPACY_MODEL", "en_core_web_sm")
BATCH_SIZE: int = getattr(config, "SPACY_BATCH_SIZE", 32)
USE_GAZETTEER: bool = getattr(config, "USE_GAZETTEER", True)
CACHE_MANIFEST: Path = CACHE_DIR / "cache_manifest.json"
CACHE_SUFFIX = ".spacy"


# --------------------------------------------------------------------------
# Doc extensions: metadata that must survive the round-trip to disk
# --------------------------------------------------------------------------

#: Extension name -> default value.  ``dct`` is stored as an ISO-8601 *string*
#: rather than a ``datetime.date`` because ``DocBin`` serialises user_data with
#: msgpack, which has no native date type.  Use :func:`get_dct` to read it back
#: as a real ``date``.
_EXTENSIONS: dict[str, object] = {
    "doc_id": None,
    "title": None,
    "dct": None,        # ISO string, e.g. "2019-07-16"
    "categories": None,  # list[str]
}


def register_extensions() -> None:
    """Register the ``Doc._.`` extensions used by Entropy.

    Idempotent, and safe to call from anywhere.  Must be called *before*
    deserialising a cached ``Doc``, otherwise the extension values in
    ``user_data`` have nowhere to land.
    """
    for name, default in _EXTENSIONS.items():
        if not Doc.has_extension(name):
            Doc.set_extension(name, default=default)


def attach_metadata(doc: Doc, *, doc_id: str, title: str | None = None,
                    dct: dt.date | str | None = None,
                    categories: Sequence[str] | None = None) -> Doc:
    """Attach corpus metadata to a ``Doc`` so it travels with the annotations."""
    register_extensions()
    doc._.doc_id = doc_id
    doc._.title = title
    if isinstance(dct, dt.date):
        doc._.dct = dct.isoformat()
    else:
        doc._.dct = dct
    doc._.categories = list(categories) if categories else []
    return doc


def get_dct(doc: Doc) -> dt.date | None:
    """Return the Document Creation Time as a ``date``, or ``None``.

    Stage 5 uses this as the anchor for resolving relative TIMEX expressions
    ("on Tuesday", "last week") to absolute dates, so it is worth having one
    canonical accessor rather than parsing the string in five places.
    """
    register_extensions()
    raw = doc._.dct
    if not raw:
        return None
    if isinstance(raw, dt.date):
        return raw
    try:
        return dt.date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Custom pipeline component: force sentence breaks at paragraph boundaries
# --------------------------------------------------------------------------

@Language.component("entropy_paragraph_breaks")
def paragraph_breaks(doc: Doc) -> Doc:
    """Force a sentence boundary after a blank line.

    Corpus text stores paragraphs separated by blank lines.  Without this, a
    headline or a paragraph that lacks terminal punctuation gets glued to the
    following paragraph, which produces monster "sentences" that wreck
    per-sentence event extraction in Stage 5.

    Placement matters:
      * before the parser  -> the parser honours pre-set boundaries;
      * after senter/sentencizer -> those components would otherwise overwrite
        our flags, and writing ``is_sent_start`` is still legal because no
        dependency annotation exists yet.
    """
    for i, token in enumerate(doc):
        if i + 1 >= len(doc):
            break
        blank_line = (
            (token.is_space and token.text.count("\n") >= 2)
            or token.whitespace_.count("\n") >= 2
        )
        if blank_line:
            doc[i + 1].is_sent_start = True
    return doc


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

_NLP_CACHE: dict[tuple, Language] = {}

_MODEL_MISSING_HINT = """
spaCy model {model!r} is not installed in this environment.

    python -m spacy download {model}

For Streamlit Community Cloud, `spacy download` is not run at deploy time.
Pin the wheel in requirements.txt instead - see the project README.
""".strip()


def load_nlp(model: str = SPACY_MODEL, *, tokenize_only: bool = False,
             paragraph_breaks_enabled: bool = True,
             use_gazetteer: bool = USE_GAZETTEER,
             exclude: Iterable[str] = ()) -> Language:
    """Load and configure the spaCy pipeline.

    Parameters
    ----------
    model:
        Model name, default ``en_core_web_sm``.  Kept small deliberately:
        Streamlit Community Cloud gives us roughly 1 GB of RAM, and
        ``en_core_web_trf`` does not fit alongside the rest of the app.
    tokenize_only:
        Build a blank English pipeline with only a sentencizer.  Produces a
        tokenisation-and-sentences cache with no POS/NER, if you would rather
        Stage 3 do the tagging itself.
    paragraph_breaks_enabled:
        Insert the :func:`paragraph_breaks` component.
    exclude:
        Pipe names to leave out of the loaded model.
    """
    key = (model, tokenize_only, paragraph_breaks_enabled, use_gazetteer,
           tuple(exclude))
    if key in _NLP_CACHE:
        return _NLP_CACHE[key]

    if tokenize_only:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
    else:
        try:
            nlp = spacy.load(model, exclude=list(exclude))
        except OSError as exc:  # model not installed
            raise SystemExit(_MODEL_MISSING_HINT.format(model=model)) from exc
    if use_gazetteer and not tokenize_only:
        add_gazetteer(nlp)
    if paragraph_breaks_enabled:
        names = nlp.pipe_names
        if "parser" in names:
            nlp.add_pipe("entropy_paragraph_breaks", before="parser")
        elif "senter" in names:
            nlp.add_pipe("entropy_paragraph_breaks", after="senter")
        elif "sentencizer" in names:
            nlp.add_pipe("entropy_paragraph_breaks", after="sentencizer")
        else:
            nlp.add_pipe("entropy_paragraph_breaks", last=True)

    register_extensions()
    _NLP_CACHE[key] = nlp
    return nlp


def pipeline_signature(nlp: Language) -> dict[str, object]:
    """Describe the loaded pipeline, for cache-invalidation bookkeeping."""
    meta = nlp.meta
    name = meta.get("name", "blank")
    lang = meta.get("lang", "en")
    version = meta.get("version", "0.0.0")
    return {
        "spacy_version": spacy.__version__,
        "model": f"{lang}_{name}",
        "model_version": version,
        "pipes": list(nlp.pipe_names),
    }


# --------------------------------------------------------------------------
# Cache I/O
# --------------------------------------------------------------------------

def text_hash(text: str) -> str:
    """Stable content hash, used to detect corpus edits and re-run only those docs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_path(doc_id: str) -> Path:
    return CACHE_DIR / f"{doc_id}{CACHE_SUFFIX}"


def save_doc(doc: Doc, doc_id: str | None = None) -> Path:
    """Serialise one annotated ``Doc`` to the cache."""
    register_extensions()
    doc_id = doc_id or doc._.doc_id
    if not doc_id:
        raise ValueError("save_doc needs a doc_id (pass it, or set doc._.doc_id)")
    path = cache_path(doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = DocBin(store_user_data=True)
    db.add(doc)
    db.to_disk(path)
    return path


def load_doc(doc_id: str, *, nlp: Language | None = None) -> Doc:
    """Read one annotated ``Doc`` back from the cache.

    Uses a blank English vocab unless a pipeline is supplied, so callers that
    only need to inspect annotations do not pay to load the model.
    """
    register_extensions()
    path = cache_path(doc_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached annotations for {doc_id!r} at {path}. "
            "Run: python scripts/preprocess.py"
        )
    vocab = nlp.vocab if nlp is not None else spacy.blank("en").vocab
    db = DocBin().from_disk(path)
    docs = list(db.get_docs(vocab))
    if not docs:
        raise ValueError(f"Cache file {path} contains no documents")
    return docs[0]


def cached_doc_ids() -> list[str]:
    if not CACHE_DIR.exists():
        return []
    return sorted(p.stem for p in CACHE_DIR.glob(f"*{CACHE_SUFFIX}"))


def iter_cached_docs(doc_ids: Iterable[str] | None = None, *,
                     nlp: Language | None = None) -> Iterator[Doc]:
    """Yield cached ``Doc`` objects one at a time, keeping memory flat."""
    register_extensions()
    vocab = nlp.vocab if nlp is not None else spacy.blank("en").vocab
    ids = list(doc_ids) if doc_ids is not None else cached_doc_ids()
    for doc_id in ids:
        path = cache_path(doc_id)
        if not path.exists():
            continue
        db = DocBin().from_disk(path)
        for doc in db.get_docs(vocab):
            yield doc


# --------------------------------------------------------------------------
# Cache manifest: what was built, from what, with which model
# --------------------------------------------------------------------------

def read_manifest() -> dict:
    if not CACHE_MANIFEST.exists():
        return {"documents": {}}
    try:
        with CACHE_MANIFEST.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"documents": {}}
    data.setdefault("documents", {})
    return data


def write_manifest(manifest: dict) -> Path:
    CACHE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_MANIFEST.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    tmp.replace(CACHE_MANIFEST)
    return CACHE_MANIFEST


def is_cache_fresh(doc_id: str, text: str, signature: dict, manifest: dict) -> bool:
    """True if the cached annotations still match the text and the pipeline.

    Guards against the two ways a cache silently goes stale: the article text
    changed, or spaCy / the model was upgraded and the annotations no longer
    correspond to the version the rest of the pipeline assumes.
    """
    entry = manifest.get("documents", {}).get(doc_id)
    if not entry or not cache_path(doc_id).exists():
        return False
    return (
        entry.get("text_sha256") == text_hash(text)
        and entry.get("spacy_version") == signature["spacy_version"]
        and entry.get("model") == signature["model"]
        and entry.get("model_version") == signature["model_version"]
        and entry.get("pipes") == signature["pipes"]
    )
