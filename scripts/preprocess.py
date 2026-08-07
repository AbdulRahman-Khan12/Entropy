"""
scripts/preprocess.py  --  Entropy Stage 2: ingestion and preprocessing
=======================================================================

Reads the frozen corpus, runs it through spaCy exactly once, and writes:

  1. data/corpus/cache/<doc_id>.spacy   annotated Doc per article
  2. data/corpus/cache/cache_manifest.json   provenance + freshness hashes
  3. outputs/preprocessing_report.csv        per-document statistics

Every later stage reads the cache instead of re-running spaCy:

    from entropy.nlp import load_doc, iter_cached_docs
    doc = load_doc("wn-0042")

Usage
-----
    python scripts/preprocess.py                  # build / refresh the cache
    python scripts/preprocess.py --limit 20       # quick smoke test
    python scripts/preprocess.py --force          # rebuild everything
    python scripts/preprocess.py --tokenize-only  # tokens + sentences, no POS/NER
    python scripts/preprocess.py --doc-id wn-0042 # single article

Re-runs are cheap.  A document is re-processed only if its text changed or the
spaCy / model version moved; otherwise its cache file is reused and only the
report is regenerated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import time
from pathlib import Path

# Allow `python scripts/preprocess.py` from the project root without installing
# the package.  Never hardcode absolute paths - resolve relative to this file.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from spacy.tokens import Doc  # noqa: E402

from entropy.corpus import iter_corpus  # noqa: E402
from entropy import nlp as enlp  # noqa: E402


PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")

REPORT_COLUMNS = [
    # required by the exercise brief, in order
    "doc_id", "n_sentences", "n_tokens", "n_words", "dct",
    # additional context, useful for the Stage 7 report and for sanity checks
    "title", "n_chars", "n_paragraphs", "mean_words_per_sentence",
    "n_entities", "categories", "cache_file",
]


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def count_paragraphs(text: str) -> int:
    return sum(1 for block in PARAGRAPH_SPLIT.split(text) if block.strip())


def doc_stats(doc: Doc, *, raw_text: str | None = None) -> dict:
    """Per-document counts for the preprocessing report.

    Token accounting, stated explicitly because the two numbers differ and the
    report has a column for each:

      * ``n_tokens``  every spaCy token, including punctuation and the
                      whitespace tokens created by paragraph breaks.
      * ``n_words``   tokens that are neither punctuation nor whitespace.
                      This is the figure to quote as "corpus size" in the
                      write-up; ``n_tokens`` is the figure that matters for
                      model input length.
    """
    text = raw_text if raw_text is not None else doc.text
    sentences = list(doc.sents) if doc.has_annotation("SENT_START") else []
    n_tokens = len(doc)
    n_words = sum(1 for t in doc if not (t.is_punct or t.is_space))
    n_sentences = len(sentences)

    dct = enlp.get_dct(doc)
    categories = doc._.categories or []

    return {
        "doc_id": doc._.doc_id,
        "n_sentences": n_sentences,
        "n_tokens": n_tokens,
        "n_words": n_words,
        "dct": dct.isoformat() if dct else "",
        "title": doc._.title or "",
        "n_chars": len(text),
        "n_paragraphs": count_paragraphs(text),
        "mean_words_per_sentence": round(n_words / n_sentences, 2) if n_sentences else 0.0,
        "n_entities": len(doc.ents),
        "categories": "|".join(categories),
        "cache_file": enlp.cache_path(doc._.doc_id).name,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entropy Stage 2: preprocessing and annotation cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=enlp.SPACY_MODEL,
                   help="spaCy model to load")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N documents")
    p.add_argument("--doc-id", action="append", dest="doc_ids", default=None,
                   help="process only this doc_id (repeatable)")
    p.add_argument("--force", action="store_true",
                   help="re-process documents even if the cache is fresh")
    p.add_argument("--tokenize-only", action="store_true",
                   help="tokens and sentences only; no POS, parse or NER")
    p.add_argument("--no-para-breaks", action="store_true",
                   help="do not force sentence breaks at blank lines")
    p.add_argument("--batch-size", type=int, default=enlp.BATCH_SIZE,
                   help="spaCy nlp.pipe batch size")
    p.add_argument("--n-process", type=int, default=1,
                   help="worker processes for nlp.pipe (Windows: keep at 1)")
    p.add_argument("--report", type=Path, default=None,
                   help="report path (default: outputs/preprocessing_report.csv)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    started = time.perf_counter()

    report_path = args.report or (enlp.OUTPUTS_DIR / "preprocessing_report.csv")

    if args.n_process > 1 and sys.platform.startswith("win"):
        print("[warn] n_process > 1 is unreliable on Windows; falling back to 1.")
        args.n_process = 1

    # ---- pipeline -------------------------------------------------------
    nlp = enlp.load_nlp(
        args.model,
        tokenize_only=args.tokenize_only,
        paragraph_breaks_enabled=not args.no_para_breaks,
    )
    signature = enlp.pipeline_signature(nlp)

    print("Entropy Stage 2: preprocessing")
    print(f"  project root : {enlp.PROJECT_ROOT}")
    print(f"  cache dir    : {enlp.CACHE_DIR}")
    print(f"  report       : {report_path}")
    print(f"  spaCy        : {signature['spacy_version']}")
    print(f"  model        : {signature['model']}-{signature['model_version']}")
    print(f"  pipes        : {', '.join(signature['pipes']) or '(none)'}")
    print()

    # ---- select documents ----------------------------------------------
    documents = list(iter_corpus())
    if args.doc_ids:
        wanted = set(args.doc_ids)
        documents = [d for d in documents if d.doc_id in wanted]
        missing = wanted - {d.doc_id for d in documents}
        if missing:
            print(f"[warn] not in corpus: {', '.join(sorted(missing))}")
    if args.limit is not None:
        documents = documents[: args.limit]

    if not documents:
        print("No documents selected. Nothing to do.")
        return 1

    manifest = enlp.read_manifest()
    doc_entries = manifest.setdefault("documents", {})

    todo, reuse = [], []
    for document in documents:
        if not args.force and enlp.is_cache_fresh(
            document.doc_id, document.text, signature, manifest
        ):
            reuse.append(document)
        else:
            todo.append(document)

    print(f"{len(documents)} document(s) selected: "
          f"{len(todo)} to process, {len(reuse)} cached and fresh.")

    rows: list[dict] = []

    # ---- process what needs processing ----------------------------------
    if todo:
        enlp.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        stream = nlp.pipe(
            ((d.text, d) for d in todo),
            as_tuples=True,
            batch_size=args.batch_size,
            n_process=args.n_process,
        )
        for i, (doc, document) in enumerate(stream, start=1):
            enlp.attach_metadata(
                doc,
                doc_id=document.doc_id,
                title=document.title,
                dct=document.dct,
                categories=document.categories,
            )
            enlp.save_doc(doc, document.doc_id)
            doc_entries[document.doc_id] = {
                "text_sha256": enlp.text_hash(document.text),
                "processed_at": dt.datetime.now().isoformat(timespec="seconds"),
                **signature,
            }
            rows.append(doc_stats(doc, raw_text=document.text))
            if i % 50 == 0 or i == len(todo):
                print(f"  processed {i}/{len(todo)}")

    # ---- reuse the rest, but still report on it -------------------------
    if reuse:
        by_id = {d.doc_id: d for d in reuse}
        for doc in enlp.iter_cached_docs(list(by_id), nlp=nlp):
            rows.append(doc_stats(doc, raw_text=by_id[doc._.doc_id].text))
        print(f"  reused {len(reuse)} cached document(s)")

    # ---- manifest --------------------------------------------------------
    manifest["generated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    manifest["pipeline"] = signature
    manifest["n_documents"] = len(doc_entries)
    enlp.write_manifest(manifest)

    # ---- report ----------------------------------------------------------
    frame = pd.DataFrame(rows, columns=REPORT_COLUMNS).sort_values("doc_id")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(report_path, index=False, encoding="utf-8")

    elapsed = time.perf_counter() - started
    print()
    print(f"Wrote {len(frame)} rows -> {report_path}")
    print(f"Cache: {enlp.CACHE_DIR} ({len(enlp.cached_doc_ids())} files)")
    print(f"Totals: {int(frame.n_sentences.sum()):,} sentences, "
          f"{int(frame.n_tokens.sum()):,} tokens, "
          f"{int(frame.n_words.sum()):,} words")
    print(f"Per document (mean): {frame.n_sentences.mean():.1f} sentences, "
          f"{frame.n_words.mean():.0f} words")
    if frame["dct"].astype(bool).any():
        dates = frame.loc[frame["dct"].astype(bool), "dct"]
        print(f"DCT range: {dates.min()} .. {dates.max()}")
    empty = frame[frame.n_words == 0]
    if not empty.empty:
        print(f"[warn] {len(empty)} document(s) produced zero words: "
              f"{', '.join(empty.doc_id.head(5))}")
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
