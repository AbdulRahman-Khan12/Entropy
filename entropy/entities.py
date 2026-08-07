"""
entropy/entities.py
===================
The entity and POS layer of the Entropy pipeline.

Reads annotated ``Doc`` objects from the Stage 2 cache and turns them into the
tabular artefacts the exercise asks for.  Nothing here re-runs spaCy.

What this module produces
-------------------------
``EntityMention``    one row per entity occurrence, with character offsets
                     back into the source text
canonical forms      "the National Aeronautics and Space Administration",
                     "NASA's" and "NASA" all collapse to the same key, so the
                     entity table counts organisations rather than spellings
POS distributions    corpus-level tag counts for the report
CoNLL rows           per-token token/lemma/POS/dep/BIO, the standard shape for
                     inspection and for the Stage 5 evaluation

A note on temporal entities
---------------------------
``DATE`` and ``TIME`` spans are kept, not filtered out.  They are the raw
material for TIMEX normalisation in the next stage, where each one is resolved
against the document's DCT.  :func:`is_temporal` marks them so the relation
extractor can skip them while the temporal stage picks them up.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Iterable, Iterator

from spacy.tokens import Doc, Span

from entropy import config
from entropy import nlp as enlp

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

#: Labels kept in the entity table.  CARDINAL/ORDINAL/PERCENT/QUANTITY/MONEY
#: are dropped by default: on this corpus they are mostly measurement noise
#: ("about 400 km", "the third") that adds rows without adding information.
DEFAULT_ENTITY_LABELS = (
    "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
    "SPACECRAFT", "CELESTIAL",
    "DATE", "TIME",
)

ENTITY_LABELS: tuple[str, ...] = tuple(
    getattr(config, "ENTITY_LABELS", DEFAULT_ENTITY_LABELS)
)

TEMPORAL_LABELS: tuple[str, ...] = tuple(
    getattr(config, "TEMPORAL_LABELS", ("DATE", "TIME"))
)

#: Entity spans shorter than this after normalisation are discarded as noise.
MIN_ENTITY_CHARS: int = getattr(config, "MIN_ENTITY_CHARS", 2)


# --------------------------------------------------------------------------
# Surface form normalisation
# --------------------------------------------------------------------------

_LEADING_DET = re.compile(r"^(the|a|an|The|A|An)\s+")
_POSSESSIVE = re.compile(r"['\u2019]s$")
_WHITESPACE = re.compile(r"\s+")
_EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def normalise_surface(text: str, label: str | None = None) -> str:
    """Reduce an entity mention to a comparable surface form.

    Collapses internal whitespace (entity spans can straddle the newline tokens
    our paragraph handling leaves in the text), strips a leading article,
    removes a trailing possessive, and trims edge punctuation.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE.sub(" ", text).strip()
    text = _LEADING_DET.sub("", text)
    text = _POSSESSIVE.sub("", text)
    if label in TEMPORAL_LABELS:
        return text.strip()
    text = _EDGE_PUNCT.sub("", text)
    return text.strip()


#: Alias map for the domain.  Keys are lowercased normalised surface forms.
#: Without this the entity table reports NASA three times under three spellings
#: and the relation table splits the same organisation across three nodes.
ALIASES: dict[str, str] = {}


def _register_aliases(canonical: str, *variants: str) -> None:
    ALIASES[normalise_surface(canonical).lower()] = canonical
    for variant in variants:
        ALIASES[normalise_surface(variant).lower()] = canonical


_register_aliases("NASA", "N.A.S.A.",
                  "National Aeronautics and Space Administration",
                  "US National Aeronautics and Space Administration")
_register_aliases("ESA", "European Space Agency")
_register_aliases("ISRO", "Indian Space Research Organisation",
                  "Indian Space Research Organization")
_register_aliases("JAXA", "Japan Aerospace Exploration Agency")
_register_aliases("Roscosmos", "Russian Federal Space Agency",
                  "Russian Space Agency", "Rosaviakosmos")
_register_aliases("CNSA", "China National Space Administration")
_register_aliases("SpaceX", "Space Exploration Technologies",
                  "Space Exploration Technologies Corp")
_register_aliases("JPL", "Jet Propulsion Laboratory",
                  "NASA Jet Propulsion Laboratory")
_register_aliases("ULA", "United Launch Alliance")
_register_aliases("ISS", "International Space Station", "Space Station")
_register_aliases("Hubble Space Telescope", "Hubble", "HST")
_register_aliases("James Webb Space Telescope", "JWST", "Webb Telescope", "Webb")
_register_aliases("United States", "U.S.", "US", "USA", "U.S.A.",
                  "United States of America")
_register_aliases("United Kingdom", "U.K.", "UK", "Britain", "Great Britain")
_register_aliases("Russia", "Russian Federation")
_register_aliases("Mars Orbiter Mission", "Mangalyaan")
_register_aliases("European Southern Observatory", "ESO")


def canonicalise(text: str, label: str | None = None) -> str:
    """Map a normalised surface form onto its canonical name."""
    surface = normalise_surface(text, label)
    return ALIASES.get(surface.lower(), surface)


def entity_key(canonical: str, label: str) -> str:
    """Stable identifier for an entity across the whole corpus.

    Label is part of the key on purpose: "Mercury" the planet and "Mercury" the
    crewed programme are different things and must not merge into one node in
    the Stage 4 graph.
    """
    return f"{label}:{canonical}"


def is_temporal(label: str) -> bool:
    return label in TEMPORAL_LABELS


# --------------------------------------------------------------------------
# Mention extraction
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityMention:
    """One occurrence of an entity in one document."""
    doc_id: str
    sent_id: int
    mention: str          # surface text as it appears
    canonical: str        # alias-resolved, article-stripped
    entity_key: str       # "LABEL:Canonical"
    label: str
    is_temporal: bool
    start_char: int
    end_char: int
    start_token: int
    end_token: int        # exclusive
    head_token: int       # syntactic head, needed by relation extraction
    sentence: str

    def as_row(self) -> dict:
        return asdict(self)


def _sentence_index(doc: Doc) -> dict[int, int]:
    """Map token index -> sentence ordinal."""
    index: dict[int, int] = {}
    if not doc.has_annotation("SENT_START"):
        return index
    for sent_id, sent in enumerate(doc.sents):
        for token in sent:
            index[token.i] = sent_id
    return index


def keep_entity(ent: Span, labels: Iterable[str] = ENTITY_LABELS) -> bool:
    if ent.label_ not in labels:
        return False
    return len(normalise_surface(ent.text, ent.label_)) >= MIN_ENTITY_CHARS


def iter_mentions(doc: Doc, *, labels: Iterable[str] = ENTITY_LABELS
                  ) -> Iterator[EntityMention]:
    """Yield one :class:`EntityMention` per kept entity span in ``doc``."""
    doc_id = doc._.doc_id or "?"
    sent_of = _sentence_index(doc)
    label_set = tuple(labels)

    for ent in doc.ents:
        if not keep_entity(ent, label_set):
            continue
        canonical = canonicalise(ent.text, ent.label_)
        if not canonical:
            continue
        sent = ent.sent if doc.has_annotation("SENT_START") else None
        yield EntityMention(
            doc_id=doc_id,
            sent_id=sent_of.get(ent.start, -1),
            mention=_WHITESPACE.sub(" ", ent.text).strip(),
            canonical=canonical,
            entity_key=entity_key(canonical, ent.label_),
            label=ent.label_,
            is_temporal=is_temporal(ent.label_),
            start_char=ent.start_char,
            end_char=ent.end_char,
            start_token=ent.start,
            end_token=ent.end,
            head_token=ent.root.i,
            sentence=_WHITESPACE.sub(" ", sent.text).strip() if sent else "",
        )


def mention_rows(docs: Iterable[Doc], *, labels: Iterable[str] = ENTITY_LABELS
                 ) -> Iterator[dict]:
    for doc in docs:
        for mention in iter_mentions(doc, labels=labels):
            yield mention.as_row()


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def summarise_entities(rows: Iterable[dict]) -> list[dict]:
    """Collapse mentions into one row per distinct entity.

    Reports how many surface forms each entity appeared under, which is the
    quickest way to spot an alias that still needs adding to :data:`ALIASES`.
    """
    grouped: dict[str, dict] = {}
    for row in rows:
        key = row["entity_key"]
        entry = grouped.setdefault(key, {
            "entity_key": key,
            "canonical": row["canonical"],
            "label": row["label"],
            "is_temporal": row["is_temporal"],
            "n_mentions": 0,
            "_docs": set(),
            "_forms": Counter(),
        })
        entry["n_mentions"] += 1
        entry["_docs"].add(row["doc_id"])
        entry["_forms"][row["mention"]] += 1

    summary = []
    for entry in grouped.values():
        forms = entry.pop("_forms")
        docs = entry.pop("_docs")
        entry["n_documents"] = len(docs)
        entry["n_surface_forms"] = len(forms)
        entry["surface_forms"] = " | ".join(f for f, _ in forms.most_common(5))
        summary.append(entry)

    summary.sort(key=lambda e: (-e["n_mentions"], e["canonical"]))
    return summary


# --------------------------------------------------------------------------
# POS
# --------------------------------------------------------------------------

def pos_counter(docs: Iterable[Doc]) -> Counter:
    """Corpus-level (coarse POS, fine tag) counts, excluding whitespace."""
    counts: Counter = Counter()
    for doc in docs:
        for token in doc:
            if token.is_space:
                continue
            counts[(token.pos_, token.tag_)] += 1
    return counts


def pos_rows(counts: Counter, *, examples: dict | None = None) -> list[dict]:
    total = sum(counts.values()) or 1
    rows = []
    for (pos, tag), count in counts.most_common():
        row = {
            "pos": pos,
            "tag": tag,
            "count": count,
            "pct_of_tokens": round(100 * count / total, 3),
        }
        if examples is not None:
            sample = examples.get((pos, tag), [])
            row["examples"] = ", ".join(sample[:5])
        rows.append(row)
    return rows


def pos_examples(docs: Iterable[Doc], per_tag: int = 5) -> dict:
    """Collect a few representative word forms per (POS, tag) pair."""
    found: dict[tuple[str, str], list[str]] = {}
    for doc in docs:
        for token in doc:
            if token.is_space:
                continue
            key = (token.pos_, token.tag_)
            bucket = found.setdefault(key, [])
            form = token.text.strip()
            if form and len(bucket) < per_tag and form not in bucket:
                bucket.append(form)
    return found


# --------------------------------------------------------------------------
# CoNLL-style token export
# --------------------------------------------------------------------------

def conll_rows(doc: Doc) -> Iterator[dict]:
    """Per-token rows in a CoNLL-like shape, with BIO entity tags."""
    doc_id = doc._.doc_id or "?"
    sent_of = _sentence_index(doc)
    bio = {}
    for ent in doc.ents:
        for offset, token in enumerate(ent):
            bio[token.i] = ("B-" if offset == 0 else "I-") + ent.label_

    for token in doc:
        if token.is_space:
            continue
        yield {
            "doc_id": doc_id,
            "sent_id": sent_of.get(token.i, -1),
            "token_id": token.i,
            "token": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "head_id": token.head.i,
            "ner": bio.get(token.i, "O"),
        }


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

#: displacy colours for the custom labels, so the gazetteer's contribution is
#: visible at a glance in the annotated HTML.
DISPLACY_COLORS = {
    "SPACECRAFT": "#7aecec",
    "CELESTIAL": "#ffd8b5",
    "ORG": "#bfe1d9",
    "FAC": "#c887fb",
    "DATE": "#f6c1c1",
    "TIME": "#f6c1c1",
}


def displacy_options(labels: Iterable[str] = ENTITY_LABELS) -> dict:
    return {"ents": list(labels), "colors": DISPLACY_COLORS}


def load_docs(doc_ids: Iterable[str] | None = None) -> Iterator[Doc]:
    """Convenience re-export so callers need only import this module."""
    return enlp.iter_cached_docs(doc_ids)
