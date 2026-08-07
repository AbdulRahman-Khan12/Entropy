"""
The Entropy corpus: data model and loader.

Every later stage (POS tagging, NER, relation extraction, event extraction,
temporal ordering, the Streamlit app) reads documents through this module and
never touches the JSONL file directly. That keeps the on-disk format free to
change without breaking six other files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Iterator

from entropy import config


@dataclass(frozen=True)
class Document:
    """A single news article, with the metadata the pipeline needs.

    Attributes
    ----------
    doc_id:
        Stable identifier, e.g. "wn-0042". Used as the primary key everywhere
        downstream, including in entity and relation tables.
    title:
        Article headline.
    text:
        Plain-text body. Paragraphs separated by blank lines.
    dct:
        Document Creation Time. This is the anchor that lets Stage 5 resolve
        relative expressions such as "on Tuesday" or "three days later" into
        real calendar dates. Without it, temporal ordering is impossible.
    categories:
        Lowercase Wikinews category labels.
    """

    doc_id: str
    title: str
    text: str
    dct: date
    categories: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    @property
    def dct_iso(self) -> str:
        return self.dct.isoformat()

    def to_dict(self) -> dict:
        """JSON-serialisable form (date -> ISO string, tuple -> list)."""
        payload = asdict(self)
        payload["dct"] = self.dct_iso
        payload["categories"] = list(self.categories)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "Document":
        return cls(
            doc_id=payload["doc_id"],
            title=payload["title"],
            text=payload["text"],
            dct=date.fromisoformat(payload["dct"]),
            categories=tuple(payload.get("categories", ())),
        )


def iter_corpus(path: Path | None = None) -> Iterator[Document]:
    """Stream documents one at a time. Constant memory, whatever the size."""
    path = path or config.CORPUS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}.\n"
            "Run:  python -m scripts.build_corpus\n"
            "(See README.md for how to obtain the raw Wikinews JSON first.)"
        )
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield Document.from_dict(json.loads(line))


def load_corpus(path: Path | None = None) -> list[Document]:
    """Load the whole corpus into memory. Fine at our size (<= 500 docs)."""
    return list(iter_corpus(path))


def load_manifest(path: Path | None = None) -> dict:
    """Read the build manifest: counts, date range, checksum, provenance."""
    path = path or config.MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def summarise(documents: list[Document]) -> str:
    """Human-readable one-paragraph description of a document collection."""
    if not documents:
        return "Empty corpus."
    dates = [doc.dct for doc in documents]
    words = sum(doc.n_words for doc in documents)
    return (
        f"{len(documents)} documents | "
        f"{dates and min(dates)} to {max(dates)} | "
        f"{words:,} words | "
        f"{words // len(documents):,} words/doc average"
    )
