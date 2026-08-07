"""
Central configuration for Entropy.

Everything that a later stage might want to tune lives here, so that no other
module ever hard-codes a path or a magic number. Paths are built with pathlib
and are always derived from PROJECT_ROOT, which means the same code runs
unchanged on Windows (your machine) and Linux (Streamlit Community Cloud).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# __file__            -> .../Entropy/entropy/config.py
# .parents[0]         -> .../Entropy/entropy
# .parents[1]         -> .../Entropy          <- project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"        # git-ignored: big, licensed source dumps
CORPUS_DIR: Path = DATA_DIR / "corpus"  # committed: the small frozen slice
EVAL_DIR: Path = DATA_DIR / "eval"      # git-ignored: TimeBank / AQUAINT gold data
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"  # git-ignored: generated artefacts

# Input: the raw Wikinews JSON you download once (see README).
RAW_WIKINEWS_JSON: Path = RAW_DIR / "wikinews.json"

# Output: the frozen, reproducible working corpus.
CORPUS_FILE: Path = CORPUS_DIR / "entropy_corpus.jsonl"
MANIFEST_FILE: Path = CORPUS_DIR / "manifest.json"


def ensure_dirs() -> None:
    """Create every directory Entropy writes to. Safe to call repeatedly."""
    for directory in (RAW_DIR, CORPUS_DIR, EVAL_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Corpus selection
# --------------------------------------------------------------------------
# Wikinews articles carry ~7 lowercase category labels each. We keep an article
# if ANY of its categories contains ANY of these substrings. Substring matching
# is deliberate: it catches "space exploration", "outer space", "space policy"
# from the single token "space".
THEME_NAME: str = "space-and-science"

THEME_CATEGORIES: list[str] = [
    "space",
    "astronomy",
    "nasa",
    "isro",
    "spacex",
    "european space agency",
    "roscosmos",
    "satellite",
    # "science and technology",   ← add the # in front of this line
]

# Articles matching any of these are dropped even if they matched above.
# Sports and obituaries pollute a science timeline with unrelated entities.
EXCLUDE_CATEGORIES: list[str] = [
    "sport",
    "obituaries",
    "football",
    "cricket",
]

# Very short articles have no room for relations or event chains, and Wikinews
# has a long tail of two-sentence stubs. 120 words is roughly 6-8 sentences.
MIN_WORDS: int = 120

# Hard cap. Keeps the committed corpus small and the Streamlit app responsive.
MAX_DOCS: int = 500

# Wikinews ran from 2004 until it was frozen on 2026-05-04.
DATE_MIN: str = "2005-01-01"
DATE_MAX: str = "2026-05-04"

# --------------------------------------------------------------------------
# Provenance (copied into manifest.json so the corpus is self-describing)
# --------------------------------------------------------------------------
SOURCE_NAME: str = "Wikinews (English)"
SOURCE_URL: str = "https://en.wikinews.org/"
SOURCE_LICENSE: str = "CC BY 2.5"

CACHE_DIR = CORPUS_DIR / "cache"          # per-document .spacy annotations
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPACY_MODEL = "en_core_web_sm"            # NOT trf — Streamlit Cloud RAM ceiling
SPACY_BATCH_SIZE = 32

USE_GAZETTEER = True
ENTITY_LABELS = (
    "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
    "SPACECRAFT", "CELESTIAL", "DATE", "TIME",
)
TEMPORAL_LABELS = ("DATE", "TIME")
MIN_ENTITY_CHARS = 2
