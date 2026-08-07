"""
scripts/extract_entities.py  --  Entropy Stage 3a: POS tagging and NER
======================================================================

Reads the Stage 2 annotation cache and writes the entity-layer deliverables.
No spaCy pipeline is run here; if the numbers look wrong, the fix belongs in
preprocess.py or the gazetteer, not in this script.

Outputs
-------
  outputs/entities.csv            one row per entity mention, with offsets
  outputs/entity_summary.csv      one row per distinct entity, with counts
  outputs/pos_summary.csv         corpus-level POS/tag distribution
  outputs/tokens_conll.tsv        per-token token/lemma/POS/dep/BIO export
  outputs/annotated/<doc>.html    displacy entity rendering, sample of docs

Usage
-----
    python scripts/extract_entities.py
    python scripts/extract_entities.py --limit 20
    python scripts/extract_entities.py --sample 30 --no-conll
    python scripts/extract_entities.py --doc-id wn-0042 --sample 1
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from spacy import displacy  # noqa: E402

from entropy import entities as E  # noqa: E402
from entropy import nlp as enlp  # noqa: E402
from entropy import gazetteer  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entropy Stage 3a: entity and POS extraction from cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N cached documents")
    p.add_argument("--doc-id", action="append", dest="doc_ids", default=None,
                   help="process only this doc_id (repeatable)")
    p.add_argument("--sample", type=int, default=20,
                   help="how many documents to render as annotated HTML")
    p.add_argument("--no-conll", action="store_true",
                   help="skip the per-token CoNLL export")
    p.add_argument("--no-html", action="store_true",
                   help="skip the displacy HTML rendering")
    p.add_argument("--outputs", type=Path, default=None,
                   help="output directory (default: outputs/)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    started = time.perf_counter()
    out_dir = args.outputs or enlp.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids = args.doc_ids or enlp.cached_doc_ids()
    if not doc_ids:
        print("No cached documents found. Run: python scripts/preprocess.py")
        return 1
    if args.limit is not None:
        doc_ids = doc_ids[: args.limit]

    print("Entropy Stage 3a: entities and POS")
    print(f"  cache dir  : {enlp.CACHE_DIR}")
    print(f"  outputs    : {out_dir}")
    print(f"  documents  : {len(doc_ids)}")
    print(f"  labels kept: {', '.join(E.ENTITY_LABELS)}")
    print(f"  gazetteer  : {gazetteer.pattern_count()} patterns")
    print()

    mention_rows: list[dict] = []
    conll_frames: list[dict] = []
    pos_counts: Counter = Counter()
    pos_seen: dict = {}
    html_written = 0
    docs_seen = 0
    docs_without_ents: list[str] = []

    html_dir = out_dir / "annotated"
    if not args.no_html:
        html_dir.mkdir(parents=True, exist_ok=True)

    for doc in enlp.iter_cached_docs(doc_ids):
        docs_seen += 1
        doc_id = doc._.doc_id or f"doc-{docs_seen}"

        before = len(mention_rows)
        mention_rows.extend(m.as_row() for m in E.iter_mentions(doc))
        if len(mention_rows) == before:
            docs_without_ents.append(doc_id)

        for token in doc:
            if token.is_space:
                continue
            key = (token.pos_, token.tag_)
            pos_counts[key] += 1
            bucket = pos_seen.setdefault(key, [])
            form = token.text.strip()
            if form and len(bucket) < 5 and form not in bucket:
                bucket.append(form)

        if not args.no_conll:
            conll_frames.extend(E.conll_rows(doc))

        if not args.no_html and html_written < args.sample:
            html = displacy.render(
                doc, style="ent", page=True, options=E.displacy_options()
            )
            title = (doc._.title or doc_id).replace("\n", " ")
            html = html.replace(
                "<body ",
                f"<body data-doc-id='{doc_id}' data-title='{title}' ",
                1,
            )
            (html_dir / f"{doc_id}.html").write_text(html, encoding="utf-8")
            html_written += 1

        if docs_seen % 100 == 0:
            print(f"  read {docs_seen}/{len(doc_ids)}")

    # ---- entity tables ---------------------------------------------------
    mentions = pd.DataFrame(mention_rows)
    mentions_path = out_dir / "entities.csv"
    mentions.to_csv(mentions_path, index=False, encoding="utf-8")

    summary = pd.DataFrame(E.summarise_entities(mention_rows))
    summary_path = out_dir / "entity_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    # ---- POS -------------------------------------------------------------
    pos_frame = pd.DataFrame(E.pos_rows(pos_counts, examples=pos_seen))
    pos_path = out_dir / "pos_summary.csv"
    pos_frame.to_csv(pos_path, index=False, encoding="utf-8")

    # ---- CoNLL -----------------------------------------------------------
    conll_path = None
    if conll_frames:
        conll_path = out_dir / "tokens_conll.tsv"
        pd.DataFrame(conll_frames).to_csv(
            conll_path, sep="\t", index=False, encoding="utf-8"
        )

    # ---- summary ---------------------------------------------------------
    elapsed = time.perf_counter() - started
    print()
    print(f"Entities   : {len(mentions):,} mentions -> {mentions_path.name}")
    print(f"Distinct   : {len(summary):,} entities  -> {summary_path.name}")
    print(f"POS        : {int(pos_frame['count'].sum()):,} tokens over "
          f"{pos_frame['pos'].nunique()} coarse tags -> {pos_path.name}")
    if conll_path:
        print(f"CoNLL      : {len(conll_frames):,} rows -> {conll_path.name}")
    if not args.no_html:
        print(f"Annotated  : {html_written} HTML file(s) -> {html_dir}")

    if not mentions.empty:
        print()
        print("Label distribution:")
        counts = mentions["label"].value_counts()
        for label, count in counts.items():
            marker = "  <- gazetteer" if label in gazetteer.CUSTOM_LABELS else ""
            print(f"  {label:<12} {count:>6,}{marker}")

        print()
        print("Top entities:")
        for _, row in summary.head(12).iterrows():
            print(f"  {row['n_mentions']:>5,}  {row['label']:<11} "
                  f"{row['canonical']}")

        multi = summary[summary["n_surface_forms"] > 1]
        if not multi.empty:
            print()
            print(f"{len(multi)} entities appear under multiple surface forms. "
                  "Check the top few for aliases worth adding to "
                  "entities.ALIASES:")
            for _, row in multi.head(5).iterrows():
                print(f"  {row['canonical']:<28} {row['surface_forms']}")

    if docs_without_ents:
        print()
        print(f"[warn] {len(docs_without_ents)} document(s) yielded no entities: "
              f"{', '.join(docs_without_ents[:5])}")

    print()
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
