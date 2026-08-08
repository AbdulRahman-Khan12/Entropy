"""Stage 4 driver - relation and event extraction.

Reads ONLY the Stage 2/3 annotation cache. This script never calls
``nlp(text)``; Docs come back from ``entropy.nlp.iter_cached_docs`` already
tagged, parsed and NER'd, so a full 500-doc run needs no model load beyond
``spacy.blank("en").vocab``.

Examples
--------
    python scripts/extract_relations.py                     # full 500-doc run
    python scripts/extract_relations.py --limit 25          # quick smoke test
    python scripts/extract_relations.py --doc-id wn-0042    # one document
    python scripts/extract_relations.py --min-confidence 0.6
    python scripts/extract_relations.py --hf --hf-limit 50  # + HF baseline

Outputs (all under outputs/, gitignored):
    relations.csv, events.csv, relation_summary.csv, event_summary.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow "python scripts/extract_relations.py" from the repo root without an
# editable install, matching the other Stage 2/3 scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entropy import nlp as nlp_cache  # noqa: E402  (path shim must run first)
from entropy.entities import iter_mentions  # noqa: E402
from entropy.events import (  # noqa: E402
    extract_events_from_sentence,
    summarise_events,
    write_event_summary_csv,
    write_events_csv,
)
from entropy.relations import (  # noqa: E402
    HF_DEFAULT_MODEL,
    HFRelationBaseline,
    build_sentences,
    extract_relations_from_sentence,
    output_dir,
    summarise_relations,
    write_relation_summary_csv,
    write_relations_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4: relation and event extraction from the spaCy cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N cached documents")
    parser.add_argument("--doc-id", action="append", dest="doc_ids", default=None,
                        metavar="wn-XXXX", help="process one document (repeatable)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="override the outputs directory")

    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="drop relations below this rule confidence")
    parser.add_argument("--no-fallback", action="store_true",
                        help="disable the low-confidence dependency-path rule")
    parser.add_argument("--no-relations", action="store_true",
                        help="skip relation extraction")
    parser.add_argument("--no-events", action="store_true",
                        help="skip event extraction")
    parser.add_argument("--entities-only-args", action="store_true",
                        help="event agent/patient must be a Stage 3 entity mention")

    parser.add_argument("--hf", action="store_true",
                        help="also run the Hugging Face zero-shot baseline")
    parser.add_argument("--hf-model", default=HF_DEFAULT_MODEL,
                        help="distilled HF model for the baseline")
    parser.add_argument("--hf-threshold", type=float, default=0.5,
                        help="minimum entailment score to keep an HF relation")
    parser.add_argument("--hf-limit", type=int, default=50,
                        help="cap HF to the first N documents (CPU inference is slow)")
    parser.add_argument("--hf-max-pairs", type=int, default=12,
                        help="cap candidate pairs per sentence for the HF baseline")

    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser.parse_args(argv)


def select_doc_ids(args: argparse.Namespace) -> list[str]:
    available = nlp_cache.cached_doc_ids()
    if not available:
        raise SystemExit(
            "No cached documents found. Run scripts/preprocess.py first (Stage 2)."
        )
    if args.doc_ids:
        wanted, missing = [], []
        known = set(available)
        for doc_id in args.doc_ids:
            (wanted if doc_id in known else missing).append(doc_id)
        if missing:
            raise SystemExit(f"Not in the cache: {', '.join(missing)}")
        return wanted
    if args.limit is not None:
        return available[: args.limit]
    return available


def _print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    print(f"\n{title}")
    widths = [
        max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in columns
    ]
    print("  " + "  ".join(col.ljust(w) for col, w in zip(columns, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("  " + "  ".join(str(row.get(col, "")).ljust(w)
                               for col, w in zip(columns, widths)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    doc_ids = select_doc_ids(args)
    out_dir = args.out_dir or output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Stage 4: {len(doc_ids)} document(s) from the cache -> {out_dir}")

    all_contexts_for_hf: list = []
    relations: list = []
    events: list = []
    n_sentences = 0
    n_mentions = 0

    for index, (doc_id, doc) in enumerate(
        zip(doc_ids, nlp_cache.iter_cached_docs(doc_ids)), start=1
    ):
        mentions = list(iter_mentions(doc))
        n_mentions += len(mentions)
        contexts = build_sentences(doc, mentions, doc_id=doc_id)
        n_sentences += len(contexts)

        for ctx in contexts:
            if not args.no_relations and len(ctx.mentions) >= 2:
                relations.extend(
                    extract_relations_from_sentence(
                        ctx, use_fallback=not args.no_fallback
                    )
                )
            if not args.no_events:
                events.extend(
                    extract_events_from_sentence(
                        ctx, resolve_noun_args=not args.entities_only_args
                    )
                )

        if args.hf and index <= args.hf_limit:
            all_contexts_for_hf.extend(c for c in contexts if len(c.mentions) >= 2)

        if not args.quiet and index % 50 == 0:
            print(f"  {index:>4}/{len(doc_ids)} docs | "
                  f"{len(relations)} relations | {len(events)} events")

    if args.min_confidence > 0:
        before = len(relations)
        relations = [r for r in relations if r.confidence >= args.min_confidence]
        if not args.quiet:
            print(f"  confidence filter: {before} -> {len(relations)} relations")

    hf_relations: list = []
    if args.hf and all_contexts_for_hf:
        if not args.quiet:
            print(f"\nHF baseline: {args.hf_model} over "
                  f"{len(all_contexts_for_hf)} sentence(s) from "
                  f"{min(args.hf_limit, len(doc_ids))} doc(s) - CPU, be patient.")
        try:
            baseline = HFRelationBaseline(
                model_name=args.hf_model,
                threshold=args.hf_threshold,
                max_pairs_per_sentence=args.hf_max_pairs,
            )
            hf_relations = list(baseline.predict(all_contexts_for_hf))
        except ImportError:
            print("  transformers is not installed; skipping the HF baseline.\n"
                  "  pip install \"transformers>=4.40\" torch --index-url "
                  "https://download.pytorch.org/whl/cpu", file=sys.stderr)
        except Exception as error:  # keep the rule-based run's results
            print(f"  HF baseline failed ({error.__class__.__name__}: {error}); "
                  f"rule-based output is unaffected.", file=sys.stderr)

    # ---------------------------------------------------------------- outputs
    written: list[Path] = []
    if not args.no_relations:
        combined = relations + hf_relations
        written.append(write_relations_csv(combined, out_dir / "relations.csv"))
        summary = summarise_relations(relations, hf_relations)
        written.append(write_relation_summary_csv(summary, out_dir / "relation_summary.csv"))
        if not args.quiet:
            columns = ["relation", "rule_count", "rule_distinct_pairs",
                       "rule_mean_confidence"]
            if hf_relations:
                columns += ["hf_count", "agreed_count", "rule_only_count",
                            "hf_only_count", "jaccard"]
            _print_table("Relations", summary, columns)

    if not args.no_events:
        written.append(write_events_csv(events, out_dir / "events.csv"))
        event_summary = summarise_events(events)
        written.append(write_event_summary_csv(event_summary, out_dir / "event_summary.csv"))
        if not args.quiet:
            _print_table(
                "Events", event_summary,
                ["event_type", "count", "docs", "with_agent", "with_patient",
                 "with_time", "negated"],
            )

    if not args.quiet:
        elapsed = time.perf_counter() - started
        print(f"\n{len(doc_ids)} docs | {n_sentences} sentences | {n_mentions} mentions")
        print(f"{len(relations)} rule relations"
              + (f" | {len(hf_relations)} HF relations" if hf_relations else "")
              + f" | {len(events)} events  ({elapsed:.1f}s)")
        for path in written:
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
