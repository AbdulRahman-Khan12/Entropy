"""Stage 5 driver - evaluation.

Three things, in the order you would actually do them:

1. ``--make-sample``  writes annotation sheets for precision estimation.
   Open them, put y or n in the ``correct`` column, save.
2. ``--make-gold``    writes an exhaustive candidate sheet over a handful of
   documents, for full precision/recall/F1.
3. no flag            scores whatever you have annotated and writes the
   evaluation CSVs. Rule-vs-HF agreement is computed either way, since it
   needs no annotation at all.

    python scripts/evaluate.py --make-sample --per-type 15
    python scripts/evaluate.py --make-gold --docs 8
    python scripts/evaluate.py

Annotation sheets live in data/gold/ (committed - they are hand-made and
expensive). Scores go to outputs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entropy.evaluation import (  # noqa: E402
    candidate_gold_rows,
    compare_sources,
    event_key,
    precision_from_sample,
    relation_key,
    score_against_gold,
    stratified_sample,
    write_rows_csv,
    write_scores_csv,
)
from entropy.relations import RELATION_SIGNATURES, output_dir  # noqa: E402

GOLD_DIR_NAME = Path("data") / "gold"

RELATION_SAMPLE_FIELDS = [
    "correct", "doc_id", "sent_id", "relation", "subj", "obj",
    "trigger", "pattern", "confidence", "source", "subj_key", "obj_key", "sentence",
]
EVENT_SAMPLE_FIELDS = [
    "correct", "doc_id", "sent_id", "event_type", "trigger_word", "agent",
    "patient", "time_expr", "negated", "confidence", "trigger_lemma", "sentence",
]
GOLD_RELATION_FIELDS = [
    "relation", "doc_id", "sent_id", "subj_key", "obj_key", "possible", "sentence",
]
GOLD_EVENT_FIELDS = [
    "event_type", "doc_id", "sent_id", "trigger_lemma", "agent", "patient", "sentence",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 5: evaluate the extraction pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--gold-dir", type=Path, default=None)
    parser.add_argument("--make-sample", action="store_true",
                        help="write precision-sampling sheets and exit")
    parser.add_argument("--make-gold", action="store_true",
                        help="write exhaustive gold candidate sheets and exit")
    parser.add_argument("--per-type", type=int, default=15,
                        help="rows sampled per relation/event type")
    parser.add_argument("--docs", type=int, default=8,
                        help="documents to annotate exhaustively for --make-gold")
    parser.add_argument("--mode", choices=["strict", "relaxed"], default="strict",
                        help="strict matches the sentence too; relaxed does not")
    parser.add_argument("--seed", type=int, default=20250809)
    parser.add_argument("--force", action="store_true",
                        help="overwrite annotation sheets that already exist")
    return parser.parse_args(argv)


def read_csv(path: Path, required: bool = True) -> list[dict]:
    if not path.exists():
        if required:
            raise SystemExit(f"Missing {path}. Run the earlier stages first.")
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _guard(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"  exists, not overwriting: {path.name}  (use --force)")
        return False
    return True


def make_samples(args, out_dir: Path, gold_dir: Path) -> int:
    relations = read_csv(out_dir / "relations.csv")
    events = read_csv(out_dir / "events.csv")
    rule_relations = [r for r in relations if r.get("source", "rule") == "rule"]

    print(f"Precision sampling: {args.per_type} rows per type\n")
    relation_path = gold_dir / "sample_relations.csv"
    if _guard(relation_path, args.force):
        sample = stratified_sample(rule_relations, "relation", args.per_type, args.seed)
        for row in sample:
            row["correct"] = ""
        write_rows_csv(sample, relation_path, RELATION_SAMPLE_FIELDS)
        print(f"  wrote {relation_path}  ({len(sample)} rows)")

    event_path = gold_dir / "sample_events.csv"
    if _guard(event_path, args.force):
        sample = stratified_sample(events, "event_type", args.per_type, args.seed)
        for row in sample:
            row["correct"] = ""
        write_rows_csv(sample, event_path, EVENT_SAMPLE_FIELDS)
        print(f"  wrote {event_path}  ({len(sample)} rows)")

    print("\nNow open both files and fill the 'correct' column with y or n.")
    print("A relation is correct when the sentence really asserts it.")
    print("An event is correct when the trigger really denotes that event type")
    print("and the agent/patient are not wrong.")
    print("\nThen run:  python scripts/evaluate.py")
    return 0


def make_gold(args, out_dir: Path, gold_dir: Path) -> int:
    entities = read_csv(out_dir / "entities.csv")
    if not entities:
        raise SystemExit("entities.csv is required for --make-gold (Stage 3).")

    doc_ids = sorted({row["doc_id"] for row in entities})
    chosen = random.Random(args.seed).sample(doc_ids, min(args.docs, len(doc_ids)))
    print(f"Exhaustive gold over {len(chosen)} document(s): {', '.join(chosen)}\n")

    relation_path = gold_dir / "gold_relations.csv"
    if _guard(relation_path, args.force):
        rows = candidate_gold_rows(entities, RELATION_SIGNATURES, chosen)
        write_rows_csv(rows, relation_path, GOLD_RELATION_FIELDS)
        print(f"  wrote {relation_path}  ({len(rows)} candidate pairs)")

    event_path = gold_dir / "gold_events.csv"
    if _guard(event_path, args.force):
        sentences: dict[tuple[str, str], str] = {}
        for row in entities:
            if row["doc_id"] in chosen:
                sentences.setdefault(
                    (row["doc_id"], str(row["sent_id"])), row.get("sentence", "")
                )
        rows = [
            {"event_type": "", "doc_id": doc_id, "sent_id": sent_id,
             "trigger_lemma": "", "agent": "", "patient": "", "sentence": text}
            for (doc_id, sent_id), text in sorted(sentences.items())
        ]
        write_rows_csv(rows, event_path, GOLD_EVENT_FIELDS)
        print(f"  wrote {event_path}  ({len(rows)} sentences)")

    (gold_dir / "chosen_docs.json").write_text(
        json.dumps(chosen, indent=2), encoding="utf-8")

    print("\nRelations: write a relation name in the 'relation' column, or NONE.")
    print("           The 'possible' column lists what the schema allows.")
    print("Events:    one row per event in the sentence - duplicate the row if a")
    print("           sentence has two events, delete it if it has none.")
    print("\nThen run:  python scripts/evaluate.py")
    return 0


def _print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    print(f"\n{title}")
    widths = [max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns]
    print("  " + "  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print("  " + "  ".join(str(row.get(c, "")).ljust(w)
                               for c, w in zip(columns, widths)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir or output_dir()
    repo_root = Path(__file__).resolve().parents[1]
    gold_dir = args.gold_dir or (repo_root / GOLD_DIR_NAME)
    gold_dir.mkdir(parents=True, exist_ok=True)

    if args.make_sample:
        return make_samples(args, out_dir, gold_dir)
    if args.make_gold:
        return make_gold(args, out_dir, gold_dir)

    relations = read_csv(out_dir / "relations.csv")
    events = read_csv(out_dir / "events.csv")
    written: list[Path] = []
    summary: dict = {"mode": args.mode}

    # ------------------------------------------------ rule vs HF (no gold)
    agreement = compare_sources(relations, mode=args.mode)
    if any(row["hf"] for row in agreement):
        written.append(write_rows_csv(
            agreement, out_dir / "evaluation_agreement.csv",
            ["relation", "rule", "hf", "agreed", "rule_only", "hf_only",
             "jaccard", "hf_recall_of_rule"],
        ))
        _print_table("Rule vs Hugging Face agreement (no gold needed)", agreement,
                     ["relation", "rule", "hf", "agreed", "rule_only", "hf_only",
                      "jaccard"])
        summary["agreement"] = agreement
    else:
        print("No HF rows in relations.csv - run extract_relations.py --hf "
              "for the baseline comparison.")

    # -------------------------------------------------- precision sampling
    for name, label_field, fields_out in (
        ("relations", "relation", "evaluation_precision_relations.csv"),
        ("events", "event_type", "evaluation_precision_events.csv"),
    ):
        sample = read_csv(gold_dir / f"sample_{name}.csv", required=False)
        if not sample:
            continue
        results, unjudged = precision_from_sample(sample, label_field)
        if not results or results[-1]["judged"] == 0:
            print(f"\nsample_{name}.csv found but nothing judged yet - fill the "
                  f"'correct' column with y/n.")
            continue
        written.append(write_rows_csv(
            results, out_dir / fields_out,
            ["label", "judged", "correct", "precision", "ci_low", "ci_high"],
        ))
        _print_table(
            f"Precision on a hand-judged sample - {name}"
            + (f"  ({unjudged} rows still unjudged)" if unjudged else ""),
            results, ["label", "judged", "correct", "precision", "ci_low", "ci_high"],
        )
        summary[f"precision_{name}"] = results

    # ------------------------------------------------------ gold scoring
    gold_relations = read_csv(gold_dir / "gold_relations.csv", required=False)
    if gold_relations and any((r.get("relation") or "").strip() for r in gold_relations):
        rule_relations = [r for r in relations if r.get("source", "rule") == "rule"]
        scores = score_against_gold(
            gold_relations, rule_relations, relation_key, "relation", args.mode)
        written.append(write_scores_csv(scores, out_dir / "evaluation_relations.csv"))
        _print_table("Relations vs gold", [vars(s) for s in scores],
                     ["label", "tp", "fp", "fn", "precision", "recall", "f1",
                      "support"])
        summary["relations_vs_gold"] = [vars(s) for s in scores]

    gold_events = read_csv(gold_dir / "gold_events.csv", required=False)
    if gold_events and any((r.get("event_type") or "").strip() for r in gold_events):
        scores = score_against_gold(
            gold_events, events, event_key, "event_type", args.mode)
        written.append(write_scores_csv(scores, out_dir / "evaluation_events.csv"))
        _print_table("Events vs gold", [vars(s) for s in scores],
                     ["label", "tp", "fp", "fn", "precision", "recall", "f1",
                      "support"])
        summary["events_vs_gold"] = [vars(s) for s in scores]

    if len(summary) == 1:
        print("\nNothing annotated yet. Start with:\n"
              "  python scripts/evaluate.py --make-sample")
        return 0

    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(summary_path)
    print()
    for path in written:
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
