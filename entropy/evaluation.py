"""Stage 5 - evaluation.

Two evaluation modes, because they cost very different amounts of annotation
effort and answer different questions.

**Precision sampling** (cheap). Draw a stratified sample of what the system
extracted, judge each row correct or incorrect, and report precision with a
Wilson confidence interval. Needs no exhaustive annotation, but says nothing
about recall - you cannot see what the system missed.

**Gold evaluation** (expensive). Exhaustively annotate every candidate in a
sample of sentences, then score precision, recall and F1. The candidate set is
generated with the same type signatures the extractor uses, so recall here means
"of the relations expressible in this schema, how many did we find" - it does
not count relations the schema cannot represent. That limitation belongs in the
report.

Nothing here imports spaCy: evaluation runs off the CSVs.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Iterable, Sequence

__all__ = [
    "Score",
    "prf",
    "wilson_interval",
    "relation_key",
    "event_key",
    "score_against_gold",
    "precision_from_sample",
    "stratified_sample",
    "candidate_gold_rows",
    "compare_sources",
    "write_scores_csv",
]

NONE_LABELS = {"", "none", "no", "n", "-", "na", "n/a", "no_relation", "false", "0"}
YES_LABELS = {"y", "yes", "1", "true", "t", "correct", "ok"}


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Score:
    label: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    support: int          # gold instances
    predicted: int        # system instances


SCORE_FIELDS = tuple(f.name for f in fields(Score))


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - behaves sensibly at small n, unlike normal-approx.

    With 30 sampled relations and 24 correct, the naive interval can run past
    1.0; Wilson stays inside [0, 1], which matters at the sample sizes a course
    project can realistically annotate.
    """
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    spread = z * math.sqrt(
        (proportion * (1 - proportion) + z**2 / (4 * total)) / total
    )
    return (
        round(max(0.0, (centre - spread) / denominator), 4),
        round(min(1.0, (centre + spread) / denominator), 4),
    )


# --------------------------------------------------------------------------
# matching keys
# --------------------------------------------------------------------------
def relation_key(row: dict, mode: str = "strict") -> tuple:
    """Identity of a relation instance.

    strict  - same sentence, same typed argument pair
    relaxed - same document, ignores which sentence stated it (useful because a
              fact repeated across sentences is arguably one relation)
    """
    base = (row["relation"], row["subj_key"], row["obj_key"])
    if mode == "relaxed":
        return (row["doc_id"],) + base
    return (row["doc_id"], str(row.get("sent_id", ""))) + base


def event_key(row: dict, mode: str = "strict") -> tuple:
    """Identity of an event instance.

    Keyed on the trigger *lemma* rather than the token index, because a human
    annotator marks a word, not an offset.
    """
    lemma = (row.get("trigger_lemma") or row.get("trigger_word", "")).strip().lower()
    if mode == "relaxed":
        return (row["doc_id"], str(row.get("sent_id", "")), row["event_type"])
    return (row["doc_id"], str(row.get("sent_id", "")), row["event_type"], lemma)


# --------------------------------------------------------------------------
# gold evaluation
# --------------------------------------------------------------------------
def score_against_gold(
    gold: Sequence[dict],
    predicted: Sequence[dict],
    key_fn: Callable[[dict, str], tuple],
    label_field: str,
    mode: str = "strict",
    scope_field: str | None = "doc_id",
) -> list[Score]:
    """Precision / recall / F1, per label and micro-averaged.

    Only predictions inside the annotated scope are scored - otherwise every
    extraction from an unannotated document counts as a false positive and
    precision collapses for reasons that have nothing to do with the system.
    """
    gold_rows = [row for row in gold if _label_of(row, label_field)]
    if scope_field:
        scope = {row[scope_field] for row in gold}
        predicted = [row for row in predicted if row.get(scope_field) in scope]

    gold_keys: set[tuple] = set()
    gold_by_label: dict[str, set[tuple]] = defaultdict(set)
    for row in gold_rows:
        key = key_fn(row, mode)
        gold_keys.add(key)
        gold_by_label[_label_of(row, label_field)].add(key)

    predicted_keys: set[tuple] = set()
    predicted_by_label: dict[str, set[tuple]] = defaultdict(set)
    for row in predicted:
        key = key_fn(row, mode)
        predicted_keys.add(key)
        predicted_by_label[row[label_field]].add(key)

    scores: list[Score] = []
    for label in sorted(set(gold_by_label) | set(predicted_by_label)):
        gold_set = gold_by_label.get(label, set())
        predicted_set = predicted_by_label.get(label, set())
        tp = len(gold_set & predicted_set)
        fp = len(predicted_set - gold_set)
        fn = len(gold_set - predicted_set)
        precision, recall, f1 = prf(tp, fp, fn)
        scores.append(Score(label, tp, fp, fn, precision, recall, f1,
                            len(gold_set), len(predicted_set)))

    tp = len(gold_keys & predicted_keys)
    fp = len(predicted_keys - gold_keys)
    fn = len(gold_keys - predicted_keys)
    precision, recall, f1 = prf(tp, fp, fn)
    scores.append(Score("MICRO_AVG", tp, fp, fn, precision, recall, f1,
                        len(gold_keys), len(predicted_keys)))

    per_label = [s for s in scores if s.label != "MICRO_AVG"]
    if per_label:
        count = len(per_label)
        scores.append(Score(
            "MACRO_AVG", 0, 0, 0,
            round(sum(s.precision for s in per_label) / count, 4),
            round(sum(s.recall for s in per_label) / count, 4),
            round(sum(s.f1 for s in per_label) / count, 4),
            sum(s.support for s in per_label),
            sum(s.predicted for s in per_label),
        ))
    return scores


def _label_of(row: dict, field: str) -> str:
    value = (row.get(field) or "").strip()
    return "" if value.lower() in NONE_LABELS else value


# --------------------------------------------------------------------------
# precision sampling
# --------------------------------------------------------------------------
def stratified_sample(
    rows: Sequence[dict], stratum_field: str, per_stratum: int, seed: int = 20250809
) -> list[dict]:
    """Sample evenly across labels so rare types are not swamped.

    A uniform sample of 100 relations would be mostly OPERATED_BY and would
    leave ORBITS with two or three rows - too few to say anything about.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[row.get(stratum_field, "")].append(row)
    rng = random.Random(seed)
    sampled: list[dict] = []
    for stratum in sorted(buckets):
        group = buckets[stratum]
        sampled.extend(rng.sample(group, min(per_stratum, len(group))))
    return sampled


def precision_from_sample(
    rows: Sequence[dict], label_field: str, judgement_field: str = "correct"
) -> tuple[list[dict], int]:
    """Aggregate a hand-judged sample into per-label precision estimates."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    unjudged = 0
    for row in rows:
        verdict = (row.get(judgement_field) or "").strip().lower()
        if verdict not in YES_LABELS and verdict not in NONE_LABELS - {""}:
            unjudged += 1
            continue
        if verdict == "":
            unjudged += 1
            continue
        buckets[row.get(label_field, "")].append(row)

    results: list[dict] = []
    total_correct = total_judged = 0
    for label in sorted(buckets):
        group = buckets[label]
        correct = sum(
            1 for row in group
            if (row.get(judgement_field) or "").strip().lower() in YES_LABELS
        )
        low, high = wilson_interval(correct, len(group))
        total_correct += correct
        total_judged += len(group)
        results.append({
            "label": label,
            "judged": len(group),
            "correct": correct,
            "precision": round(correct / len(group), 4) if group else 0.0,
            "ci_low": low,
            "ci_high": high,
        })
    low, high = wilson_interval(total_correct, total_judged)
    results.append({
        "label": "OVERALL",
        "judged": total_judged,
        "correct": total_correct,
        "precision": round(total_correct / total_judged, 4) if total_judged else 0.0,
        "ci_low": low,
        "ci_high": high,
    })
    return results, unjudged


# --------------------------------------------------------------------------
# gold template generation
# --------------------------------------------------------------------------
def candidate_gold_rows(
    entities: Sequence[dict],
    signatures: dict[str, tuple[frozenset[str], frozenset[str]]],
    doc_ids: Iterable[str],
    max_pairs_per_sentence: int = 12,
) -> list[dict]:
    """Every type-compatible entity pair in the chosen documents.

    This is the annotation unit for gold evaluation: the annotator writes a
    relation name, or NONE, against each pair.
    """
    wanted = set(doc_ids)
    by_sentence: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in entities:
        if row["doc_id"] in wanted:
            by_sentence[(row["doc_id"], str(row["sent_id"]))].append(row)

    rows: list[dict] = []
    for (doc_id, sent_id), mentions in sorted(by_sentence.items()):
        seen: set[tuple[str, str]] = set()
        pairs = 0
        for subject in mentions:
            for obj in mentions:
                if subject["entity_key"] == obj["entity_key"]:
                    continue
                pair = (subject["entity_key"], obj["entity_key"])
                if pair in seen:
                    continue
                allowed = [
                    name for name, (subj_labels, obj_labels) in signatures.items()
                    if subject["label"] in subj_labels and obj["label"] in obj_labels
                ]
                if not allowed:
                    continue
                seen.add(pair)
                pairs += 1
                rows.append({
                    "doc_id": doc_id,
                    "sent_id": sent_id,
                    "subj_key": subject["entity_key"],
                    "obj_key": obj["entity_key"],
                    "relation": "",          # <- annotator fills this in
                    "possible": "|".join(allowed),
                    "sentence": subject.get("sentence", ""),
                })
                if pairs >= max_pairs_per_sentence:
                    break
            if pairs >= max_pairs_per_sentence:
                break
    return rows


# --------------------------------------------------------------------------
# system comparison
# --------------------------------------------------------------------------
def compare_sources(
    relations: Sequence[dict], mode: str = "strict"
) -> list[dict]:
    """Rule vs Hugging Face agreement, per relation type.

    Reported as agreement rather than accuracy: with no gold labels, neither
    system is the reference, so the honest statistic is overlap.
    """
    rule = [r for r in relations if r.get("source", "rule") == "rule"]
    hf = [r for r in relations if r.get("source") == "hf"]
    labels = sorted({r["relation"] for r in relations})

    rows: list[dict] = []
    for label in labels + ["ALL"]:
        rule_keys = {
            relation_key(r, mode) for r in rule
            if label == "ALL" or r["relation"] == label
        }
        hf_keys = {
            relation_key(r, mode) for r in hf
            if label == "ALL" or r["relation"] == label
        }
        both = rule_keys & hf_keys
        union = rule_keys | hf_keys
        rows.append({
            "relation": label,
            "rule": len(rule_keys),
            "hf": len(hf_keys),
            "agreed": len(both),
            "rule_only": len(rule_keys - hf_keys),
            "hf_only": len(hf_keys - rule_keys),
            "jaccard": round(len(both) / len(union), 4) if union else 0.0,
            "hf_recall_of_rule": round(len(both) / len(rule_keys), 4)
            if rule_keys else 0.0,
        })
    return rows


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def write_scores_csv(scores: Sequence[Score], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SCORE_FIELDS))
        writer.writeheader()
        writer.writerows(asdict(score) for score in scores)
    return path


def write_rows_csv(rows: Sequence[dict], path: Path, fieldnames: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
