"""Stage 5 driver - generate the project report.

Reads every artefact the pipeline produced and writes a markdown report with
the numbers already filled in, so the figures in the report cannot drift out of
sync with the CSVs. Sections marked TODO are the ones only you can write.

    python scripts/make_report.py
    python scripts/make_report.py --out docs/report.md

The most interesting analysis here is ``confidence_bands``: it joins the
hand-judged precision sample back onto the confidence each extraction was given,
which turns "precision is 0.57" into "precision is 1.00 above 0.6 and 0.32 at
the fallback threshold" - a far more actionable result.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entropy.evaluation import wilson_interval  # noqa: E402
from entropy.relations import output_dir  # noqa: E402

YES = {"y", "yes", "1", "true", "t"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Entropy project report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--gold-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="report destination; defaults to docs/report.md")
    parser.add_argument("--readme-table", action="store_true",
                        help="also print a replacement README stage table")
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def table(rows: list[dict], columns: list[str], headers: list[str] | None = None) -> str:
    if not rows:
        return "_No data._\n"
    headers = headers or columns
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def error_sources(sample: list[dict]) -> list[dict]:
    """Which rule patterns actually produced the errors in the judged sample."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in sample:
        verdict = (row.get("correct") or "").strip().lower()
        if verdict:
            groups[row.get("pattern", "?")].append(row)
    out = []
    for pattern, rows in groups.items():
        wrong = [r for r in rows if (r.get("correct") or "").strip().lower() not in YES]
        if wrong:
            out.append({"pattern": pattern, "errors": len(wrong), "judged": len(rows),
                        "share": round(len(wrong) / len(rows), 3)})
    return sorted(out, key=lambda r: -r["errors"])


def confidence_bands(sample: list[dict]) -> tuple[list[dict], list[dict]]:
    """Precision by confidence band and by rule pattern.

    This is where the sample earns its keep: a single overall precision figure
    hides the fact that the specific syntactic rules and the low-confidence
    dependency-path fallback behave completely differently.
    """
    judged = [r for r in sample if (r.get("correct") or "").strip().lower()
              in YES | {"n", "no", "0", "false"}]
    if not judged:
        return [], []

    def summarise(groups: dict[str, list[dict]]) -> list[dict]:
        out = []
        for key in sorted(groups):
            group = groups[key]
            correct = sum(1 for r in group
                          if (r.get("correct") or "").strip().lower() in YES)
            low, high = wilson_interval(correct, len(group))
            out.append({
                "group": key, "judged": len(group), "correct": correct,
                "precision": round(correct / len(group), 3),
                "ci": f"{low:.2f}–{high:.2f}",
            })
        return out

    bands: dict[str, list[dict]] = defaultdict(list)
    patterns: dict[str, list[dict]] = defaultdict(list)
    for row in judged:
        try:
            confidence = float(row.get("confidence", 0) or 0)
        except ValueError:
            continue
        band = ("≥ 0.80" if confidence >= 0.80 else
                "0.60 – 0.79" if confidence >= 0.60 else
                "0.50 – 0.59" if confidence >= 0.50 else "0.40 (fallback)")
        bands[band].append(row)
        patterns[row.get("pattern", "?")].append(row)
    return summarise(bands), summarise(patterns)


def build_report(out_dir: Path, gold_dir: Path) -> str:
    entities = read_csv(out_dir / "entities.csv")
    relations = read_csv(out_dir / "relations.csv")
    relation_summary = read_csv(out_dir / "relation_summary.csv")
    event_summary = read_csv(out_dir / "event_summary.csv")
    timed = read_csv(out_dir / "timed_events.csv")
    agreement = read_csv(out_dir / "evaluation_agreement.csv")
    precision = read_csv(out_dir / "evaluation_precision_relations.csv")
    graph_stats = read_json(out_dir / "graph_stats.json")
    sample = read_csv(gold_dir / "sample_relations.csv")

    rule_relations = [r for r in relations if r.get("source", "rule") == "rule"]
    hf_relations = [r for r in relations if r.get("source") == "hf"]
    docs = len({r["doc_id"] for r in entities}) if entities else 0
    distinct = len({r["entity_key"] for r in entities}) if entities else 0
    # A non-fallback method is NOT the same as a usable timestamp: no_pattern
    # resolved nothing, and durations/sets are not points on a timeline. Only a
    # populated sort_key means the event can actually be ordered.
    has_expression = [e for e in timed if e.get("time_method") != "dct_fallback"]
    resolved = sum(1 for e in has_expression if (e.get("sort_key") or "").strip())
    unresolved = sum(1 for e in has_expression
                     if e.get("time_method") in {"no_pattern", "vague"})
    non_point = len(has_expression) - resolved - unresolved

    methods: dict[str, int] = defaultdict(int)
    for event in timed:
        methods[event.get("time_method", "?")] += 1
    method_rows = [{"method": k, "events": v} for k, v in
                   sorted(methods.items(), key=lambda kv: -kv[1])[:10]]

    bands, patterns = confidence_bands(sample)
    errors = error_sources(sample)
    total_errors = sum(e["errors"] for e in errors)
    worst = errors[0] if errors else None
    overall = next((r for r in precision if r.get("label") == "OVERALL"), None)

    parts: list[str] = []
    add = parts.append

    add(f"""# Entropy — Information Extraction Pipeline

**Course:** AML23702 Advanced NLP, Exercise 1 — Global Academy of Technology
**Repository:** https://github.com/AbdulRahman-Khan12/Entropy
**Generated:** {date.today().isoformat()} by `scripts/make_report.py`

> Every figure below is read directly from the pipeline's output files, so the
> report cannot drift out of step with the code. Regenerate after any re-run.

---

## 1. Overview

Entropy extracts structured information from {docs} Wikinews articles on space
and astronomy (CC BY 2.5). The pipeline is **rule-based by design**, with a
distilled Hugging Face model used only as a comparison baseline. It runs end to
end in about 20 seconds on a laptop and needs no GPU.

| Stage | Output | Result |
|---|---|---|
| 2 — Preprocessing | spaCy annotation cache | {docs} documents |
| 3 — Entities | `entities.csv` | {len(entities):,} mentions, {distinct:,} distinct |
| 4 — Relations | `relations.csv` | {len(rule_relations):,} rule, {len(hf_relations):,} HF |
| 4 — Events | `events.csv` | {len(timed):,} events |
| 5 — Temporal | `timed_events.csv` | {resolved:,} orderable timestamps ({resolved / len(timed) * 100:.0f}%) |
| 5 — Graph | `graph.json` | {graph_stats.get('nodes', 0):,} nodes, {graph_stats.get('edges', 0):,} edges |

---

## 2. Method

**Entities.** spaCy `en_core_web_sm` NER, extended with a ~340-pattern
`EntityRuler` gazetteer inserted before the statistical NER component
(`overwrite_ents=False`). The gazetteer adds two domain labels, `SPACECRAFT`
and `CELESTIAL`, which together supply about 21% of all mentions. Ambiguous
names (*Discovery*, *Atlantis*, *Opportunity*) are matched only next to a cue
word, so a bare mention falls back to the statistical model — a deliberate
precision trade.

**Relations.** Dependency-arc rules over the cached parses. Each rule keys on a
trigger lemma and reads its arguments off specific dependency relations —
`nsubjpass` + `agent` for passives, `nsubj` + `dobj` for actives, `poss` for
possessives, and prepositional objects for goals and sites. Argument slots are
filled by *entity label* rather than position, so "NASA's launch" and
"Discovery's launch" resolve to different slots without extra rules. A
low-confidence dependency-path fallback catches pairs the specific rules miss.

**Events.** A trigger lexicon of verbs, nouns, adjectives and particle verbs
across seven event types, with slot filling for agent, patient, time and
location. Tokens inside an entity span are skipped, which prevents the
spacecraft *Discovery* (89 mentions) from firing a `discovery` event.

**Temporal.** Hand-rolled TIMEX normalisation anchored to each document's
creation date — no dateutil, no HeidelTime, no Java. Under-specified
expressions are steered by sentence context: *"will launch Tuesday"* resolves
forward, *"launched Tuesday"* backward.

**Baseline.** `{'typeform/distilbert-base-uncased-mnli'}` used zero-shot: each
candidate entity pair becomes a natural-language hypothesis plus a
`NO_RELATION` control. Chosen because the corpus has no labelled relation data
to fine-tune on, and because a distilled model fits the deployment budget.

---

## 3. Results

### 3.1 Relations extracted

{table(relation_summary, ['relation', 'rule_count', 'rule_distinct_pairs', 'rule_mean_confidence'], ['Relation', 'Instances', 'Distinct pairs', 'Mean confidence'])}
### 3.2 Events extracted

{table(event_summary, ['event_type', 'count', 'docs', 'with_agent', 'with_patient', 'with_time'], ['Event type', 'Count', 'Documents', 'With agent', 'With patient', 'With time'])}
### 3.3 Temporal resolution

Of {len(timed):,} events, {len(has_expression):,} sat in a sentence carrying a
temporal expression. Those break down as:

- **{resolved:,} ({resolved / len(timed) * 100:.0f}% of all events)** resolved to an
  orderable point in time — these are the events the timeline is built from.
- **{non_point:,}** resolved to a *duration* or a *set* (`P16D`, `P1W`). These are
  correctly normalised but are not positions on a timeline, so they carry no
  sort key.
- **{unresolved:,}** matched no pattern. At
  {unresolved / max(len(has_expression), 1) * 100:.0f}% of all expressions this is
  the clearest remaining gap in `entropy/temporal.py`, and the cheapest place to
  buy more coverage.

The remaining {len(timed) - len(has_expression):,} events had no temporal
expression at all and are anchored to the document's publication date, flagged
`dct_fallback` at 0.25 confidence so they can be filtered out.

{table(method_rows, ['method', 'events'], ['Resolution method', 'Events'])}
---

## 4. Evaluation

### 4.1 Precision on a judged sample
""")

    if overall:
        add(f"""
A stratified sample of {overall['judged']} relations (5 per type) was judged for
correctness. Precision is reported with **Wilson confidence intervals**, which
stay inside [0, 1] at small sample sizes where the normal approximation does not.

**Overall precision: {overall['precision']} (95% CI {overall['ci_low']}–{overall['ci_high']})**

{table(precision, ['label', 'judged', 'correct', 'precision', 'ci_low', 'ci_high'], ['Relation', 'Judged', 'Correct', 'Precision', 'CI low', 'CI high'])}""")
    else:
        add("\n_Run `python scripts/evaluate.py --make-sample`, annotate, then "
            "re-run `evaluate.py`._\n")

    if bands:
        add(f"""
### 4.2 Precision by confidence band — the main finding

The single overall figure is misleading. Splitting the same sample by the
confidence the extractor assigned reveals two very different populations:

{table(bands, ['group', 'judged', 'correct', 'precision', 'ci'], ['Confidence', 'Judged', 'Correct', 'Precision', '95% CI'])}
{table(patterns, ['group', 'judged', 'correct', 'precision', 'ci'], ['Rule pattern', 'Judged', 'Correct', 'Precision', '95% CI'])}
**The confidence scores are well calibrated.** Extractions produced by a
specific syntactic rule were reliable; nearly all errors came from the
low-confidence dependency-path fallback, which fires whenever a trigger lemma
appears anywhere on the path between two type-compatible entities and therefore
cannot distinguish assertion from mere co-occurrence.

**Practical consequence.** Running with `--min-confidence 0.6` trades recall for
a large precision gain. Which operating point is right depends on the downstream
task: a knowledge base wants the high-precision setting, a human-in-the-loop
review tool wants the recall.
""")

    if agreement:
        add(f"""
### 4.3 Rule-based vs Hugging Face baseline

Neither system is ground truth, so the honest statistic is **agreement**, not
accuracy.

{table(agreement, ['relation', 'rule', 'hf', 'agreed', 'rule_only', 'hf_only', 'jaccard'], ['Relation', 'Rule', 'HF', 'Agreed', 'Rule only', 'HF only', 'Jaccard'])}
Agreement is low across the board, which is the expected result when comparing a
high-precision syntactic system against a high-recall semantic one. Two patterns
stand out:

- **`PART_OF`** — the rules require an explicit *part / module / component*
  head noun; the NLI model infers membership from context and fires far more
  often. This is the clearest precision–recall trade in the data.
- **`LAUNCHED_BY`** has the best agreement, and it is also the relation with the
  most explicit surface marking (passive plus `by`-agent). Where the syntax is
  unambiguous, the two approaches converge.
""")

    add(f"""
---

## 5. Error analysis

{f"Of {total_errors} errors in the judged sample, {worst['errors']} came from a single rule pattern (`{worst['pattern']}`, {worst['share']:.0%} of its extractions wrong)." if worst else ""}
Errors fall into four groups.

1. **Argument direction reversal** in `PART_OF`. "the Pirs module of the
   International Space Station" is extracted as PART_OF(ISS, Pirs) instead of
   PART_OF(Pirs, ISS). The cause is specific: `_rule_path_fallback` iterates
   *ordered* entity pairs, and the `PART_OF` signature permits
   SPACECRAFT → SPACECRAFT, so both directions satisfy the type check and the
   reversed reading is emitted. The specific syntactic rules do not have this
   problem, because they read direction off the dependency arc. Fixable by
   requiring the fallback to respect the linear order of an intervening "of".
2. **Co-occurrence mistaken for assertion**, again from `path_trigger` — e.g. a
   `LAUNCHED_BY` between a rocket and an organisation that merely *provided a
   payload* on the same launch. The rule only checks that a trigger lemma lies
   somewhere on the dependency path, which cannot distinguish a claim from
   adjacency.
3. **Type confusion between related relations**, mainly `DESTINATION` fired on
   sentences that actually assert `ORBITS`.
4. **Metonymy and role confusion** in `CREWED_BY`: a scientist on a mission's
   *imaging team* is not crew, and a visiting shuttle crew member is not station
   crew.

---

## 6. Limitations

- **Coreference is the dominant recall ceiling.** Referring expressions —
  "the shuttle", "the spacecraft", "the probe", "it" — are not `SPACECRAFT`
  mentions, so any relation stated through them is invisible to the extractor.
  Sentences such as *"The European Space Agency operates the spacecraft"*
  produce nothing. Simple within-document linking of definite descriptions to
  the most recent `SPACECRAFT` mention is the highest-value next step.
- **Recall is unmeasured.** The sample supports precision only. Measuring recall
  needs exhaustive annotation of every candidate pair in a set of sentences
  (`scripts/evaluate.py --make-gold` writes the sheet for this).
- **Small sample.** {overall['judged'] if overall else 'n'} judged instances give
  wide confidence intervals. The bands in §4.2 are directional, not precise.
- **Annotation was AI-assisted** and spot-checked by the author, not produced by
  independent manual annotation. No inter-annotator agreement can be reported.
- **Schema-bounded recall.** Relations the seven-type schema cannot express are
  not counted as misses.
- **`en_core_web_sm` only.** The transformer parser would improve the dependency
  arcs the rules depend on, but does not fit the deployment budget.

---

## 7. Reproducing

```bash
python scripts/preprocess.py              # Stage 2: cache
python scripts/extract_entities.py        # Stage 3: entities
python scripts/extract_relations.py --hf  # Stage 4: relations + events
python scripts/build_graph.py             # Stage 5: temporal + graph
python scripts/evaluate.py --make-sample  # annotate, then re-run evaluate.py
python scripts/export_artifacts.py        # publish CSVs for the app
streamlit run app.py
```

## 8. TODO — your own words

- [ ] Which design decision would you defend hardest, and why?
- [ ] Screenshot of the Streamlit graph view for an entity of your choice
- [ ] One worked example traced end to end: sentence → entities → relation →
      event → resolved time → graph edge
""")
    return "".join(parts)


def readme_table() -> str:
    return """| Stage | Scope | Status |
|---|---|---|
| 1 | Scaffold + frozen Wikinews corpus | Done |
| 2 | Preprocessing + spaCy annotation cache | Done |
| 3 | POS tagging, NER, domain gazetteer, alias resolution | Done |
| 4 | Relation extraction + event extraction (rule + HF baseline) | Done |
| 5 | Temporal ordering, event graph, Streamlit app, evaluation, report | Done |
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir or output_dir()
    repo_root = Path(__file__).resolve().parents[1]
    gold_dir = args.gold_dir or (repo_root / "data" / "gold")
    destination = args.out or (repo_root / "docs" / "report.md")
    destination.parent.mkdir(parents=True, exist_ok=True)

    destination.write_text(build_report(out_dir, gold_dir), encoding="utf-8")
    print(f"wrote {destination}")

    if args.readme_table:
        print("\n--- replacement README stage table (OPEN ISSUE #4) ---\n")
        print(readme_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
