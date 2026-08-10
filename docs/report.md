# Entropy — Information Extraction Pipeline

**Course:** AML23702 Advanced NLP, Exercise 1 — Global Academy of Technology
**Repository:** https://github.com/AbdulRahman-Khan12/Entropy
**Generated:** 2026-08-09 by `scripts/make_report.py`

> Every figure below is read directly from the pipeline's output files, so the
> report cannot drift out of step with the code. Regenerate after any re-run.

---

## 1. Overview

Entropy extracts structured information from 500 Wikinews articles on space
and astronomy (CC BY 2.5). The pipeline is **rule-based by design**, with a
distilled Hugging Face model used only as a comparison baseline. It runs end to
end in about 20 seconds on a laptop and needs no GPU.

| Stage | Output | Result |
|---|---|---|
| 2 — Preprocessing | spaCy annotation cache | 500 documents |
| 3 — Entities | `entities.csv` | 14,500 mentions, 5,121 distinct |
| 4 — Relations | `relations.csv` | 197 rule, 285 HF |
| 4 — Events | `events.csv` | 3,043 events |
| 5 — Temporal | `timed_events.csv` | 1,103 orderable timestamps (36%) |
| 5 — Graph | `graph.json` | 3,331 nodes, 15,132 edges |

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

**Baseline.** `typeform/distilbert-base-uncased-mnli` used zero-shot: each
candidate entity pair becomes a natural-language hypothesis plus a
`NO_RELATION` control. Chosen because the corpus has no labelled relation data
to fine-tune on, and because a distilled model fits the deployment budget.

---

## 3. Results

### 3.1 Relations extracted

| Relation | Instances | Distinct pairs | Mean confidence |
|---|---|---|---|
| LAUNCHED_BY | 23 | 19 | 0.571 |
| OPERATED_BY | 61 | 32 | 0.641 |
| DESTINATION | 30 | 18 | 0.521 |
| ORBITS | 14 | 9 | 0.566 |
| LAUNCHED_FROM | 30 | 24 | 0.582 |
| PART_OF | 15 | 15 | 0.413 |
| CREWED_BY | 24 | 23 | 0.513 |
| ALL | 197 | 140 | 0.567 |

### 3.2 Events extracted

| Event type | Count | Documents | With agent | With patient | With time |
|---|---|---|---|---|---|
| docking | 81 | 41 | 15 | 17 | 64 |
| delay | 168 | 94 | 11 | 2 | 115 |
| failure | 479 | 204 | 19 | 44 | 214 |
| flyby | 58 | 27 | 3 | 15 | 26 |
| landing | 266 | 105 | 7 | 36 | 120 |
| launch | 1175 | 261 | 50 | 175 | 723 |
| discovery | 816 | 257 | 79 | 22 | 251 |
| ALL | 3043 | 466 | 184 | 311 | 1513 |

### 3.3 Temporal resolution

Of 3,043 events, 1,513 sat in a sentence carrying a
temporal expression. Those break down as:

- **1,103 (36% of all events)** resolved to an
  orderable point in time — these are the events the timeline is built from.
- **184** resolved to a *duration* or a *set* (`P16D`, `P1W`). These are
  correctly normalised but are not positions on a timeline, so they carry no
  sort key.
- **226** matched no pattern. At
  15% of all expressions this is
  the clearest remaining gap in `entropy/temporal.py`, and the cheapest place to
  buy more coverage.

The remaining 1,530 events had no temporal
expression at all and are anchored to the document's publication date, flagged
`dct_fallback` at 0.25 confidence so they can be filtered out.

| Resolution method | Events |
|---|---|
| dct_fallback | 1530 |
| year | 243 |
| no_pattern | 226 |
| duration | 178 |
| relative_weekday | 138 |
| bare_month | 130 |
| month_day | 118 |
| clock | 104 |
| day_word | 102 |
| month_year | 89 |

---

## 4. Evaluation

### 4.1 Precision on a judged sample

A stratified sample of 35 relations (5 per type) was judged for
correctness. Precision is reported with **Wilson confidence intervals**, which
stay inside [0, 1] at small sample sizes where the normal approximation does not.

**Overall precision: 0.5714 (95% CI 0.4086–0.7202)**

| Relation | Judged | Correct | Precision | CI low | CI high |
|---|---|---|---|---|---|
| CREWED_BY | 5 | 3 | 0.6 | 0.2307 | 0.8824 |
| DESTINATION | 5 | 3 | 0.6 | 0.2307 | 0.8824 |
| LAUNCHED_BY | 5 | 1 | 0.2 | 0.0362 | 0.6245 |
| LAUNCHED_FROM | 5 | 3 | 0.6 | 0.2307 | 0.8824 |
| OPERATED_BY | 5 | 5 | 1.0 | 0.5655 | 1.0 |
| ORBITS | 5 | 4 | 0.8 | 0.3755 | 0.9638 |
| PART_OF | 5 | 1 | 0.2 | 0.0362 | 0.6245 |
| OVERALL | 35 | 20 | 0.5714 | 0.4086 | 0.7202 |

### 4.2 Precision by confidence band — the main finding

The single overall figure is misleading. Splitting the same sample by the
confidence the extractor assigned reveals two very different populations:

| Confidence | Judged | Correct | Precision | 95% CI |
|---|---|---|---|---|
| 0.40 (fallback) | 22 | 7 | 0.318 | 0.16–0.53 |
| 0.60 – 0.79 | 7 | 7 | 1.0 | 0.65–1.00 |
| ≥ 0.80 | 6 | 6 | 1.0 | 0.61–1.00 |

| Rule pattern | Judged | Correct | Precision | 95% CI |
|---|---|---|---|---|
| active_svo | 3 | 3 | 1.0 | 0.44–1.00 |
| noun_of | 1 | 1 | 1.0 | 0.21–1.00 |
| np_prep | 1 | 1 | 1.0 | 0.21–1.00 |
| path_trigger | 22 | 7 | 0.318 | 0.16–0.53 |
| poss | 5 | 5 | 1.0 | 0.57–1.00 |
| verb_prep | 3 | 3 | 1.0 | 0.44–1.00 |

**The confidence scores are well calibrated.** Extractions produced by a
specific syntactic rule were reliable; nearly all errors came from the
low-confidence dependency-path fallback, which fires whenever a trigger lemma
appears anywhere on the path between two type-compatible entities and therefore
cannot distinguish assertion from mere co-occurrence.

**Practical consequence.** Running with `--min-confidence 0.6` trades recall for
a large precision gain. Which operating point is right depends on the downstream
task: a knowledge base wants the high-precision setting, a human-in-the-loop
review tool wants the recall.

### 4.3 Rule-based vs Hugging Face baseline

Neither system is ground truth, so the honest statistic is **agreement**, not
accuracy.

| Relation | Rule | HF | Agreed | Rule only | HF only | Jaccard |
|---|---|---|---|---|---|---|
| CREWED_BY | 24 | 33 | 0 | 24 | 33 | 0.0 |
| DESTINATION | 30 | 11 | 1 | 29 | 10 | 0.025 |
| LAUNCHED_BY | 23 | 8 | 3 | 20 | 5 | 0.1071 |
| LAUNCHED_FROM | 30 | 47 | 0 | 30 | 47 | 0.0 |
| OPERATED_BY | 61 | 8 | 0 | 61 | 8 | 0.0 |
| ORBITS | 14 | 31 | 2 | 12 | 29 | 0.0465 |
| PART_OF | 15 | 105 | 1 | 14 | 104 | 0.0084 |
| ALL | 197 | 243 | 7 | 190 | 236 | 0.0162 |

Agreement is low across the board, which is the expected result when comparing a
high-precision syntactic system against a high-recall semantic one. Two patterns
stand out:

- **`PART_OF`** — the rules require an explicit *part / module / component*
  head noun; the NLI model infers membership from context and fires far more
  often. This is the clearest precision–recall trade in the data.
- **`LAUNCHED_BY`** has the best agreement, and it is also the relation with the
  most explicit surface marking (passive plus `by`-agent). Where the syntax is
  unambiguous, the two approaches converge.

---

## 5. Error analysis

Of 15 errors in the judged sample, 15 came from a single rule pattern (`path_trigger`, 68% of its extractions wrong).
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
- **Small sample.** 35 judged instances give
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
