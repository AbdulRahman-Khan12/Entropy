# Entropy

An information extraction pipeline over a frozen news corpus. Entropy performs
POS tagging, named entity recognition, relation extraction, event extraction and
temporal ordering, then presents the results — annotated text, entity and
relation tables, and an event timeline — through a Streamlit interface.

Coursework for **AML23702 — Advanced Natural Language Processing**, Exercise 1.

---

## Why this design

The hard part of Exercise 1 is temporal ordering. Ordering events requires a
**Document Creation Time (DCT)**: without knowing when a document was written,
expressions like *"on Tuesday"* or *"three days later"* cannot be resolved to
real dates, and the timeline degrades into a list of whatever absolute dates
happened to appear in the text.

Wikinews satisfies this. Every article carries a publication date, the register
is newswire (so relative time expressions are dense), and articles cluster
around shared stories — which gives *cross-document* timelines rather than one
timeline per article.

The corpus is **frozen and committed** to this repository. Cloning the repo is
enough to reproduce every result; no downloads, no API keys, no drift.

---

## Setup

Python 3.11 or 3.12 is recommended. spaCy, Stanza and transformers (added from
Stage 2 onward) have the most reliable Windows wheels on those versions.

```
cd "D:\GlobalAcademic\7thsem\Advanced NLP\Projects\Entropy"

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Note the quotes around the path — `Advanced NLP` contains a space.

If PowerShell blocks the activation script, run once:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Every command below assumes the environment is active — the prompt should begin
with `(.venv)`. Without it you will hit `ModuleNotFoundError: No module named 'spacy'`.

---

## Getting the data

### Working corpus (required)

1. Download the Wikinews article dataset (a JSON array of ~21,000 articles,
   CC BY 2.5): <https://www.kaggle.com/datasets/datagator/wikinews-article-dataset>
2. Save the JSON file as `data/raw/wikinews.json`.
3. Build the frozen slice:

```
python -m scripts.build_corpus --dry-run   # preview the selection
python -m scripts.build_corpus             # write it
```

This produces `data/corpus/entropy_corpus.jsonl` and `data/corpus/manifest.json`.
Both are committed; `data/raw/` is not.

Tune the selection in `entropy/config.py` — `THEME_CATEGORIES`, `MIN_WORDS` and
`MAX_DOCS` are the knobs that matter.

### Evaluation data (optional, Stage 5b)

TimeBank + AQUAINT in TimeML format, used to measure extraction accuracy
against gold annotations. Place under `data/eval/`. Not redistributed here —
see `NOTICE.md`.

---

## Running the pipeline

From the repository root, with the environment active:

```
python scripts/preprocess.py
python scripts/extract_entities.py
python scripts/extract_relations.py --hf --hf-limit 50
python scripts/build_graph.py
python scripts/evaluate.py
python scripts/make_report.py --readme-table
python scripts/export_artifacts.py
streamlit run app.py
```

**Do not run `evaluate.py --make-sample`.** It overwrites
`data/gold/sample_relations.csv` and destroys the existing manual judgements.
`make_report.py` only reads, so it is always safe to re-run.

---

## Layout

```
Entropy/
├─ entropy/                 importable package — the Streamlit app depends on this
│  ├─ config.py             all paths and tunable parameters
│  ├─ corpus.py             Document model + loaders
│  └─ relations.py          relation patterns and extraction rules
├─ scripts/                 one-off command-line jobs
│  ├─ build_corpus.py       raw dump -> frozen JSONL
│  ├─ preprocess.py         segmentation and tokenisation
│  ├─ extract_entities.py   POS tagging and NER
│  ├─ extract_relations.py  rule-based and transformer relation extraction
│  ├─ build_graph.py        entity and event graph construction
│  ├─ evaluate.py           scoring against gold judgements
│  ├─ make_report.py        report tables and figures (read-only)
│  └─ export_artifacts.py   final exports
├─ data/
│  ├─ raw/                  git-ignored — large source dumps
│  ├─ corpus/               committed — the frozen working corpus
│  ├─ gold/                 manual judgements used for evaluation
│  └─ eval/                 git-ignored — gold annotations
├─ docs/
│  └─ report.md             written report
├─ outputs/                 git-ignored — generated tables and figures
└─ app.py                   Streamlit interface
```

The `entropy/` vs `scripts/` split matters for deployment: Streamlit Community
Cloud only needs to import `entropy/`, so build-time code never ships.

---

## Stages

## Stages

| Stage | Scope                                                    | Status |
| ----- | -------------------------------------------------------- | ------ |
| 1     | Scaffold + frozen Wikinews corpus                        | Done   |
| 2     | Preprocessing + spaCy annotation cache                   | Done   |
| 3     | POS tagging, NER, domain gazetteer, alias resolution     | Done   |
| 4     | Relation extraction + event extraction (rule + HF baseline) | Done |
| 5     | **Delivery**                                             | Done   |
| 5a    | Temporal ordering + event graph                          | Done   |
| 5b    | Streamlit app                                            | Done   |
| 5c    | Evaluation + report                                      | Done   |

---

## Attribution

See `NOTICE.md`.
