# Data sources and attribution

## Working corpus — Wikinews (English)

The corpus in `data/corpus/` is derived from English Wikinews.

- Source: https://en.wikinews.org/
- Licence: Creative Commons Attribution 2.5 (CC BY 2.5) —
  https://creativecommons.org/licenses/by/2.5/
- Copyright: the Wikinews contributors
- Status: English Wikinews was made read-only on 4 May 2026. The archive is
  therefore frozen, which is what allows this project to be exactly
  reproducible.

Modifications made by this project: articles were filtered by category, date
range and length; whitespace was normalised; stable document identifiers were
assigned. No article text was rewritten. The full selection procedure is
recorded in `data/corpus/manifest.json` and implemented in
`scripts/build_corpus.py`.

## Evaluation data — TimeBank and AQUAINT (TimeML)

Used only for measuring the accuracy of the extraction pipeline. **Not
redistributed in this repository** — `data/eval/` is git-ignored. Obtain it
separately as described in `README.md`.

- TimeBank 1.2 — Linguistic Data Consortium (LDC2006T08)
- AQUAINT TimeML corpus
- TempEval-3 `TBAQ-cleaned` release (reformatted, schema-compatible versions
  of both of the above)

## This project

Coursework for AML23702 (Advanced Natural Language Processing), Department of
Artificial Intelligence and Machine Learning, Global Academy of Technology.
