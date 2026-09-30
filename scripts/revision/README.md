# Revision analyses

Scripts used for the revised manuscript. They read the raw model outputs in `../../results/`
and need no API access.

```bash
cd scripts/revision
python reanalysis.py                  # -> revision_reanalysis.xlsx   (~2 min, includes power simulation)
python multireference_sensitivity.py  # -> multireference_sensitivity.xlsx (Supplementary Table S1)
python human_eval_sampling.py         # -> human_eval_sample.csv
```

| Script | Reproduces |
|---|---|
| `reanalysis.py` | Sentiment parse failures and accuracy with failures counted as incorrect (Tables 2, 4); QA format compliance and response length (Table 3); extracted-answer EM/Containment (Section 4.5); reference counts per question (Section 3.1); exact McNemar tests with Holm correction and discordance odds ratios (Tables 6, 8, 9) and the closed-book sensitivity analyses (Section 4.6); tier-by-condition GEE (Section 4.4); minimum detectable differences (Section 3.5) |
| `multireference_sensitivity.py` | Max-over-references vs. first reference only vs. all reference spans required (Section 4.6, Supplementary Table S1) |
| `human_eval_sampling.py` | Random samples for human evaluation: 60 closed-book and 40 context-supported QA items, 20 double-annotated closed-book items, 100 sentiment reviews (Section 3.7) |
| `score_human_eval.py` | Inter-annotator agreement, metric validity against human labels, human accuracy by model and human-label McNemar tests (Section 4.7, Tables 10, 11) |
| `extract.py` | Normalization, answer extraction (Appendix A, Table A3), format-compliance rule, matching metrics |
| `stats.py` | Exact McNemar test, Clopper–Pearson interval for the discordance odds ratio, Holm correction |
| `load.py` | Loads the result files |

Five-criterion scores for context-supported QA (Table 5) are produced by `../metric_compare_real.py`.

## Human-evaluation data

`score_human_eval.py` requires the annotation workbooks, the blinding key and the adjudication file.
These individual annotation records are not public, so Section 4.7 cannot be fully reproduced from
this repository. The script is included so that the scoring procedure can be inspected.

## Notes

- Items are paired by position (item identifier), not by question text: the 300 QA items contain
  293 distinct question strings.
- Sentiment parse failures are counted as incorrect in the main analysis; `sent_exclfail` in
  `revision_reanalysis.xlsx` is the sensitivity analysis that excludes them.
- The main QA outcomes use the stored `em` and `contain` columns, computed by
  `../kaz_qad_experiment.py`. `multireference_sensitivity.py` uses the same normalization.
