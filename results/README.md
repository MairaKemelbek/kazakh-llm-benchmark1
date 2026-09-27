# Raw model outputs

Item-level outputs of the five models, one file per regime and model (300 rows each).
Row order = item order of the sampled files produced by `scripts/prepare_data.py` (seed 42);
rows are paired across models by position (item ID), not by question text.

| File pattern | Regime | Columns |
|---|---|---|
| `kazsandra_test300_<model>.csv` | Sentiment (KazSAnDRA) | text, domain, gold, raw, pred (-1 = parse failure) |
| `kazqad_test300_<model>.csv` | QA with context (KazQAD) | question, gold (references separated by `\|\|\|`), raw, em, f1, contain |
| `kazqad_cb300_<model>.csv` | QA closed-book (KazQAD) | same as above |

Models: gpt-4o, gpt-4o-mini, claude-sonnet (claude-sonnet-4-6), claude-haiku (claude-haiku-4-5-20251001),
gemini-flash (gemini-flash-lite-latest). Collected in July 2026, temperature 0.

Licences: KazSAnDRA texts CC BY 4.0 (Yeshpanov & Varol, 2024); KazQAD questions/answers CC BY-SA (Yeshpanov et al., 2024).
