# Benchmarking LLMs on Kazakh Text Understanding and Knowledge Tasks

Code for the paper *"Benchmarking Large Language Models on Kazakh Text Understanding
and Knowledge Tasks: Implications for Educational AI Assistants."*

The study evaluates five commercial LLMs (GPT-4o, GPT-4o-mini, Claude Sonnet,
Claude Haiku, Gemini Flash Lite) across three zero-shot regimes that progressively
remove external support, so that language competence and parametric knowledge are
measured separately:

| Regime | Dataset | What it measures |
|---|---|---|
| Sentiment | KazSAnDRA | understanding user-generated Kazakh |
| QA with context | KazQAD | reading comprehension |
| QA closed-book | KazQAD (no passage) | parametric knowledge |

## Data

**We do not redistribute the datasets.** KazQAD has controlled access and must be
obtained through its official approval procedure. The scripts below regenerate the
exact evaluation samples used in the paper from the official sources, using a fixed
seed (42), so the samples are reproducible without redistribution.

- KazSAnDRA — https://github.com/IS2AI/KazSAnDRA (test split `05_pc_test.zip`)
- KazQAD — https://huggingface.co/datasets/issai/kazqad (request access first)

Please cite the dataset authors:
- Yeshpanov & Varol (2024), *KazSAnDRA*, LREC-COLING 2024
- Yeshpanov et al. (2024), *KazQAD*, LREC-COLING 2024

## Setup

```bash
pip install -r requirements.txt

export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export HF_TOKEN=...          # required for KazQAD
```

## Reproducing the samples

```bash
python scripts/prepare_data.py
```

Produces `kazsandra_test300.csv` (150 positive / 150 negative, balanced) and
`kazqad_test300.csv` (300 items). Both use seed 42 and are byte-identical to the
samples used in the paper.

## Running the experiments

```bash
# 1. Sentiment (KazSAnDRA)
python scripts/kaz_experiment.py --data kazsandra_test300.csv

# 2. Reading-comprehension QA (KazQAD, with context)
python scripts/kaz_qad_experiment.py --data kazqad_test300.csv

# 3. Closed-book QA (same items, no context)
python scripts/kaz_qad_cb.py --data kazqad_test300.csv
```

Each script writes per-item responses to `results/` and appends summary metrics.
Use `--models` to run a subset, e.g. `--models gpt-4o-mini`.

Note: `kaz_qad_cb.py` is `kaz_qad_experiment.py` with the context removed from the
prompt; see `scripts/make_closed_book.py`.

## Analysis

```bash
# five answer-matching criteria (Table 4)
python scripts/metric_compare_real.py

# statistics: McNemar + Holm + odds ratios with CIs (Tables 6, 7)
python scripts/analysis.py
```

## Notes on reproducibility

- All sampling uses seed 42; decoding temperature is fixed at 0.
- A temperature of 0 does not guarantee full determinism in production LLM APIs;
  scores are single-run estimates.
- Model versions are pinned where the provider exposes dated snapshots
  (`gpt-4o-2024-08-06`, `gpt-4o-mini-2024-07-18`, `claude-sonnet-4-6`,
  `claude-haiku-4-5-20251001`). Gemini was accessed through the alias
  `gemini-flash-lite-latest`, which does not expose a dated version;
  experiments were run in July 2026.
- Requests are issued sequentially with a fixed delay (`SLEEP`) to stay within
  provider rate limits. Adjust it to your quota.

## Citation

```bibtex
@article{kemelbek2026kazakh,
  title   = {Benchmarking Large Language Models on Kazakh Text Understanding
             and Knowledge Tasks: Implications for Educational AI Assistants},
  author  = {Kemelbek, Maira},
  journal = {(not published)},
  year    = {2026-2027}
}
```

## License

Code: MIT. The datasets remain under their original licences and are not
redistributed here.
