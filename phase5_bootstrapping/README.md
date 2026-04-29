# Phase 5 — bootstrap statistical analysis

Turns the per-(model × condition × question) judged answers from Phase 4
into the paper's headline numbers: accuracy with 95 % CIs, safety-rate
breakdowns (high-risk / unsafe / contradiction), self-consistency
confidence, ensemble majority votes, paired-bootstrap p-values, and the
per-question safety aggregations.

## Inputs

- `${WORKSPACE}/phase4_checking_results/results/<model>/risk_radiorag_checked.jsonl`
  — one line per question, with the judge-confirmed letter and the per-option
  safety flags from the dataset.
- `${WORKSPACE}/datasets/risk_radiorag_full_merged.json` — the canonical
  question metadata and high-risk / unsafe / contradiction labels per option.

## Outputs (`bootstrap_results/`)

| File                            | Produced by                  | Feeds into            |
| ------------------------------- | ---------------------------- | --------------------- |
| `summary_all.json`              | `run_bootstrap.py`           | Tables 1, 2           |
| `safety_summary.json`           | `run_safety_rates.py`        | Tables 1, 2           |
| `confidence_summary.json`       | `run_confidence.py`          | Table 1               |
| `calibration_bins.csv`          | `run_confidence.py`          | reliability diagrams  |
| `confidence_per_question.csv`   | `run_confidence.py`          | confidence sensitivity|
| `ensemble_summary.json`         | `run_ensembles.py`           | Table 3               |
| `per_question_summary.json`     | `run_per_question_safety.py` | per-question analyses |
| `pvalue_*.csv`                  | `run_pvalues.py`             | Tables 1, 2 footnotes |

The paper's Tables 1–4 are direct restylings of these JSON / CSV files;
the specific layout helper used by the authors is not redistributed.

## Reproduction

```bash
SIF=$WORKSPACE/environment/inference.sif
EXEC="apptainer exec --bind $WORKSPACE:$WORKSPACE $SIF python3"

$EXEC run_bootstrap.py
$EXEC run_safety_rates.py
$EXEC run_confidence.py
$EXEC run_ensembles.py
$EXEC run_per_question_safety.py
$EXEC run_pvalues.py
```

All scripts are idempotent and re-read directly from
`phase4_checking_results/results/`. The bootstrap uses 1 000 resamples by
default with a fixed seed (`seed = 42`); CPU runtime is ~ 20 min total.

## Notes on the methodology

- **Single regime.** The greedy decode (`temperature = 0`) is taken at face
  value; abstentions (no parseable letter) count as wrong.
- **Self-consistency regime.** Twenty stochastic samples (`temperature
  = 0.7`); the reported answer is the mode of the judge-confirmed letters.
  The SC confidence is `1 − H(p) / log K` over the empirical answer
  distribution.
- **Safety rates.** For each (model, condition, question) we look at the
  *selected* answer letter and read the corresponding option's `high_risk`,
  `unsafe`, `contradicts` flags from the dataset; pooled rates are the
  weighted mean across questions.
- **Bootstrap.** `B = 1000` bootstrap replicates by default, with `n = 200`
  questions per replicate and `seed = 42`. The stored bootstrap `indices`
  array is therefore a `1000 × 200` matrix of resampled question indices.
  Resampling is with replacement *over questions* (not over models); CIs are
  the percentile-method 2.5 / 97.5 quantiles.
- **Ensembles.** Three-member majority vote of greedy answers; ties default
  to alphabetical-first, and any tie containing a NULL ballot is treated as
  an abstention (i.e. wrong).
