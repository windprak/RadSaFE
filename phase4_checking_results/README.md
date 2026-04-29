# Phase 4 — Answer Checking

Verify that the `parsed_answer` extracted by the phase 3 parser actually matches
the answer the model expressed in its `raw_output`. This is especially relevant
for reasoning models (DeepSeek-R1, Qwen3, gpt-oss, Gemma-4, medgemma-27b, …)
whose raw outputs contain chain-of-thought followed by a final letter.

The verifier model is **Mistral-Small-4-119B-2603**, served via vLLM (OpenAI
compatible endpoint).

## Layout

```
phase4_checking_results/
├── scripts/
│   ├── count_raw_output_tokens.py    # token-count stats across all models
│   └── run_answer_check.py           # async checker
├── slurm/
│   └── run_check.sh                  # SLURM launcher (starts vLLM + checker)
├── logs/                             # slurm stdout
└── results/
    └── <MODEL_NAME>/risk_radiorag_checked.jsonl
```

## 1) Sizing — count raw-output tokens

Reports token counts across every `raw_output` / `raw_outputs[]` inside
`phase3_inference/results/*/risk_radiorag.jsonl`, using the same
Mistral-Large-Instruct-2407 tokenizer that Mistral-Small-4 uses.

```bash
apptainer exec --bind ${WORKSPACE}:${WORKSPACE} \
    ${WORKSPACE}/environment/inference.sif \
    python3 ${WORKSPACE}/phase4_checking_results/scripts/count_raw_output_tokens.py \
        --json-out ${WORKSPACE}/phase4_checking_results/logs/raw_token_stats.json
```

Output: per-model + global `count / total / min / p50 / p95 / p99 / max`,
largest single example, and a sizing recommendation for the verifier's input
window.

## 2) Choose which models to check

The checker is driven by a plain-text model list (one model directory name per
line; `#` and blank lines are ignored). The shipped `models.txt` lists every
model evaluated in the paper.

For large runs you can split the list across two SLURM jobs (e.g. by raw-output
token count) to halve the wall-time:

```bash
MODELS_FILE=${WORKSPACE}/phase4_checking_results/models_half_1.txt \
  sbatch ${WORKSPACE}/phase4_checking_results/slurm/run_check.sh
```

Each `risk_radiorag_checked.jsonl` is per-model and the script resumes at the
`question_id` level, so re-running with an updated list only processes what's
missing — safe to edit `models.txt` between runs.

**Do not put the same model in two list files run concurrently** — they will
race on the same output file.

## 3) Run the checker

Every `raw_output` becomes a JSON-constrained call:

```json
{"reasoning": "...", "confirmed_answer": "A" | "B" | "C" | "D" | "E" | null}
```

Appended to each record as:

```
conditions[cond]["greedy"]["checked_answer"]         = {...}
conditions[cond]["stochastic"]["checked_answers"]    = [{...}, ...]
```

Trivial cases where the raw output is exactly a single letter in `ANSWER_LETTERS`
are short-circuited locally (no LLM call) and flagged with `"skipped":
"single_letter"`, which keeps the run cheap.

### Mismatch log (hand-check queue)

For every model, a companion file is written:

```
phase4_checking_results/results/<MODEL_NAME>/mismatches.jsonl
```

One line is appended for every sample where:

- the checker's `confirmed_answer` differs from the parser's `parsed_answer`
  (`"reason": "disagreement"`), **or**
- the checker could not identify an answer (`confirmed_answer is null`,
  `"reason": "no_answer"`).

Each entry contains: `model`, `question_id`, `condition`, `sample`
(e.g. `greedy` or `stochastic[7]`), `correct_answer`, `parsed_answer`,
`confirmed_answer`, `checker_reasoning`, optional `checker_error`, and a
`raw_output_preview` (head+tail, 600 chars). Trivial single-letter samples are
excluded. Mismatches are append-only and never duplicated on resume because
they follow the same per-record skip logic as the main output.

### Submit the job

```bash
sbatch ${WORKSPACE}/phase4_checking_results/slurm/run_check.sh
```

Options via `EXTRA_CHECK_ARGS`:

```bash
EXTRA_CHECK_ARGS="--model DeepSeek-R1" \
    sbatch ${WORKSPACE}/phase4_checking_results/slurm/run_check.sh

EXTRA_CHECK_ARGS="--models-filter 'gpt-oss-120b,Qwen3-32B' --limit 5" \
    sbatch ${WORKSPACE}/phase4_checking_results/slurm/run_check.sh
```

Resume is automatic: re-running skips `question_id`s already present in
`risk_radiorag_checked.jsonl`.

### Direct invocation (if you already have vLLM running)

```bash
python3 scripts/run_answer_check.py \
    --base-url http://localhost:6000/v1 \
    --skip-wait --concurrency 128
```

## Prompt

The checker is instructed to treat the parser's answer as a hint and ignore the
ground-truth correct answer; it must produce strict JSON with `reasoning` +
`confirmed_answer`. Raw outputs longer than 24k characters are head+tail
truncated before being passed in.
