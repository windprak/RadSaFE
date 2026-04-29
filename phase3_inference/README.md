# Phase 3 — Inference

Runs every (model × condition × question) combination over the RadSaFE-200
dataset, producing the raw model outputs that Phase 4 then verifies with an
LLM judge.

## Layout

```
phase3_inference/
├── config.py                  Model list, context limits, decoding params,
│                              condition definitions, prompt templates.
├── scripts/
│   └── 02_run_inference.py            vLLM client driver (all open-weight models)
├── slurm/                     Cluster job scripts (see HPC.md)
└── results/<model>/risk_radiorag.jsonl  ← consumed by Phase 4
```

## Models

The open-weight models reported in the paper are listed in `config.py`
under `MODELS`: Qwen2.5/Qwen3 families, Gemma-3/Gemma-4/MedGemma families,
Llama-3/3.2/3.3 families, Ministral/Mistral-Small, and gpt-oss-20b/120b.
Set `MODELS_DIR` to your local HuggingFace cache.

> The largest models in the paper panel (DeepSeek-R1, DeepSeek-V3.2,
> Mistral-Large-3-675B, Qwen3-VL-235B, Llama-4-Scout-17B-16E) were served
> on B200 / MI300X nodes using the same `02_run_inference.py` driver
> against the corresponding vLLM endpoint.

## Conditions

| Condition          | Source                                        | Tokens (≈)  |
| ------------------ | --------------------------------------------- | ----------- |
| `zero_shot`        | None                                          | 0           |
| `top_1`            | Phase-2 retrieval (top-1)                     | 260         |
| `top_5`            | Phase-2 retrieval (top-5)                     | 1 300       |
| `top_10`           | Phase-2 retrieval (top-10)                    | 2 600       |
| `evidence_clean`   | Curated clean evidence field                  | varies      |
| `evidence_conflict`| Curated conflicting-evidence field            | varies      |
| `deep_research`    | Pre-baked agentic-RAG (`deep_research_report`)| up to ~30 k |
| `context_32k`      | Extended-context truncated to 32 k            | 30 000      |
| `context_100k`     | Extended-context truncated to 100 k           | 96 000      |
| `context_max`      | Extended-context truncated to model max       | up to ~120 k|

Extended conditions use **each model's own tokenizer** for truncation, so
the token budget is respected per-model rather than per-tokenizer.

## Decoding

- **Greedy.** `temperature=0`, `max_tokens=10`, `n=1`. Used as the
  Single-decode regime in the paper.
- **Stochastic.** `temperature=0.7`, `n=20`. Used downstream by Phase 5 for
  the self-consistency confidence estimate (entropy-based) and the
  majority-vote answer.
- **Reasoning models.** Same temperatures, but `max_tokens=4096` to give
  them room to emit chain-of-thought before the final letter. Detected by
  model-name pattern (`gpt-oss`, `Qwen3`, `gemma-4`, `DeepSeek-R1`,
  `medgemma-27b`).

The released inference script does **not** request token logprobs; the
paper's confidence metric is computed in Phase 5 from the empirical
distribution over the 20 stochastic samples (1 − normalised entropy).

See `config.py` (`GREEDY_PARAMS`, `STOCHASTIC_PARAMS`,
`REASONING_PARAMS_*`) for the full schedule.

## Quick start

```bash
export WORKSPACE=/abs/path/to/this/checkout
export MODELS_DIR=/abs/path/to/HF/cache

# Smoke test (smallest model, single condition)
sbatch phase3_inference/slurm/run_debug.sh

# Full sweep across all locally hosted models
sbatch phase3_inference/slurm/run_inference_array.sh
```

## Output schema (one line per question)

```json
{
  "question_id": "risk_radiorag_42",
  "dataset": "risk_radiorag",
  "correct_answer": "D",
  "model": "Qwen3-32B",
  "conditions": {
    "zero_shot": {
      "greedy":     { "raw_output": "...", "parsed_answer": "D",
                      "input_tokens": 120, "output_tokens": 1,
                      "elapsed_s": 0.23, "error": null },
      "stochastic": { "raw_outputs": [...], "parsed_answers": [...],
                      "majority_vote": "D", "vote_counts": {...},
                      "input_tokens": 120, "elapsed_s": 1.2, "error": null }
    },
    "top_1":  { "..." },
    "top_5":  { "..." },
    "top_10": { "..." },
    "context_32k":  { "..." },
    "context_100k": { "..." },
    "context_max":  { "..." }
  }
}
```

**Note on `context_max`.** `context_max` is derived from each model's
`context_limits["max"]` in `config.py`, computed as
`max_model_len − 2048 (question) − 4096 (max_new_tokens) − 1000 (margin)`.
For models with short native windows it can coincide with (or fall below)
`context_32k` / `context_100k`; for example, Qwen2.5 (32 k window) has
`context_max ≈ 25 624`, so only `context_32k` and `context_max` exist and
they carry nearly the same payload. Llama-4-Scout, by contrast, has a
`context_max` of about 10 M tokens but only 6M tokens were used because the radiopedia dataset "only" has 6M tokens. Conditions whose budget cannot be
satisfied by the model are simply omitted from the per-question record.

Phase 4 (`run_answer_check.py`) then re-parses every `raw_output` /
`raw_outputs` entry with an LLM judge to recover answers from free-text
responses and to mark abstentions.

## Parse-rate monitoring

`02_run_inference.py` warns when the regex letter-extractor fails on more
than 5 % of outputs (`WARNING: parse rate below 95%`). For models where the
warning fires, Phase 4 takes over and re-parses with a stronger judge; the
final results in Phase 5 always use the judge-confirmed letter.
