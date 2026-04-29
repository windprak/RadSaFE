# Phase 3 — large-model deployment (≥ 400 B params)

The five largest models in the paper panel do not fit on the standard
NHR@FAU H100 / H200 nodes used for `config.py`'s MODELS list and were
served separately on B200 and MI300X hardware. This file documents the
exact vLLM launch flags and the per-model overrides used for those runs.
The `02_run_inference.py` driver is unchanged — only the endpoint and the
per-model config block differ.

## Shared vLLM launch flags

```
--kv-cache-dtype            fp8
--compilation-config        '{"pass_config":{"fuse_allreduce_rms":true,"eliminate_noops":true}}'
--async-scheduling          true
--enable-prefix-caching     true
--max-num-batched-tokens    32768
--gpu-memory-utilization    0.98
```

`--max-model-len`, `--max-num-seqs`, and `--tensor-parallel-size` are set
per model (see below).

## Per-model config entries

These are drop-in additions to `MODELS` in `config.py` for a large-model
endpoint. They reuse the same schema as the standard entries; only
`gpu_type`, `concurrency`, and `reasoning` are extra metadata used by the
launcher.

```python
{   # H200, reasoning
    "name":           "deepseek-r1",
    "path":           "/scratch/models/DeepSeek-R1",
    "served_name":    "deepseek-r1",
    "tp":             8,
    "max_model_len":  131_072,
    "max_num_seqs":   32,
    "context_limits": {"100k": 100_000},
    "gpu_type":       "h200",
    "vision":         False,
    "concurrency":    32,
    "reasoning":      True,
},
{   # H200
    "name":           "deepseek-v3.2",
    "path":           "/scratch/models/DeepSeek-V3.2",
    "served_name":    "deepseek-v3.2",
    "tp":             8,
    "max_model_len":  131_072,
    "max_num_seqs":   64,
    "context_limits": {"100k": 100_000},
    "gpu_type":       "h200",
    "vision":         False,
    "concurrency":    32,
    "reasoning":      False,
},
{   # B200, BF16 full-precision Llama-4-Scout, 6M-token context
    "name":           "llama4-scout-instruct-b200",
    "path":           "/scratch/models/Llama-4-Scout-17B-16E-Instruct",
    "served_name":    "llama4-scout-instruct-b200",
    "tp":             8,
    "max_model_len":  6_291_456,
    "max_num_seqs":   4,
    "context_limits": {"6m": 6_000_000, "100k": 100_000},
    "gpu_type":       "b200",
    "vision":         False,
    "concurrency":    1,
    "reasoning":      False,
},
```

## GPU-type summary

| Model                           | GPUs     | TP | `max_model_len` | `max_num_seqs` |
| ------------------------------- | -------- | -- | --------------- | -------------- |
| DeepSeek-R1                     | 8×H200   | 8  | 131 072         | 32             |
| DeepSeek-V3.2                   | 8×H200   | 8  | 131 072         | 64             |
| Llama-4-Scout-17B-16E-Instruct  | 8×B200   | 8  | 6 291 456       | 4              |
| Mistral-Large-3-675B-Instruct   | 8×MI300X/8xH200 | 8  | 131 072         | —              |
| Qwen3-VL-235B-A22B-Instruct     | 8×MI300X | 8  | 131 072         | —              |

For Llama-4-Scout the 6 M-token context is what enables the
`context_max` condition for that model; all other conditions are served
from the same endpoint with no reconfiguration.
