# HPC environment

This pipeline was developed on a heterogeneous SLURM cluster with the
following hardware classes:

| Class                | Used for                                   |
| -------------------- | ------------------------------------------ |
| 8×NVIDIA H200/B200/MI300X  | vLLM inference for ≥ 30 B-parameter models |
| 1–4×NVIDIA H100      | smaller open-weight models                 |
| CPU partition        | LLM-judge re-parsing, statistics            |

You will need:

- **Apptainer ≥ 1.2** for the runtime images.
- **SLURM ≥ 20.11** if you want to use the supplied job scripts as-is.
- A local cache of HuggingFace model weights (`MODELS_DIR`) for vLLM.

## Apptainer images

Two images cover the entire pipeline:

```bash
apptainer build environment/rag.sif        environment/rag.def         # phase 2
apptainer build environment/inference.sif  environment/inference.def   # phases 3-5
```

`inference.sif` provides PyTorch + vLLM + transformers (for inference) plus
numpy (for analysis), so the *same* image runs phases 3, 4, and 5.

## SLURM conventions used in `phase{3,4}/slurm/*.sh`

- `--output=$WORKSPACE/phase<N>/logs/...` for stdout/stderr.
- `--bind $WORKSPACE:$WORKSPACE` and `--bind $MODELS_DIR:/models` for the
  apptainer mounts.
- vLLM listens on `localhost:6000` inside the container; the calling Python
  script targets `http://localhost:6000/v1`.

## Adapting to your cluster

Most scripts read a small set of environment variables:

| Variable             | Meaning                                                    |
| -------------------- | ---------------------------------------------------------- |
| `WORKSPACE`          | Absolute path of the checked-out repository                |
| `MODELS_DIR`         | Local HuggingFace model cache (mounted as `/models`)       |
| `HTTP_PROXY`         | Outbound HTTP proxy if your cluster requires one           |
| `HF_TOKEN`           | HuggingFace token, needed to pull gated models             |
| `VLLM_SIF`           | Override path to the vLLM apptainer image                  |

If you don't have SLURM, the inner `apptainer exec ... python3 ...`
invocations in each `slurm/*.sh` are self-contained shell commands and can
be executed directly.

## Single-machine fallback

For a *small* single-GPU run (debugging, sanity checks):

```bash
# 1) Pick one model and one condition, edit phase3_inference/config.py
#    to leave only that entry in `MODELS` and one of `FIXED_CONDITIONS`.
# 2) Start vLLM manually:
apptainer exec --nv --bind $MODELS_DIR:/models environment/inference.sif \
    vllm serve /models/<your-model> --port 6000

# 3) In another shell, run inference:
apptainer exec environment/inference.sif \
    python3 phase3_inference/scripts/02_run_inference.py \
        --models <your-model> --conditions zero_shot
```

Reproducing the full leaderboard requires multi-node GPU resources.
