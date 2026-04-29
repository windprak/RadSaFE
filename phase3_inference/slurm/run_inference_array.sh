#!/bin/bash -l
#SBATCH --job-name=rag_inference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --partition=h100
#SBATCH --array=0-36          # locally hostable open-weight models; indices match MODELS list in config.py
#SBATCH --time=24:00:00
#SBATCH -o ${WORKSPACE}/phase3_inference/logs/out-%x-%A_%a-on-%N.out
#SBATCH --export=NONE

# ── Environment ────────────────────────────────────────────────────────────────
unset SLURM_EXPORT_ENV
ulimit -n 131072

export https_proxy="${HTTP_PROXY:-}"
export http_proxy="${HTTP_PROXY:-}"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export HF_TOKEN="${HF_TOKEN:-}"

export TMPDIR=${TMPDIR:-/tmp}
export HOME=$TMPDIR
export TRITON_CACHE_DIR=$TMPDIR/.triton
export OUTLINES_CACHE_DIR=$TMPDIR
export HF_HOME=$TMPDIR/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export VLLM_CACHE_ROOT=$TMPDIR
export HF_HOME=$TMPDIR/huggingface
export TRANSFORMERS_CACHE=$TMPDIR/huggingface
export XDG_CACHE_HOME=$TMPDIR/.cache
# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_IDX=$SLURM_ARRAY_TASK_ID
VLLM_PORT=6000

WORKSPACE="${WORKSPACE}"
# vLLM image used to serve open-weight models. The default targets the
# gemma-4 build of vllm-openai (newer transformers, supports gemma-4 /
# mistral-4 / etc.); override with VLLM_SIF if you prefer another build.
VLLM_SIF="${VLLM_SIF:-${MODELS_DIR}/vllm-openai_gemma4.sif}"
INFERENCE_SIF="${WORKSPACE}/environment/inference.sif"

# Resolve model path from Python config (index → path)
MODEL_PATH=$(apptainer exec --bind $WORKSPACE:/workspace $INFERENCE_SIF \
    python3 -c "
import sys; sys.path.insert(0, '/workspace/phase3_inference')
from config import MODELS
m = MODELS[${MODEL_IDX}]
print(m['path'])
")
MODEL_NAME=$(apptainer exec --bind $WORKSPACE:/workspace $INFERENCE_SIF \
    python3 -c "
import sys; sys.path.insert(0, '/workspace/phase3_inference')
from config import MODELS
print(MODELS[${MODEL_IDX}]['name'])
")
MAX_LEN=$(apptainer exec --bind $WORKSPACE:/workspace $INFERENCE_SIF \
    python3 -c "
import sys; sys.path.insert(0, '/workspace/phase3_inference')
from config import MODELS
print(MODELS[${MODEL_IDX}]['max_model_len'])
")
IS_VISION=$(apptainer exec --bind $WORKSPACE:/workspace $INFERENCE_SIF \
    python3 -c "
import sys; sys.path.insert(0, '/workspace/phase3_inference')
from config import MODELS
print('1' if MODELS[${MODEL_IDX}]['vision'] else '0')
")
TP=$(apptainer exec --bind $WORKSPACE:/workspace $INFERENCE_SIF \
    python3 -c "
import sys; sys.path.insert(0, '/workspace/phase3_inference')
from config import MODELS
print(MODELS[${MODEL_IDX}]['tp'])
")
MAX_NUM_SEQS=$(apptainer exec --bind $WORKSPACE:/workspace $INFERENCE_SIF \
    python3 -c "
import sys; sys.path.insert(0, '/workspace/phase3_inference')
from config import MODELS
print(MODELS[${MODEL_IDX}].get('max_num_seqs', 256))
")

echo "============================================================"
echo "  SLURM Array Task : $SLURM_ARRAY_TASK_ID"
echo "  Model Index      : $MODEL_IDX"
echo "  Model Name       : $MODEL_NAME"
echo "  Model Path       : $MODEL_PATH"
echo "  Max Model Len    : $MAX_LEN"
echo "  Tensor Parallel  : $TP"
echo "  Node             : $(hostname)"
echo "  GPUs             : $CUDA_VISIBLE_DEVICES"
echo "  Vision model     : $IS_VISION"
echo "============================================================"

# ── Start vLLM server ──────────────────────────────────────────────────────────
echo "[$(date)] Starting vLLM server..."

# Convert host path to container path for vLLM (${MODELS_DIR}/X -> /models/X).
MODEL_NAME_ONLY=$(basename "$MODEL_PATH")
CONTAINER_MODEL_PATH="/models/$MODEL_NAME_ONLY"

apptainer exec --nv \
    --bind ${MODELS_DIR}:/models \
    --bind $WORKSPACE:/workspace \
    $VLLM_SIF \
    vllm serve "$CONTAINER_MODEL_PATH" \
        --tensor-parallel-size $TP \
        --max-model-len $MAX_LEN \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --gpu-memory-utilization 0.92 \
        --max-num-seqs $MAX_NUM_SEQS \
        --trust-remote-code \
        --allow-deprecated-quantization &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# ── Wait for vLLM to be ready ──────────────────────────────────────────────────
echo "[$(date)] Waiting for vLLM to be ready..."
READY=0
for i in $(seq 1 240); do
    sleep 5
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] vLLM is ready after $((i*5))s"
        READY=1
        break
    fi
done

if [ $READY -eq 0 ]; then
    echo "ERROR: vLLM did not start in time. Killing and exiting."
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ── Run inference ──────────────────────────────────────────────────────────────
echo "[$(date)] Running inference for model index $MODEL_IDX ($MODEL_NAME)..."

apptainer exec --nv \
    --bind $WORKSPACE:/workspace \
    --bind ${MODELS_DIR}:/models \
    $INFERENCE_SIF \
    python3 /workspace/phase3_inference/scripts/02_run_inference.py \
        --model-index $MODEL_IDX \
        --port $VLLM_PORT \
        --skip-wait \
        ${EXTRA_INFERENCE_ARGS:-}

INFERENCE_EXIT=$?
echo "[$(date)] Inference finished with exit code $INFERENCE_EXIT"

# ── Cleanup ────────────────────────────────────────────────────────────────────
echo "[$(date)] Stopping vLLM server (PID=$VLLM_PID)..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "[$(date)] Job complete."
exit $INFERENCE_EXIT
