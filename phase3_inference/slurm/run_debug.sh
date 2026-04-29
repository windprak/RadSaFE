#!/bin/bash -l
#SBATCH --job-name=rag_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --partition=h100
#SBATCH --time=02:00:00
#SBATCH -o ${WORKSPACE}/phase3_inference/logs/out-debug-%j-on-%N.out
#SBATCH --export=NONE

# ── Debug job: runs Qwen2.5-0.5B-Instruct (model index 0) ─────────────────────
# Use this to validate the full pipeline before submitting the array job.

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

MODEL_IDX=0     # Qwen2.5-0.5B-Instruct
VLLM_PORT=6000
WORKSPACE="${WORKSPACE}"
VLLM_SIF="${MODELS_DIR}/vllm-openai_gptoss.sif"
INFERENCE_SIF="${WORKSPACE}/phase3_inference/phase3_inference.sif"

MODEL_PATH="${MODELS_DIR}/Qwen2.5-0.5B-Instruct"
MODEL_NAME="Qwen2.5-0.5B-Instruct"
MAX_LEN=32768

echo "============================================================"
echo "  DEBUG RUN — Model: $MODEL_NAME"
echo "  Node: $(hostname) | GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# ── Start vLLM ─────────────────────────────────────────────────────────────────
# Convert host path to container path for vLLM
MODEL_NAME_ONLY=$(basename "$MODEL_PATH")
CONTAINER_MODEL_PATH="/models/$MODEL_NAME_ONLY"

apptainer exec --nv \
    --bind ${MODELS_DIR}:/models \
    --bind $WORKSPACE:/workspace \
    $VLLM_SIF \
    vllm serve "$CONTAINER_MODEL_PATH" \
        --tensor-parallel-size 1 \
        --max-model-len $MAX_LEN \
        --host 0.0.0.0 \
        --port $VLLM_PORT \
        --disable-log-requests \
        --gpu-memory-utilization 0.92 &

VLLM_PID=$!

# ── Wait ────────────────────────────────────────────────────────────────────────
READY=0
for i in $(seq 1 60); do
    sleep 5
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] vLLM ready after $((i*5))s"
        READY=1; break
    fi
done
[ $READY -eq 0 ] && { echo "ERROR: vLLM timeout"; kill $VLLM_PID; exit 1; }

# ── Smoke-test API ──────────────────────────────────────────────────────────────
echo "[$(date)] Smoke-testing vLLM API..."
curl -s http://localhost:${VLLM_PORT}/v1/models | python3 -m json.tool | head -20

# ── Run inference (only radiology_dr for speed) ────────────────────────────────
echo "[$(date)] Running inference..."
apptainer exec --nv \
    --bind $WORKSPACE:/workspace \
    $INFERENCE_SIF \
    python3 /workspace/phase3_inference/scripts/02_run_inference.py \
        --model-index $MODEL_IDX \
        --port $VLLM_PORT \
        --skip-wait

INFERENCE_EXIT=$?
echo "[$(date)] Inference exit code: $INFERENCE_EXIT"

# ── Quick evaluation ────────────────────────────────────────────────────────────
if [ $INFERENCE_EXIT -eq 0 ]; then
    echo "[$(date)] Running quick evaluation..."
    apptainer exec --nv \
        --bind $WORKSPACE:/workspace \
        $INFERENCE_SIF \
        python3 /workspace/phase3_inference/scripts/03_evaluate.py \
            --model $MODEL_NAME
fi

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "[$(date)] Debug job complete."
exit $INFERENCE_EXIT
