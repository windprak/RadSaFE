#!/bin/bash -l
#SBATCH --job-name=phase4_check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH -o ${WORKSPACE}/phase4_checking_results/logs/out-%x-%j-on-%N.out
#SBATCH --export=NONE

# ── Environment ────────────────────────────────────────────────────────────────
unset SLURM_EXPORT_ENV
ulimit -n 131072
export VLLM_CACHE_ROOT=$TMPDIR
export HF_HOME=$TMPDIR/huggingface
export TRANSFORMERS_CACHE=$TMPDIR/huggingface
export XDG_CACHE_HOME=$TMPDIR/.cache
export https_proxy="${HTTP_PROXY:-}"
export http_proxy="${HTTP_PROXY:-}"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"

export TMPDIR=${TMPDIR:-/tmp}
export HOME=$TMPDIR
export TRITON_CACHE_DIR=$TMPDIR/.triton
export OUTLINES_CACHE_DIR=$TMPDIR
export HF_HOME=$TMPDIR/huggingface

# ── Config ─────────────────────────────────────────────────────────────────────
VLLM_PORT=6000
WORKSPACE="${WORKSPACE}"
PHASE4_DIR="${WORKSPACE}/phase4_checking_results"
VLLM_SIF="${MODELS_DIR}/vllm-openai_gemma4.sif"
INFERENCE_SIF="${WORKSPACE}/phase3_inference/phase3_inference.sif"

CHECKER_MODEL_NAME="Mistral-Small-4-119B-2603"
CHECKER_MODEL_HOST_PATH="${MODELS_DIR}/${CHECKER_MODEL_NAME}"
CONTAINER_MODEL_PATH="/models/${CHECKER_MODEL_NAME}"

MAX_MODEL_LEN=131072        # Mistral-Small-4 native
TP=4
MAX_NUM_SEQS=128
CONCURRENCY=${CONCURRENCY:-128}

# Optional model list (one model dir name per line). Pass via env:
#   MODELS_FILE=/path/to/models_half_1.txt sbatch run_check.sh
MODELS_FILE=${MODELS_FILE:-}

mkdir -p "${PHASE4_DIR}/logs"

MODELS_ARG=""
if [ -n "${MODELS_FILE}" ]; then
    if [ ! -f "${MODELS_FILE}" ]; then
        echo "ERROR: MODELS_FILE not found: ${MODELS_FILE}"
        exit 1
    fi
    MODELS_ARG="--models-file ${MODELS_FILE}"
fi

echo "============================================================"
echo "  Phase 4 — Answer verification"
echo "  Checker model : ${CHECKER_MODEL_NAME}"
echo "  Model path    : ${CHECKER_MODEL_HOST_PATH}"
echo "  Concurrency   : ${CONCURRENCY}"
echo "  Models file   : ${MODELS_FILE:-<all>}"
echo "  Node          : $(hostname)"
echo "  GPUs          : ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

# ── Start vLLM ────────────────────────────────────────────────────────────────
echo "[$(date)] Starting vLLM server..."

apptainer exec --nv \
    --bind ${MODELS_DIR}:/models \
    --bind ${WORKSPACE}:/workspace \
    ${VLLM_SIF} \
    vllm serve "${CONTAINER_MODEL_PATH}" \
        --tensor-parallel-size ${TP} \
        --max-model-len ${MAX_MODEL_LEN} \
        --host 0.0.0.0 \
        --port ${VLLM_PORT} \
        --gpu-memory-utilization 0.92 \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --trust-remote-code \
        --allow-deprecated-quantization &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# ── Wait for vLLM ready ───────────────────────────────────────────────────────
READY=0
for i in $(seq 1 240); do
    sleep 5
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] vLLM ready after $((i*5))s"
        READY=1
        break
    fi
done
if [ ${READY} -eq 0 ]; then
    echo "ERROR: vLLM did not start in time."
    kill ${VLLM_PID} 2>/dev/null
    exit 1
fi

# ── Run checker ───────────────────────────────────────────────────────────────
echo "[$(date)] Running answer checker..."

apptainer exec --nv \
    --bind ${WORKSPACE}:/workspace \
    --bind ${MODELS_DIR}:/models \
    ${INFERENCE_SIF} \
    python3 /workspace/phase4_checking_results/scripts/run_answer_check.py \
        --port ${VLLM_PORT} \
        --concurrency ${CONCURRENCY} \
        --checker-model-path ${CONTAINER_MODEL_PATH} \
        --skip-wait \
        ${MODELS_ARG} \
        ${EXTRA_CHECK_ARGS:-}

EXIT=$?
echo "[$(date)] Checker finished with exit code ${EXIT}"

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo "[$(date)] Stopping vLLM (PID=${VLLM_PID})..."
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null

exit ${EXIT}
