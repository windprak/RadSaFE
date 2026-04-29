#!/bin/bash
#SBATCH --job-name=model_download
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/hnvme/workspace/unrz108h-llm_hub/logs/download-%x-%j.out

# Cache/tmp configuration
export TMPDIR=${TMPDIR:-/tmp}
export HF_HOME=$TMPDIR/hf_home
export HF_HUB_CACHE=$TMPDIR/hf_hub_cache
export HUGGINGFACE_HUB_CACHE=$TMPDIR/hf_hub_cache
export XDG_CACHE_HOME=$TMPDIR/xdg_cache
export https_proxy="${HTTP_PROXY:-}"
export http_proxy="${HTTP_PROXY:-}"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export HF_TOKEN="${HF_TOKEN:-}"



SIF=${WORKSPACE}/phase2_rag_context/rag_environment.sif
MODELS_DIR=${MODELS_DIR}

mkdir -p /hnvme/workspace/unrz108h-llm_hub/logs

MODELS=(
    # Mistral
    "mistralai/Ministral-3-3B-Instruct-2512"
    "mistralai/Ministral-3-8B-Instruct-2512"
    "mistralai/Ministral-3-14B-Instruct-2512"
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    "mistralai/Mistral-Small-4-119B-2603"
    # Qwen
    "Qwen/Qwen3.5-0.8B"
    "Qwen/Qwen3.5-2B"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-27B"
    "Qwen/Qwen3.5-35B-A3B"
    "Qwen/Qwen3.5-122B-A10B"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL")
    echo "========================================"
    echo "Downloading: $MODEL -> $MODELS_DIR/$MODEL_NAME"
    echo "========================================"

    apptainer exec --nv \
        --bind $MODELS_DIR:/models \
        --bind $TMPDIR:$TMPDIR \
        --env TMPDIR=$TMPDIR \
        --env HF_HOME=$HF_HOME \
        --env HF_HUB_CACHE=$HF_HUB_CACHE \
        --env HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE \
        --env XDG_CACHE_HOME=$XDG_CACHE_HOME \
        $SIF \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='/models/$MODEL_NAME', local_dir_use_symlinks=False, max_workers=10)
"

    if [ $? -eq 0 ]; then
        echo "✓ Done: $MODEL_NAME"
    else
        echo "✗ FAILED: $MODEL_NAME — continuing..."
    fi
done

echo "All downloads complete."