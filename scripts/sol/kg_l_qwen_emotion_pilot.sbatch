#!/bin/bash
# ==============================================================================
# ASU SOL Supercomputer SLURM Job Script: Phase 3E Dedicated Emotion Pilot
# ==============================================================================
# Job name and resource allocation
#SBATCH --job-name=plotgraph_emotion_pilot
#SBATCH --output=logs/sol/%x_%j.out
#SBATCH --error=logs/sol/%x_%j.err
#SBATCH -q class
#SBATCH -p general
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # 4 CPU cores for data loading and vLLM worker threads
#SBATCH --mem=32G                      # 32 GB host RAM
#SBATCH --time=01:00:00                 # 1 hour wallclock limit (30 movies take ~2-4 mins)
# ==============================================================================

set -euo pipefail

# Environment name is configurable via PLOTGRAPH_CONDA_ENV, defaulting to 'vllm312'
CONDA_ENV="${PLOTGRAPH_CONDA_ENV:-vllm312}"
# Emotion prompt version is configurable via PLOTGRAPH_EMOTION_PROMPT_VERSION, defaulting to 'v1.0.0'
EMOTION_PROMPT_VERSION="${PLOTGRAPH_EMOTION_PROMPT_VERSION:-v1.0.0}"

echo "========================================================================"
echo "      ASU SOL SLURM EXECUTION: PLOTGRAPH DEDICATED EMOTION PILOT (PHASE 3E)"
echo "========================================================================"
echo "Job Name      : ${SLURM_JOB_NAME:-plotgraph_emotion_pilot}"
echo "Job ID        : ${SLURM_JOB_ID:-manual}"
echo "Compute Node  : $(hostname)"
echo "Target Env    : ${CONDA_ENV}"
echo "Emotion Prompt: ${EMOTION_PROMPT_VERSION}"
echo "Execution Time: $(date)"
echo "Working Dir   : $(pwd)"
echo "========================================================================"

# 1. Create logs directory for SLURM stdout/stderr
mkdir -p logs/sol

# 2. Environment Variables & Hugging Face Cache Setup
# CRITICAL: Place Hugging Face cache on scratch storage to avoid exceeding $HOME quotas
export SCRATCH_DIR="${SCRATCH:-/scratch/${USER}}"
export HF_HOME="${HF_HOME:-${SCRATCH_DIR}/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Hugging Face Cache Dir : ${HF_HOME}"
mkdir -p "${HF_HOME}"

# 3. Activate Conda Environment ('vllm312')
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda/etc/profile.d/conda.sh"
elif [ -f "${SCRATCH_DIR}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${SCRATCH_DIR}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

if command -v conda &> /dev/null; then
    echo "Activating Conda environment: ${CONDA_ENV}"
    conda activate "${CONDA_ENV}"
elif [ -f ".venv/bin/activate" ]; then
    echo "Fallback: Activating local virtual environment: .venv"
    source .venv/bin/activate
else
    echo "WARNING: Could not find conda or local .venv to activate."
fi

# 4. Cluster & GPU Node Diagnostics
echo ""
echo "--- Environment & GPU Diagnostics ---"
echo "Requested Conda Env : ${CONDA_ENV}"
echo "Active Conda Env    : ${CONDA_DEFAULT_ENV:-none}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi command not found."
fi

echo ""
echo "--- Python & CUDA Verification ---"
python -c "
import os
import sys
print(f'Active Conda Env : {os.environ.get(\"CONDA_DEFAULT_ENV\", \"none\")}')
print(f'Python Executable: {sys.executable}')
print(f'Python Version   : {sys.version.split()[0]}')
try:
    import torch
    print(f'PyTorch Version  : {torch.__version__}')
    print(f'CUDA Available   : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU Device Name  : {torch.cuda.get_device_name(0)}')
        print(f'GPU Count        : {torch.cuda.device_count()}')
    else:
        print('WARNING: PyTorch cannot detect any CUDA GPUs.')
except ImportError:
    print('WARNING: torch is not installed.')

try:
    import vllm
    print(f'vLLM Version     : {vllm.__version__}')
except ImportError:
    print('WARNING: vllm is not installed.')
"

# 5. Execute Phase 3E Dedicated Emotion Pilot with vLLM Backend
echo ""
echo "--- Starting Dedicated Qwen2.5-7B-Instruct Emotion Pilot (${EMOTION_PROMPT_VERSION}) ---"
python scripts/run_phase3e_emotion_pilot.py \
    --config configs/kg_l.yaml \
    --backend vllm \
    --prompt-version "${EMOTION_PROMPT_VERSION}"

echo ""
echo "========================================================================"
echo "Job completed successfully at: $(date)"
echo "Review output artifacts at: data/processed/ml-1m/kg_l/emotion_pilot_v1/"
echo "========================================================================"
