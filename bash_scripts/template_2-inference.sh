#!/bin/bash
#SBATCH --account ufdatastudios
#SBATCH --job-name audioflamingo3-medical_nli
#SBATCH --output=/orange/ufdatastudios/c.okocha/Afro_entailment/outputs/AudioFlamingo3/medical_nli/logs/infer_%j.out
#SBATCH --error=/orange/ufdatastudios/c.okocha/Afro_entailment/outputs/AudioFlamingo3/medical_nli/logs/infer_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=c.okocha@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition hpg-b200

set -euo pipefail

echo "===== GPU Info ====="
nvidia-smi || true

# Go to model project root (customize this path)
cd /orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_3

# Load CUDA toolkit
module load cuda/12.8.1 || true

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Activate project venv
source .venv_af3/bin/activate

# Fix NumPy/Numba compatibility (Numba requires NumPy <= 2.0, only for Audio Flamingo models)
# This check will be replaced by generate_slurm_scripts.py for Audio Flamingo models
NUMPY_FIX_NEEDED="true"
if [[ "$NUMPY_FIX_NEEDED" == "true" ]]; then
    echo "Checking NumPy version for Numba compatibility..."
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "0.0.0")
    NUMPY_MAJOR=$(echo $NUMPY_VERSION | cut -d. -f1)
    NUMPY_MINOR=$(echo $NUMPY_VERSION | cut -d. -f2)
    if [ "$NUMPY_MAJOR" -gt 2 ] || ([ "$NUMPY_MAJOR" -eq 2 ] && [ "$NUMPY_MINOR" -gt 0 ]); then
        echo "Downgrading NumPy from $NUMPY_VERSION to <2.1 for Numba compatibility..."
        uv pip install "numpy<2.1" --quiet || pip install "numpy<2.1" --quiet
        echo "NumPy downgraded successfully"
    else
        echo "NumPy version $NUMPY_VERSION is compatible with Numba"
    fi
fi

# Ensure transformers is up-to-date for Audio Flamingo 3 (if needed)
# Uncomment the following lines if transformers needs to be upgraded:
# pip install --upgrade pip
# pip install --upgrade transformers>=4.60.0 accelerate

# Paths - CUSTOMIZE THESE
JSONL_PATH="/orange/ufdatastudios/c.okocha/Afro_entailment/result/Entailment/Medical/Llama/nli/medical_nli.jsonl"
AUDIO_DIR="/orange/ufdatastudios/c.okocha/Afro_entailment/Audio/medical"
medical_nli="medical_nli"
OUTPUT_BASE="/orange/ufdatastudios/c.okocha/Afro_entailment/outputs"
AudioFlamingo3="AudioFlamingo3"
OUT_DIR="${OUTPUT_BASE}/${AudioFlamingo3}/${medical_nli}"
RESULTS_DIR="${OUT_DIR}/results"
LOGS_DIR="${OUT_DIR}/logs"
OUTPUT_PREFIX="medical_nli"
OUT_JSONL="${RESULTS_DIR}/${AudioFlamingo3}_${OUTPUT_PREFIX}.jsonl"
AudioFlamingo3_PATH="nvidia/audio-flamingo-3-hf"
TASK="nli"

# Ensure output dirs exist
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

# Use /orange for model caches
BASE_DIR="/orange/ufdatastudios/c.okocha/child__speech_analysis"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${BASE_DIR}/.cache/transformers"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

# Performance knobs
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face token
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]] && [[ -f "/orange/ufdatastudios/c.okocha/.cache/huggingface/token" ]]; then
  export HUGGINGFACE_HUB_TOKEN=$(cat /orange/ufdatastudios/c.okocha/.cache/huggingface/token)
fi

echo "===== Running inference ====="
echo "Model: ${AudioFlamingo3_PATH}"
echo "Task: ${TASK}"
echo "Input: ${JSONL_PATH}"
echo "Audio: ${AUDIO_DIR}"
echo "Output: ${OUT_JSONL}"

srun -N 1 --gpus 1 --cpus-per-task 8 \
  python infer_jsonl.py \
    --model_path "${AudioFlamingo3_PATH}" \
    --variant base \
    --jsonl_path "${JSONL_PATH}" \
    --audio_dir "${AUDIO_DIR}" \
    --task "${TASK}" \
    --out_jsonl "${OUT_JSONL}" \
    --max_new_tokens 512

echo "Done. Results in ${RESULTS_DIR}, logs in ${LOGS_DIR}"