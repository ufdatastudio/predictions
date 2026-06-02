#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=nanochat-multigpu
#SBATCH --output=/orange/ufdatastudios/dj.brinkley/predictions/bash_scripts/logs/multigpu_%j.out
#SBATCH --error=/orange/ufdatastudios/dj.brinkley/predictions/bash_scripts/logs/multigpu_%j.err
#SBATCH --nodes=1 (can be 1 to..)
#SBATCH --gpus=8 (1 to 8)
#SBATCH --time=24:00:00
#SBATCH --mem=256GB (x to 256)
#SBATCH --cpus-per-task=32 (1 to 32)
#SBATCH --mail-user=dj.brinkley@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-b200, hpg-l4

set -euo pipefail

echo "===== NANOCHAT MULTI-GPU TRAINING (8x B200) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start Time: $(date)"

# Function to log with timestamp
log_with_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

echo "===== GPU Info ====="
nvidia-smi || true

# Load CUDA toolkit
log_with_time "Loading CUDA module..."
module load cuda/12.8.1 || true

export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Go to project root
cd /orange/ufdatastudios/c.okocha/chibu-chat

# ============================================================================
# TORCH.COMPILE DISABLED for stability with multi-GPU
# ============================================================================
export TORCH_COMPILE_DISABLE=1

# ============================================================================
# Multi-GPU / Distributed Training Settings
# ============================================================================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# NCCL settings for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# ============================================================================
# CUDA Settings
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# ============================================================================
# Environment Setup
# ============================================================================
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export WANDB_RUN="multigpu_8xB200_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M)"
mkdir -p $NANOCHAT_BASE_DIR

log_with_time "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
log_with_time "WANDB_RUN: $WANDB_RUN"
log_with_time "TORCH_COMPILE_DISABLE: $TORCH_COMPILE_DISABLE"
log_with_time "Number of GPUs: 8"

# Activate virtual environment
log_with_time "Activating virtual environment..."
source .venv/bin/activate

# Install nanochat package
log_with_time "Installing NanoChat package..."
uv pip install -e . --quiet

# Verify environment
log_with_time "===== Environment Check ====="
python --version
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    props = torch.cuda.get_device_properties(i)
    print(f'    Memory: {props.total_memory / 1024**3:.1f} GB')
" || true

# Verify checkpoint exists
log_with_time "===== Checkpoint Check ====="
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d20"
if [ -f "$CHECKPOINT_DIR/model_001000.pt" ]; then
    log_with_time "✓ Found checkpoint at step 1000"
    ls -lh "$CHECKPOINT_DIR/"
else
    log_with_time "WARNING: Checkpoint not found - will start fresh training"
fi

# WandB setup
log_with_time "===== WANDB Setup ====="
wandb --version
python -c "import wandb; wandb.login(); print('WandB authenticated successfully')" || log_with_time "WandB authentication failed"

# Initialize report
log_with_time "===== Initializing Report System ====="
python -m nanochat.report reset || true

log_with_time "===== Starting Multi-GPU Training ====="
log_with_time "Training parameters:"
log_with_time "  - GPUs: 8x NVIDIA B200"
log_with_time "  - Resuming from step: 1000"
log_with_time "  - torch.compile: DISABLED (stability)"
log_with_time "  - Model depth: 20 layers"
log_with_time "  - Context length: 2048 tokens"
log_with_time "  - Device batch size: 32 per GPU"
log_with_time "  - Total batch size: 524,288 tokens"
log_with_time "  - Expected speedup: ~8x vs single GPU"

log_with_time "Launching distributed training with torchrun..."

# Use torchrun for multi-GPU training
# With 8 GPUs, each processing 32 samples of 2048 tokens = 65536 tokens per GPU
# Total per step: 8 * 65536 = 524288 tokens (matches total_batch_size, so grad_accum=1)
torchrun \
    --standalone \
    --nproc_per_node=8 \
    -m scripts.base_train \
    --depth=20 \
    --max_seq_len=2048 \
    --device_batch_size=32 \
    --total_batch_size=524288 \
    --resume_from_step=1000 \
    --eval_every=250 \
    --save_every=1000 \
    --core_metric_every=2000 \
    --sample_every=2000

TRAIN_EXIT_CODE=$?

log_with_time "===== Training Complete ====="
log_with_time "Exit code: $TRAIN_EXIT_CODE"
log_with_time "End Time: $(date)"

# Final GPU status
nvidia-smi || true

# Show final checkpoint
log_with_time "===== Final Checkpoints ====="
ls -lh "$CHECKPOINT_DIR/" || true

exit $TRAIN_EXIT_CODE