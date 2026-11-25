#!/bin/bash
#SBATCH --job-name=sr_factory_120b
#SBATCH --output=logs/sr_factory_gpt_oss_120b/logs_%x_%j.out
#SBATCH --error=logs/sr_factory_gpt_oss_120b/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

# Self-rewarding data factory for TOLSA Task-2 ground-truth augmentation.
# Mirrors bash_scripts/hpg_slurm_scripts/extract_prediction_properties/
# run_llm_gpt_oss_120b-pipeline-ground_truth.sbatch (same venv, log layout).
# The factory is RESUME-SAFE: if the job hits the 24 h limit, resubmit the
# same script and it continues from the last judged candidate.

set -e

mkdir -p logs/sr_factory_gpt_oss_120b

# Repo root (adjust if the script lives elsewhere)
cd <REPO_ROOT>

source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

export GROQ_CLOUD_API_KEY=<KEY_OR_USE_HPG_SECRETS>

MODEL_BACKEND="groq:openai/gpt-oss-120b"
GOLD="data/ground_truth_annotations/ground_truth_final_200.csv"
SEED=3

echo "============================================================"
echo "Self-Rewarding Data Factory - TOLSA Task 2"
echo "Backend: ${MODEL_BACKEND}   Seed: ${SEED}"
echo "============================================================"

# ------------------------------------------------------------
# STEP 1: validate the judge on the 200 human gold examples
# ------------------------------------------------------------
python -m self_rewarding.run_factory validate-gold \
    --backend ${MODEL_BACKEND} \
    --gold ${GOLD} \
    --out runs/validation_gold_gpt_oss_120b \
    --judge-runs 3 --threshold 4.5 --seed ${SEED}

# ------------------------------------------------------------
# STEP 2: generate ~1200 judge-approved examples (resume-safe)
# ------------------------------------------------------------
python -m self_rewarding.run_factory generate \
    --backend ${MODEL_BACKEND} \
    --gold ${GOLD} \
    --out runs/factory_gpt_oss_120b_seed${SEED} \
    --target 1200 --threshold 4.5 --judge-runs 3 --seed ${SEED} \
    --batch-size 10

echo "============================================================"
echo "SELF-REWARDING FACTORY COMPLETE"
echo "============================================================"
