#!/bin/bash
#SBATCH --job-name=ml_spacy_small
#SBATCH --output=logs/ml_pipeline_spacy_small/logs_%x_%j.out
#SBATCH --error=logs/ml_pipeline_spacy_small/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

mkdir -p logs/ml_pipeline_spacy_small

cd ../../../../prediction_classification_experiments-v2

source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

EXPERIMENT="tolsa_naacl_2026_2027-final"
EMBEDDING_MODEL="spacy_small"

echo "============================================================"
echo "ML Pipeline (In-Domain Baseline) — ${EMBEDDING_MODEL}"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Embedding Model: ${EMBEDDING_MODEL}"
echo "Current directory: $(pwd)"
echo ""

# ============================================================

# ============================================================
# TRAIN, TEST & EVALUATE
# ============================================================
echo ""
echo "Running Baseline (Standard) — Seeds 3, 7, 33"
for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""
    echo ">>> Running Baseline (Standard) — ${EMBEDDING_MODEL}"
    python ml-experiment.py \
        --dataset ../data/combined_datasets/tolsa_naacl_2026_2027-final/tolsa_naacl_2026_2027_main_columns_v1.csv \
        --val_size 0.2 \
        --seed $seed \
        --embedding_model ${EMBEDDING_MODEL}
done

# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "All training complete. Aggregating results..."
echo "======================================"
mkdir -p ../data/classification_results/${EXPERIMENT}/averaged/

python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --embedding_model ${EMBEDDING_MODEL} \
    --experiments seed3 seed7 seed33

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ ${EMBEDDING_MODEL} experiments completed for seeds: 3, 7, 33"
echo "✓ Results aggregated and saved"
echo ""