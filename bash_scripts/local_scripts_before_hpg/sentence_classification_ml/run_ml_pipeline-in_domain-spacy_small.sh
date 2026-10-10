#!/bin/bash
# Usage:
    # chmod +x run_ml_pipeline-in_domain-spacy_small.sh
    # bash run_ml_pipeline-in_domain-spacy_small.sh

set -euo pipefail

cd ../../../prediction_classification_experiments-v2

echo "Starting ML Pipeline (In-Domain Baseline) — spacy_small"
echo "Current directory: $(pwd)"

# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT="tolsa_naacl_2026_2027-final"
OUTPUT_DIR="../data/classification_results/naacl_2026_submission"
DATASET="../data/combined_datasets/tolsa_naacl_2026_2027-final/tolsa_naacl_2026_2027_main_columns_v1.csv"
EMBEDDING_MODEL="spacy_small"
SEEDS=(3 7 33)

echo "Dataset ready: ${DATASET}"

# ============================================================
# TRAIN, TEST & EVALUATE
# ============================================================

echo ""
echo "Running Baseline — Seeds: ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
echo ""
echo "============================================================"
echo "SEED: ${seed}"
echo "============================================================"
echo ""
echo ">>> Running ${EMBEDDING_MODEL}"


python ml-experiment.py \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}/${EXPERIMENT}" \
    --val_size 0.2 \
    --seed "${seed}" \
    --embedding_model "${EMBEDDING_MODEL}"

done

# ============================================================
# AGGREGATE RESULTS
# ============================================================

echo ""
echo "======================================"
echo "All training complete. Aggregating results..."
echo "======================================"

python average_classification_results.py \
    --mode single \
    --experiment "${EXPERIMENT}" \
    --model_type ml \
    --embedding_model "${EMBEDDING_MODEL}" \
    --experiments seed3 seed7 seed33 \
    --results_dir "${OUTPUT_DIR}"

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "Model: ${EMBEDDING_MODEL}"
echo "Seeds completed: ${SEEDS[*]}"
echo "Results directory: ${OUTPUT_DIR}/${EXPERIMENT}"
echo "======================================"
