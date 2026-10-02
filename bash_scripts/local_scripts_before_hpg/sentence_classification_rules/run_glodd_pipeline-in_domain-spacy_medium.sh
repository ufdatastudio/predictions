#!/bin/bash
# Glodd & Hristova Step 0 FLS candidate baseline with spacy_medium
# Usage:
#   chmod +x run_glodd_pipeline-in_domain-spacy_medium.sh
#   bash run_glodd_pipeline-in_domain-spacy_medium.sh


set -e

cd ../../../prediction_classification_experiments-v2

DATE=$(date +%Y-%m-%d)
EXPERIMENT="july_2026_results_${DATE}"

DATASET="../data/combined_datasets/july_2026_results/july_2026_results.csv"
SAVE_PATH="../data/classification_results/naacl_2026_submission"

TEXT_COLUMN="Base Sentence"
LABEL_COLUMN="Ground Truth"
EMBEDDING_MODEL="spacy_medium"

echo "============================================================"
echo "GLODD & HRISTOVA STEP 0 CANDIDATE BASELINE - spacy_medium"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Dataset: ${DATASET}"
echo "Save path: ${SAVE_PATH}"
echo "spaCy model: ${EMBEDDING_MODEL}"
echo "Current directory: $(pwd)"
echo ""

for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: ${seed}"
    echo "============================================================"
    echo ""

    python glodd-hristova-step0-experiment.py \
        --dataset "${DATASET}" \
        --save_path "${SAVE_PATH}" \
        --text_column "${TEXT_COLUMN}" \
        --label_column "${LABEL_COLUMN}" \
        --embedding_model "${EMBEDDING_MODEL}" \
        --seed "${seed}" \
        --val_size 0.2 \
        --test_size 0.2
done

# ============================================================
# AVERAGE GLODD RESULTS ACROSS SEEDS
# ============================================================

echo ""
echo "======================================"
echo "All runs complete. Aggregating results..."
echo "======================================"

mkdir -p "${SAVE_PATH}/${EXPERIMENT}/averaged"

python average_classification_results.py \
    --mode single \
    --model_type glodd \
    --results_dir "${SAVE_PATH}" \
    --experiment "${EXPERIMENT}" \
    --experiments seed3 seed7 seed33

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ Glodd & Hristova Step 0 completed for seeds: 3, 7, 33"
echo "✓ Glodd results saved under: ${SAVE_PATH}/${EXPERIMENT}/"
echo "✓ Averaged results saved under: ${SAVE_PATH}/${EXPERIMENT}/averaged/"
echo ""