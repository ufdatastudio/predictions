#!/bin/bash
# run_rules_pipeline-in_domain-conll.sh - Run CoNLL-2010 hedge classification for all seeds
#
# Usage:
#   chmod +x run_rules_pipeline-in_domain-conll.sh
#   bash run_rules_pipeline-in_domain-conll.sh

set -e

cd ../../../prediction_classification_experiments-v2

DATE=$(date +%Y-%m-%d)
EXPERIMENT="july_2026_results_${DATE}"

# INPUT: July 2026 dataset
DATASET="../data/combined_datasets/july_2026_results/july_2026_results.csv"

# OUTPUT: save CoNLL experiments under NAACL results
SAVE_PATH="../data/classification_results/naacl_2026_submission"

TEXT_COLUMN="Base Sentence"
LABEL_COLUMN="Ground Truth"

echo "============================================================"
echo "CoNLL-2010 HEDGE PIPELINE"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Dataset: ${DATASET}"
echo "Save path: ${SAVE_PATH}"
echo "Current directory: $(pwd)"
echo ""

# ============================================================
# TEST AND EVALUATE
# ============================================================

for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: ${seed}"
    echo "============================================================"
    echo ""

    python conll-2010-hedge-experiment.py \
        --dataset "${DATASET}" \
        --text_column "${TEXT_COLUMN}" \
        --label_column "${LABEL_COLUMN}" \
        --seed "${seed}" \
        --val_size 0.2 \
        --test_size 0.2 \
        --save_path "${SAVE_PATH}"
done

# ============================================================
# AVERAGE CoNLL RESULTS ACROSS SEEDS
# ============================================================

echo ""
echo "======================================"
echo "All runs complete. Aggregating results..."
echo "======================================"

mkdir -p "${SAVE_PATH}/${EXPERIMENT}/averaged"

python average_classification_results.py \
    --mode single \
    --model_type conll \
    --results_dir "${SAVE_PATH}" \
    --experiment "${EXPERIMENT}" \
    --model_name "conll-2010-hedge" \
    --experiments seed3 seed7 seed33

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ CoNLL-2010 hedge completed for seeds: 3, 7, 33"
echo "✓ CoNLL results saved under: ${SAVE_PATH}/${EXPERIMENT}/"
echo "✓ Averaged results saved under: ${SAVE_PATH}/${EXPERIMENT}/averaged/"
echo ""