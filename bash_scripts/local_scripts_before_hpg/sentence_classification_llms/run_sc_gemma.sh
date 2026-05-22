# NOTE: This script only classifies. Run after all LLM classifiers complete, then run run_sc_avg.sh to combine and average.
#!/bin/bash
# run_sc_gemma.sh - Run Gemma sentence classification for all seeds
#
# Usage:
#   chmod +x run_sc_gemma.sh
#   bash run_sc_gemma.sh
set -e
cd ../../../prediction_classification_experiments-v2

EXPERIMENT="emnlp_2026_results_$(date +%Y-%m-%d)"
BASE_RESULTS="../data/classification_results/${EXPERIMENT}"

echo "============================================================"
echo "     SENTENCE CLASSIFICATION (LOCAL): gemma-3-27b-it"
echo "============================================================"
for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""
    python llm-classifiers.py \
        --model_name gemma-3-27b-it \
        --test_dataset ${BASE_RESULTS}/seed${seed}/in_domain/x_y_test_set.csv \
        --label_column 'Ground Truth' \
        --seed $seed
done

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ Gemma classification completed for seeds: 3, 7, 33"
echo ""