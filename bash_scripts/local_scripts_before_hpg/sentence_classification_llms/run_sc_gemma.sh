#!/bin/bash
# run_llm_classifiers.sh - Run LLM sentence classification for all seeds and average
#
# Usage:
#   chmod +x run_sc_gemma.sh
#   bash run_sc_gemma.sh

set -e

cd ../../../prediction_classification_experiments-v2

# EXPERIMENT="synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_$(date +%Y-%m-%d)"
EXPERIMENT="synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_2026-04-22"
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
echo "✓ LLM classification completed for seeds: 3, 7, 33"
# echo "✓ Results aggregated and saved"
echo ""