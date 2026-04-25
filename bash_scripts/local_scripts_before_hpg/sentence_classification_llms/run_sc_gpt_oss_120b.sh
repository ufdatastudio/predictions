#!/bin/bash
# run_llm_classifiers.sh - Run LLM sentence classification for all seeds and average
#
# Usage:
#   chmod +x run_sc_gpt_oss_120b.sh
#   bash run_sc_ggpt_oss_120b.sh

set -e

cd ../../../prediction_classification_experiments-v2

# EXPERIMENT="synthetic-fpb-chronicle2050-yt-news-timebank_$(date +%Y-%m-%d)"
EXPERIMENT="synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_2026-04-22"
BASE_RESULTS="../data/classification_results/${EXPERIMENT}"

echo "============================================================"
echo "     SENTENCE CLASSIFICATION (LOCAL): gpt_oss_120b"
echo "============================================================"

for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""

    python llm-classifiers.py \
        --model_name gpt-oss-120b \
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