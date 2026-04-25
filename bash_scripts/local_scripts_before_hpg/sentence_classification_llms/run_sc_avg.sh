#!/bin/bash
# run_sc_avg.sh - Average LLM classification results across seeds
#
# Usage:
#   chmod +x run_sc_avg.sh
#   bash run_sc_avg.sh

set -e

cd ../../../prediction_classification_experiments-v2

EXPERIMENT="synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_$(date +%Y-%m-%d)"
EXPERIMENT="synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_2026-04-22"


echo "============================================================"
echo "     AVERAGE LLM CLASSIFICATION RESULTS"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo ""

python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --experiments seed3 seed7 seed33 \
    --model_type llm

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ LLM results averaged for seeds: 3, 7, 33"
echo "✓ Results saved to: ../data/classification_results/${EXPERIMENT}/averaged/"
echo ""