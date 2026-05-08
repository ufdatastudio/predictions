#!/bin/bash
# Usage:
# chmod +x run_ground_truth.sh
# bash run_ground_truth.sh

echo "============================================"
echo "Extract Properties: Ground Truth"
echo "llama-3.1-8b-instant"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../script_experiments"

START_TIME=$(date +%s)
echo "Start time: $(date)"

python3 extract_properties.py \
    --dataset_path combined_datasets/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate.csv \
    --model_name "llama-3.1-8b-instant" \
    --task_name ground_truth

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo "✓ Finished: Ground Truth — llama-3.1-8b-instant"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"