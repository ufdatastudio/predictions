#!/bin/bash
# Usage:
# chmod +x run_classification_llama_versatile.sh
# bash run_classification_llama_versatile.sh

echo "============================================"
echo "Extract Properties: Classification"
echo "llama-3.3-70b-versatile"
echo "============================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../script_experiments"

START_TIME=$(date +%s)
echo "Start time: $(date)"

# Step 1 — Extract properties
python3 extract_properties.py \
    --dataset_path combined_datasets/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate.csv \
    --model_name "llama-3.3-70b-versatile" \
    --task_name classification

# Step 2 — Evaluate against ground truth
python3 evaluate_properties_extraction.py \
    --y_path extract_properties/ground_truth/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate/llama-3.1-8b-instant/extracted_properties.csv \
    --y_hat_path extract_properties/classification/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate/llama-3.3-70b-versatile/extracted_properties.csv \
    --model_name "llama-3.3-70b-versatile" \
    --seed 42

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo "✓ Finished: Classification + Evaluation — llama-3.3-70b-versatile"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"