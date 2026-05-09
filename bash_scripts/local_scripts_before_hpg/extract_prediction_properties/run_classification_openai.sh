#!/bin/bash
# Usage:
# chmod +x run_classification_openai.sh
# bash run_classification_openai.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../properties_extraction_experiments"

START_TIME=$(date +%s)
echo "Start time: $(date)"

for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo "  Extract Properties: Classification"
    echo "  openai/gpt-oss-120b"
    echo "============================================"
    echo ""

    # Step 1 — Extract properties
    python3 extract_properties.py \
        --dataset_path combined_datasets/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate.csv \
        --model_name "openai/gpt-oss-120b" \
        --task_name classification \
        --seed $seed

    # Step 2 — Evaluate against ground truth
    python3 evaluate_properties_extraction.py \
        --y_path classification_results/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate/extract_properties/ground_truth/seed${seed}/llama-3.1-8b-instant/extracted_properties.csv \
        --y_hat_path classification_results/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate/extract_properties/classification/seed${seed}/openai_gpt-oss-120b/extracted_properties.csv \
        --model_name "openai/gpt-oss-120b" \
        --seed $seed

done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo "✓ Finished: Classification + Evaluation — openai/gpt-oss-120b"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"