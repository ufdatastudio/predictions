#!/bin/bash
# train_ml_models-in_domain.sh - Baseline with three seeds
#
# Usage:
#   chmod +x train_ml_models-in_domain.sh
#   bash train_ml_models-in_domain.sh

set -e

cd ../../../prediction_classification_experiments-v2

echo "Starting ML training pipeline (In-Domain Baseline)"
echo "Current directory: $(pwd)"

# ============================================================
# PRE-GENERATE COMBINED DATASET
# ============================================================
echo ""
echo "======================================"
echo "Pre-generating combined dataset..."
echo "======================================"

python3 create_combined_dataset.py \
    --datasets synthetic financial_phrasebank chronicle2050 news_api yt timebank mf_climate \
    --output_name synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate \
    --no_version

echo "Dataset ready."

# ============================================================
# TRAIN
# ============================================================
echo ""
echo "Running Baseline (Standard) — Seeds 3, 7, 33"

for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""

    echo ">>> Running Baseline (Standard)"
    python ml-train.py \
        --dataset ../data/combined_datasets/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate/synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate.csv \
        --val_size 0.2 \
        --seed $seed
done

# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "All training complete. Aggregating results..."
echo "======================================"

EXPERIMENT="synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_$(date +%Y-%m-%d)"

mkdir -p ../data/classification_results/${EXPERIMENT}/averaged/in_dataset_comparisons/

python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --experiments seed3 seed7 seed33

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ Baseline experiments completed for seeds: 3, 7, 33"
echo "✓ Results aggregated and saved"
echo ""