#!/bin/bash
# train_ml_models-in_domain-each_variation.sh - All variations with three seeds
#
# Usage:
#   chmod +x train_ml_models-in_domain-each_variation.sh
#   bash train_ml_models-in_domain-each_variation.sh

set -e

cd ../../../prediction_classification_experiments-v2

echo "Starting ML training pipeline (In-Domain All Variations)"
echo "Current directory: $(pwd)"

# ============================================================
# PRE-GENERATE COMBINED DATASET
# ============================================================
echo ""
echo "======================================"
echo "Pre-generating combined dataset..."
echo "======================================"

python3 create_combined_dataset.py \
    --datasets synthetic financial_phrasebank chronicle2050 news_api yt timebank \
    --output_name synthetic-fpb-c2050-yt-news-timebank

echo "Dataset ready."

# ============================================================
# TRAIN — ALL VARIATIONS
# ============================================================
echo ""
echo "Running All Variations — Seeds 3, 7, 33"

for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""

    echo ">>> Running Standard"
    python ml-train.py \
        --dataset ../data/combined_datasets/synthetic-fpb-c2050-yt-news-timebank/synthetic-fpb-c2050-yt-news-timebank.csv \
        --val_size 0.2 \
        --seed $seed

    echo ">>> Running Weighted"
    python ml-train.py \
        --dataset ../data/combined_datasets/synthetic-fpb-c2050-yt-news-timebank/synthetic-fpb-c2050-yt-news-timebank.csv \
        --val_size 0.2 \
        --seed $seed \
        --reweight_class 'balanced' \
        --experiment_suffix="-weighted"

    echo ">>> Running Oversampled"
    python ml-train.py \
        --dataset ../data/combined_datasets/synthetic-fpb-c2050-yt-news-timebank/synthetic-fpb-c2050-yt-news-timebank.csv \
        --val_size 0.2 \
        --seed $seed \
        --resample_method oversample \
        --experiment_suffix="-oversampled"

    echo ">>> Running Undersampled"
    python ml-train.py \
        --dataset ../data/combined_datasets/synthetic-fpb-c2050-yt-news-timebank/synthetic-fpb-c2050-yt-news-timebank.csv \
        --val_size 0.2 \
        --seed $seed \
        --resample_method undersample \
        --experiment_suffix="-undersampled"
done

# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "All training complete. Aggregating results..."
echo "======================================"

EXPERIMENT="synthetic-fpb-c2050-yt-news-timebank_$(date +%Y-%m-%d)"

mkdir -p ../data/classification_results/${EXPERIMENT}/averaged/in_dataset_comparisons/

python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --experiments seed3 seed7 seed33

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ All variation experiments completed for seeds: 3, 7, 33"
echo "✓ Results aggregated and saved"
echo ""