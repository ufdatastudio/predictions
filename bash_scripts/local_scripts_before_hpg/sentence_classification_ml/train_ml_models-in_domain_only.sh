#!/bin/bash
# train_ml_models-in_domain_only.sh - Run E7 experiments with multiple seeds
#
# Usage: 
    # chmod +x train_ml_models-in_domain_only.sh to make an executable
    # bash train_ml_models-in_domain_only.sh

set -e  # Exit on error

# Navigate to experiment directory
cd ../../../prediction_classification_experiments-v2

echo "Starting ML training pipeline for E7 (In-Domain)"
echo "Current directory: $(pwd)"
echo ""

echo "======================================"
echo "Pre-generating combined datasets..."
echo "======================================"

# E1, E2, E3, E6 external dataset: Full Synthetic
python create_combined_dataset.py \
    --datasets predictions non_predictions financial_phrasebank chronicle2050 news_api yt \
    --output_name all-preds-non_preds-fpb-c2050-news-yt \
    --save_path ../data/combined_datasets/ \
    --no_version

echo "Datasets ready."
echo ""

for seed in 3 7; do
    echo ""
    echo "======================================"
    echo "Running experiments with seed: $seed"
    echo "======================================"
    echo ""
    
    # ============================================================
    # E6: Train on FPB+Chronicle2050 → Test on Synthetic
    # ============================================================
    echo "Running E6 (Standard)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/all-preds-non_preds-fpb-c2050-news-yt/all-preds-non_preds-fpb-c2050-news-yt.csv \
        --val_size 0.2 \
        --seed $seed
    echo ""
    echo "✓ Completed all experiments for seed: $seed"
    echo ""
done


# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "All training complete. Aggregating results..."
echo "======================================"

mkdir -p ../data/classification_results/all-preds-non_preds-fpb-c2050-news-yt_$(date +%Y-%m-%d)/averaged/in_dataset_comparisons/

python average_classification_results.py \
    --mode single \
    --experiment all-preds-non_preds-fpb-c2050-news-yt_$(date +%Y-%m-%d) \
    --experiments seed3 seed7
    | tee ../data/classification_results/all-preds-non_preds-fpb-c2050-news-yt_$(date +%Y-%m-%d)/averaged/in_dataset_comparisons/final_results_summary.txt


echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ All experiments (E6-E7) completed for seed: 3"
echo "✓ Results aggregated and saved"
echo ""