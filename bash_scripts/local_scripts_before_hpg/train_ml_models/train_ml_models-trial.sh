#!/bin/bash
# train_ml_models.sh - Run E6-E7 experiments with 1 seed for local testing
#
# Usage: bash train_ml_models.sh
# Run from: predictions/scripts/ directory

set -e  # Exit on error

# Navigate to experiment directory
cd ../../../prediction_classification_experiments-v2

echo "Starting ML training pipeline for Local Test (E6-E7)"
echo "Current directory: $(pwd)"
echo ""

echo "======================================"
echo "Pre-generating combined datasets..."
echo "======================================"

python create_combined_dataset.py \
    --datasets financial_phrasebank chronicle2050 \
    --output_name combined-fpb-chronicle2050-v1 \
    --save_path ../data/combined_datasets/ \
    --no_version

python create_combined_dataset.py \
    --datasets predictions non_predictions financial_phrasebank chronicle2050 \
    --output_name combined-synthetic_fpb_chr2050-v1 \
    --save_path ../data/combined_datasets/ \
    --no_version

python create_combined_dataset.py \
    --datasets predictions non_predictions \
    --output_name combined-full_synthetic-v1 \
    --save_path ../data/combined_datasets/ \
    --no_version

echo "Datasets ready."
echo ""

for seed in 3; do
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
        --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv \
        --no_test_split \
        --val_size 0.2 \
        --seed $seed \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv

    echo "Running E6 (Weighted)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv \
        --no_test_split \
        --val_size 0.2 \
        --seed $seed \
        --reweight_class 'balanced' \
        --experiment_suffix="-weighted" \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv

    echo "Running E6 (Oversampled)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv \
        --no_test_split \
        --val_size 0.2 \
        --seed $seed \
        --resample_method oversample \
        --experiment_suffix="-oversampled" \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv

    echo "Running E6 (3-Fold)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv \
        --no_test_split \
        --seed $seed \
        --stratified_kfold 3 \
        --experiment_suffix="-kfold3" \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv

    echo "Running E6 (7-Fold)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv \
        --no_test_split \
        --seed $seed \
        --stratified_kfold 7 \
        --experiment_suffix="-kfold7" \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv

    echo "Running E6 (Weighted 3-Fold)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv \
        --no_test_split \
        --seed $seed \
        --reweight_class 'balanced' \
        --experiment_suffix="-weighted-kfold3" \
        --stratified_kfold 3 \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv

    echo "Running E6 (Weighted 7-Fold)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv \
        --no_test_split \
        --seed $seed \
        --reweight_class 'balanced' \
        --experiment_suffix="-weighted-kfold7" \
        --stratified_kfold 7 \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv

    # ============================================================
    # E7: Full Combined Dataset (Internal Evaluation Only)
    # ============================================================
    echo "Running E7 (Standard)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv \
        --val_size 0.2 \
        --seed $seed

    echo "Running E7 (Weighted)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv \
        --val_size 0.2 \
        --seed $seed \
        --reweight_class 'balanced' \
        --experiment_suffix="-weighted"

    echo "Running E7 (3-Fold)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv \
        --seed $seed \
        --stratified_kfold 3 \
        --experiment_suffix="-kfold3"

    echo "Running E7 (7-Fold)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv \
        --seed $seed \
        --stratified_kfold 7 \
        --experiment_suffix="-kfold7"

    echo "Running E7 (Weighted 3-Fold)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv \
        --seed $seed \
        --reweight_class 'balanced' \
        --experiment_suffix="-weighted-kfold3" \
        --stratified_kfold 3

    echo "Running E7 (Weighted 7-Fold)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv \
        --seed $seed \
        --reweight_class 'balanced' \
        --experiment_suffix="-weighted-kfold7" \
        --stratified_kfold 7
    
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

mkdir -p ../data/classification_results/cross_dataset_comparisons/
python average_classification_results.py --mode cross_dataset | tee ../data/classification_results/cross_dataset_comparisons/final_results_summary.txt

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ All experiments (E6-E7) completed for seed: 3"
echo "✓ Results aggregated and saved"
echo ""