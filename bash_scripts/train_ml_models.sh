#!/bin/bash
# train_ml_models.sh - Run E1-E7 experiments with multiple seeds
#
# Usage: bash train_ml_models.sh
# Run from: predictions/scripts/ directory
set -e  # Exit on error

# Navigate to experiment directory
cd ../prediction_classification_experiments-v2
echo "Starting ML training pipeline for E1-E7"
echo "Current directory: $(pwd)"
echo ""

echo "======================================"
echo "Pre-generating combined datasets..."
echo "======================================"

# E1, E1.2, E2, E3, E6 external dataset: Full Synthetic
python create_combined_dataset.py \
    --datasets predictions non_predictions \
    --output_name combined-full_synthetic-v1 \
    --save_path ../data/combined_datasets/ \
    --no_version

# E4 dataset: Synthetic + FPB
python create_combined_dataset.py \
    --datasets predictions non_predictions financial_phrasebank \
    --output_name combined-synthetic-fpb-v1 \
    --save_path ../data/combined_datasets/ \
    --no_version

# E5 dataset: Synthetic + Chronicle2050
python create_combined_dataset.py \
    --datasets predictions non_predictions chronicle2050 \
    --output_name combined-synthetic-chronicle2050-v1 \
    --save_path ../data/combined_datasets/ \
    --no_version

# E6 dataset: FPB + Chronicle2050
python create_combined_dataset.py \
    --datasets financial_phrasebank chronicle2050 \
    --output_name combined-fpb-chronicle2050-v1 \
    --save_path ../data/combined_datasets/ \
    --no_version

# E7 dataset: Synthetic + FPB + Chronicle2050
python create_combined_dataset.py \
    --datasets predictions non_predictions financial_phrasebank chronicle2050 \
    --output_name combined-synthetic_fpb_chr2050-v1 \
    --save_path ../data/combined_datasets/ \
    --no_version

echo "Datasets ready."
echo ""

# Run each experiment 3 times with different seeds
for seed in 3 7 33; do
    echo ""
    echo "======================================"
    echo "Running experiments with seed: $seed"
    echo "======================================"
    echo ""

    # ============================================================
    # E1: Train on Synthetic → Test on FPB + Chronicle2050
    # ============================================================
    echo "Running E1 (Synthetic with val_size=0.2)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-full_synthetic-v1.csv \
        --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets \
            ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv \
            ../data/chronicle2050/chronicle2050-renamed_cols.csv

    # ============================================================
    # E1.2: Train on Synthetic (no val) → Test on FPB + Chronicle2050
    # ============================================================
    echo "Running E1.2 (Synthetic without val_size)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-full_synthetic-v1.csv \
        --no_test_split --seed $seed \
        --test_datasets \
            ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv \
            ../data/chronicle2050/chronicle2050-renamed_cols.csv

    # ============================================================
    # E2: Train on FPB → Test on Synthetic + Chronicle2050
    # ============================================================
    echo "Running E2..."
    python ml-train.py \
        --dataset ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv \
        --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets \
            ../data/combined_datasets/combined-full_synthetic-v1.csv \
            ../data/chronicle2050/chronicle2050-renamed_cols.csv

    # ============================================================
    # E3: Train on Chronicle2050 → Test on Synthetic + FPB
    # ============================================================
    echo "Running E3..."
    python ml-train.py \
        --dataset ../data/chronicle2050/chronicle2050-renamed_cols.csv \
        --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets \
            ../data/combined_datasets/combined-full_synthetic-v1.csv \
            ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv

    # ============================================================
    # E4: Train on Synthetic+FPB → Test on Chronicle2050
    # ============================================================
    echo "Running E4..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic-fpb-v1.csv \
        --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets ../data/chronicle2050/chronicle2050-renamed_cols.csv

    # ============================================================
    # E5: Train on Synthetic+Chronicle2050 → Test on FPB
    # ============================================================
    echo "Running E5..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic-chronicle2050-v1.csv \
        --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv

    # ============================================================
    # E6: Train on FPB+Chronicle2050 → Test on Synthetic
    # ============================================================
    echo "Running E6..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv \
        --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv

    # ============================================================
    # E7: Standard train/val/test split on all combined
    # ============================================================
    echo "Running E7 (standard split on full combined dataset)..."
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv \
        --val_size 0.2 --seed $seed
        
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

# Aggregate both in-domain (E7) and external (E1-E6) results
python average_classification_results-v2.py --mode both

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ All experiments (E1-E7) completed for seeds: 3, 7, 33"
echo "✓ Results aggregated and saved"
echo ""