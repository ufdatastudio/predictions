#!/bin/bash
# train_ml_models-trial.sh - Comprehensive local test of all pipeline logic paths
# Usage: bash train_ml_models-trial.sh

set -e  # Exit on error

# Navigate to experiment directory and activate environment
cd ../../../prediction_classification_experiments-v2
source ../.venv_predictions/bin/activate

echo "Starting ML Trial Pipeline (Testing All Logic Cases)"
echo "Current directory: $(pwd)"
echo ""

echo "======================================"
echo "Pre-generating combined datasets..."
echo "======================================"

python create_combined_dataset.py --datasets predictions non_predictions --output_name combined-full_synthetic-v1 --save_path ../data/combined_datasets/ --no_version
python create_combined_dataset.py --datasets predictions non_predictions financial_phrasebank --output_name combined-synthetic-fpb-v1 --save_path ../data/combined_datasets/ --no_version
python create_combined_dataset.py --datasets predictions non_predictions chronicle2050 --output_name combined-synthetic-chronicle2050-v1 --save_path ../data/combined_datasets/ --no_version
python create_combined_dataset.py --datasets financial_phrasebank chronicle2050 --output_name combined-fpb-chronicle2050-v1 --save_path ../data/combined_datasets/ --no_version
python create_combined_dataset.py --datasets predictions non_predictions financial_phrasebank chronicle2050 --output_name combined-synthetic_fpb_chr2050-v1 --save_path ../data/combined_datasets/ --no_version

echo "Datasets ready."
echo ""

for seed in 3; do
    echo "============================================================"
    echo "                 TESTING ALL CASES (SEED $seed)"
    echo "============================================================"
    
    # 1. Test Standard Cross-Domain
    echo ">>> Case 1: Standard (E1)"
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-full_synthetic-v1.csv \
        --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv ../data/chronicle2050/chronicle2050-renamed_cols.csv

    # 2. Test Weighted Cross-Domain
    echo ">>> Case 2: Weighted (E2)"
    python ml-train.py \
        --dataset ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv \
        --no_test_split --val_size 0.2 --seed $seed \
        --reweight_class 'balanced' --experiment_suffix="-weighted" \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv ../data/chronicle2050/chronicle2050-renamed_cols.csv

    # 3. Test Oversampled Cross-Domain
    echo ">>> Case 3: Oversampled (E3)"
    python ml-train.py \
        --dataset ../data/chronicle2050/chronicle2050-renamed_cols.csv \
        --no_test_split --val_size 0.2 --seed $seed \
        --resample_method oversample --experiment_suffix="-oversampled" \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv

    # 4. Test Undersampled Cross-Domain
    echo ">>> Case 4: Undersampled (E4)"
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic-fpb-v1.csv \
        --no_test_split --val_size 0.2 --seed $seed \
        --resample_method undersample --experiment_suffix="-undersampled" \
        --test_datasets ../data/chronicle2050/chronicle2050-renamed_cols.csv

    # 5. Test Standard K-Fold
    echo ">>> Case 5: Standard K-Fold (E5, k=3)"
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic-chronicle2050-v1.csv \
        --no_test_split --seed $seed --stratified_kfold 3 \
        --experiment_suffix="-kfold3" \
        --test_datasets ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv

    # 6. Test Weighted K-Fold
    echo ">>> Case 6: Weighted K-Fold (E6, k=3)"
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv \
        --no_test_split --seed $seed --stratified_kfold 3 \
        --reweight_class 'balanced' --experiment_suffix="-weighted-kfold3" \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv

    # 7. Test Oversampled K-Fold (Most complex logic check)
    echo ">>> Case 7: Oversampled K-Fold (E6, k=3)"
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv \
        --no_test_split --seed $seed --stratified_kfold 3 \
        --resample_method oversample --experiment_suffix="-oversampled-kfold3" \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv

    # 8. Test E7 Standard (In-Domain Split)
    echo ">>> Case 8: E7 Standard (Internal Split)"
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv \
        --val_size 0.2 --seed $seed

    # 9. Test E7 Undersampled K-Fold (In-Domain K-Fold)
    echo ">>> Case 9: E7 Undersampled K-Fold (Internal Split)"
    python ml-train.py \
        --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv \
        --seed $seed --stratified_kfold 3 \
        --resample_method undersample --experiment_suffix="-undersampled-kfold3"

    echo "✓ Completed all trial cases!"
done

# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "Testing Aggregation Script..."
echo "======================================"

mkdir -p ../data/classification_results/cross_dataset_comparisons/
python average_classification_results.py --mode cross_dataset | tee ../data/classification_results/cross_dataset_comparisons/trial_results_summary.txt

echo "======================================"
echo "ALL TRIALS PASSED!"
echo "======================================"