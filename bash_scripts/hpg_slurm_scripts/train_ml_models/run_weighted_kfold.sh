#!/bin/bash
#SBATCH --job-name=ml_w_kfold
#SBATCH --output=../logs/train_ml_models/logs_%x_%j.out
#SBATCH --error=../logs/train_ml_models/logs_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e
cd ../../../prediction_classification_experiments-v2
module load python/3
source ../.venv/bin/activate

echo "Running WEIGHTED KFOLD (E1-E6)"
for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""
    
    for k in 3 7; do
        echo "   --- Weighted K-Fold = $k ---"
        
        echo ">>> Running E1 (Weighted $k-Fold): Train on Synthetic → Test on FPB + C2050"
        python ml-train.py --dataset ../data/combined_datasets/combined-full_synthetic-v1.csv --no_test_split --seed $seed --stratified_kfold $k \
            --reweight_class 'balanced' --experiment_suffix="-weighted-kfold$k" \
            --test_datasets ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv ../data/chronicle2050/chronicle2050-renamed_cols.csv
        
        echo ">>> Running E2 (Weighted $k-Fold): Train on FPB → Test on Synthetic + C2050"
        python ml-train.py --dataset ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv --no_test_split --seed $seed --stratified_kfold $k \
            --reweight_class 'balanced' --experiment_suffix="-weighted-kfold$k" \
            --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv ../data/chronicle2050/chronicle2050-renamed_cols.csv
        
        echo ">>> Running E3 (Weighted $k-Fold): Train on C2050 → Test on Synthetic + FPB"
        python ml-train.py --dataset ../data/chronicle2050/chronicle2050-renamed_cols.csv --no_test_split --seed $seed --stratified_kfold $k \
            --reweight_class 'balanced' --experiment_suffix="-weighted-kfold$k" \
            --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv
        
        echo ">>> Running E4 (Weighted $k-Fold): Train on Synthetic+FPB → Test on C2050"
        python ml-train.py --dataset ../data/combined_datasets/combined-synthetic-fpb-v1.csv --no_test_split --seed $seed --stratified_kfold $k \
            --reweight_class 'balanced' --experiment_suffix="-weighted-kfold$k" \
            --test_datasets ../data/chronicle2050/chronicle2050-renamed_cols.csv
        
        echo ">>> Running E5 (Weighted $k-Fold): Train on Synthetic+C2050 → Test on FPB"
        python ml-train.py --dataset ../data/combined_datasets/combined-synthetic-chronicle2050-v1.csv --no_test_split --seed $seed --stratified_kfold $k \
            --reweight_class 'balanced' --experiment_suffix="-weighted-kfold$k" \
            --test_datasets ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv
        
        echo ">>> Running E6 (Weighted $k-Fold): Train on FPB+C2050 → Test on Synthetic"
        python ml-train.py --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv --no_test_split --seed $seed --stratified_kfold $k \
            --reweight_class 'balanced' --experiment_suffix="-weighted-kfold$k" \
            --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv
    done
done