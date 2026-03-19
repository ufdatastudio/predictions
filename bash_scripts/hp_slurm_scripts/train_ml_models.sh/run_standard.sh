#!/bin/bash
#SBATCH --job-name=ml_standard
#SBATCH --output=logs_%x_%j.out
#SBATCH --error=logs_%x_%j.err
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

echo "Running STANDARD (E1-E6)"
for seed in 3 7 33; do
    python ml-train.py --dataset ../data/combined_datasets/combined-full_synthetic-v1.csv --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv ../data/chronicle2050/chronicle2050-renamed_cols.csv

    python ml-train.py --dataset ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv ../data/chronicle2050/chronicle2050-renamed_cols.csv

    python ml-train.py --dataset ../data/chronicle2050/chronicle2050-renamed_cols.csv --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv

    python ml-train.py --dataset ../data/combined_datasets/combined-synthetic-fpb-v1.csv --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets ../data/chronicle2050/chronicle2050-renamed_cols.csv

    python ml-train.py --dataset ../data/combined_datasets/combined-synthetic-chronicle2050-v1.csv --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv

    python ml-train.py --dataset ../data/combined_datasets/combined-fpb-chronicle2050-v1.csv --no_test_split --val_size 0.2 --seed $seed \
        --test_datasets ../data/combined_datasets/combined-full_synthetic-v1.csv
done