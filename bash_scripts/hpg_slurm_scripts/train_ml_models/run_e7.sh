#!/bin/bash
#SBATCH --job-name=ml_e7
#SBATCH --output=logs_%x_%j.out
#SBATCH --error=logs_%x_%j.err
#SBATCH --time=24:00:00
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

echo "Running E7 (All Variations)"
for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""
    
    echo ">>> Running E7 (Standard)"
    python ml-train.py --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv --val_size 0.2 --seed $seed
    
    echo ">>> Running E7 (Weighted)"
    python ml-train.py --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv --val_size 0.2 --seed $seed \
        --reweight_class 'balanced' --experiment_suffix="-weighted"

    for k in 3 7; do
        echo ">>> Running E7 ($k-Fold)"
        python ml-train.py --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv --seed $seed --stratified_kfold $k
        
        echo ">>> Running E7 (Weighted $k-Fold)"
        python ml-train.py --dataset ../data/combined_datasets/combined-synthetic_fpb_chr2050-v1.csv --seed $seed --stratified_kfold $k \
            --reweight_class 'balanced' --experiment_suffix="-weighted"
    done
done