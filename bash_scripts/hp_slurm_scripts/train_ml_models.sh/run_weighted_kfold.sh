#!/bin/bash
#SBATCH --job-name=gpt_as_classifier
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

echo "Running KFOLD (E1-E7)"
for seed in 3; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""
    echo ">>> Running Train on Synthetic → Test on FPB"
    python llm-classifiers.py \
        --model_name llama-3.1-8b-instant \
        --label_column 'Sentence Label'
done