#!/bin/bash
#SBATCH --job-name=lt_prop_llama
#SBATCH --output=llama_prop.out
#SBATCH --error=llama_prop.err
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-b200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

module load python/3.10
module load cuda/12.4

cd ../script_experiments

python3 extract_prediction_properties.py \
    --dataset combined_datasets/combined-full_synthetic-v1.csv \
    --models "llama-3.1-8b-instant"

python3 extract_prediction_properties.py \
    --dataset financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv \
    --models "llama-3.1-8b-instant"

python3 extract_prediction_properties.py \
    --dataset chronicle2050/data.csv \
    --models "llama-3.1-8b-instant"