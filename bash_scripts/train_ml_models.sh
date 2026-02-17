#!/bin/bash
# run_experiments.sh

cd ../prediction_classification_experiments-v2

# Run each experiment 3 times with different seeds
for seed in 42 123 999; do
    python3 main-ml-v2.py --dataset_type fin_phrasebank --seed $seed
    python3 main-ml-v2.py --dataset ../data/financial_phrase_bank/resampling_maya/oversampled_96d-v4.csv --seed $seed
    python3 main-ml-v2.py --dataset ../data/financial_phrase_bank/resampling_maya/undersampled_96d-v4.csv --seed $seed
done