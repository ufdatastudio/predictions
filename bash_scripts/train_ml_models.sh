#!/bin/bash
# train_ml_models.sh

cd ../prediction_classification_experiments-v2

# Run each experiment 3 times with different seeds
# Update to run with uv run...
for seed in 7 33 40; do
    echo "Running experiments with seed: $seed"
    python3 main-ml-v2.py --dataset ../data/financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv --seed $seed
    python3 main-ml-v2.py --dataset ../data/financial_phrase_bank/resampling_maya/fpb-maya-binary-oversampled-96d-v1.csv --seed $seed
    python3 main-ml-v2.py --dataset ../data/financial_phrase_bank/resampling_maya/fpb-maya-binary-undersampled-96d-v1.csv --seed $seed
done

# Average all results after all seeds are complete
echo "All training complete. Averaging results..."
python3 average_classification_results.py
