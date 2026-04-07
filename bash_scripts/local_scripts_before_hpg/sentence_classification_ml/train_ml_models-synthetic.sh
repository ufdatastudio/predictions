#!/bin/bash
# train_ml_models.sh

cd ../../../prediction_classification_experiments-v2

# Run experiment 3 times with different seeds
for seed in 7 33 40; do
    echo "Running experiment with seed: $seed"train_ml_models-synthetic.sh
    uv run --active python3 main-ml-v2.py \
        --dataset ../data/combined_datasets/combined-full_synthetic-v1.csv \
        --seed $seed \
        --val_size 0.2
done

# Average all results after all seeds are complete
echo "All training complete. Averaging results..."
uv run --active python3 average_classification_results.py \
    --mode single \
    --experiment combined-full_synthetic-v1_$(date +%Y-%m-%d)