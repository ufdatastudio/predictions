#!/bin/bash
# run_rnn_pipeline-in_domain-spacy_medium.sh - RNN smoke test with 50 samples and 3 epochs across three seeds
#
# Usage:
#   chmod +x run_rnn_pipeline-in_domain-spacy_medium.sh
#   bash run_rnn_pipeline-in_domain-spacy_medium.sh

set -e

mkdir -p logs/rnn_pipeline

cd ../../../prediction_classification_experiments-v2

# source ../../../.venv_predictions/bin/activate

# EXPERIMENT="eacl_2026_results_2026-06-12"
EXPERIMENT="july_2026_results_2026-07-08"
EMBEDDING_MODEL="spacy_medium"

echo "============================================================"
echo "RNN TEST RUN (50 samples, 3 epochs)"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Embedding Model: ${EMBEDDING_MODEL}"
echo ""

# ============================================================
# TRAIN ALL SEEDS
# ============================================================
for seed in 3 7 33; do
    echo ""
    echo "==================== SEED: $seed ===================="
    echo ""

    python rnn-experiment.py \
        --experiment_name ${EXPERIMENT} \
        --train_path classification_results/${EXPERIMENT}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_train_set.csv \
        --val_path classification_results/${EXPERIMENT}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_val_set.csv \
        --test_path classification_results/${EXPERIMENT}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_test_set.csv \
        --embedding_model ${EMBEDDING_MODEL} \
        --sample 50 \
        --n_epochs 3 \
        --learning_rate 0.001 \
        --optimizer adam \
        --hidden_size 128 \
        --seed $seed \
        --run_name rnn_spacy_medium
done

echo ""
echo "✓ RNN TEST COMPLETE (seeds: 3, 7, 33)"

# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "Averaging RNN results across seeds..."
echo "======================================"

python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --embedding_model ${EMBEDDING_MODEL} \
    --model_type rnn \
    --experiments seed3 seed7 seed33

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ RNN test run completed"
echo "✓ 50 train / 50 test samples"
echo "✓ 3 epochs"
echo "✓ Adam optimizer (lr=0.001) with Learning Rate Scheduler (If the loss doesn't improve for 3 epochs, cut the learning rate in half.)"
echo "✓ Results averaged across seeds: 3, 7, 33"
echo ""