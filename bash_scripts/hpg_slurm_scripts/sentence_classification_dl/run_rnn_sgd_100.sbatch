#!/bin/bash
#SBATCH --job-name=rnn_sgd100
#SBATCH --output=logs/rnn_pipeline/logs_sgd100_%j.out
#SBATCH --error=logs/rnn_pipeline/logs_sgd100_%j.err
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

mkdir -p logs/rnn_pipeline
cd ../../../prediction_classification_experiments-v2
source /home/dj.brinkley/orange_dj.brinkley/predictions/.venv/bin/activate

EXPERIMENT="eacl_2026_results_2026-06-26"
EMBEDDING_MODEL="spacy_large"

echo "============================================================"
echo "RNN Experiment 2: SGD, lr=0.001, 100 epochs"
echo "============================================================"

for seed in 3 7 33; do
    echo ""
    echo "==================== SEED: $seed ===================="
    echo ""
    
    python rnn-experiment.py \
        --experiment_name ${EXPERIMENT} \
        --train_path classification_results/${EXPERIMENT}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_train_set.csv \
        --test_path classification_results/${EXPERIMENT}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_test_set.csv \
        --embedding_model ${EMBEDDING_MODEL} \
        --n_epochs 100 \
        --learning_rate 0.001 \
        --optimizer sgd \
        --hidden_size 128 \
        --run_name rnn_sgd_100 \
        --seed $seed
done

echo ""
echo "✓ SGD 100 EPOCHS COMPLETE (seeds: 3, 7, 33)"

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
echo "✓ SGD optimizer (lr=0.001, 100 epochs) completed"
echo "✓ Results averaged across seeds: 3, 7, 33"