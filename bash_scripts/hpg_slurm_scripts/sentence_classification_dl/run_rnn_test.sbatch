#!/bin/bash
#SBATCH --job-name=rnn_test_5k
#SBATCH --output=logs/rnn_pipeline/logs_test_%j.out
#SBATCH --error=logs/rnn_pipeline/logs_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

# Create log directory
mkdir -p logs/rnn_pipeline

cd ../../../prediction_classification_experiments-v2

# Activate virtual environment
source /home/dj.brinkley/orange_dj.brinkley/predictions/.venv/bin/activate

EXPERIMENT="eacl_2026_results_2026-06-26"
BASE_RESULTS="../data/classification_results/${EXPERIMENT}"
EMBEDDING_MODEL="spacy_large"

echo "============================================================"
echo "RNN TEST RUN (5000 samples)"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Embedding Model: ${EMBEDDING_MODEL}"
echo ""

python rnn-experiment.py \
    --experiment_name ${EXPERIMENT} \
    --train_path classification_results/${EXPERIMENT}/seed3/in_domain/${EMBEDDING_MODEL}/x_y_train_set.csv \
    --test_path classification_results/${EXPERIMENT}/seed3/in_domain/${EMBEDDING_MODEL}/x_y_test_set.csv \
    --embedding_model ${EMBEDDING_MODEL} \
    --train_sample 5000 \
    --test_sample 5000 \
    --n_epochs 20 \
    --learning_rate 0.001 \
    --optimizer adam \
    --hidden_size 128 \
    --seed 3

echo ""
echo "======================================"
echo "TEST COMPLETE"
echo "======================================"