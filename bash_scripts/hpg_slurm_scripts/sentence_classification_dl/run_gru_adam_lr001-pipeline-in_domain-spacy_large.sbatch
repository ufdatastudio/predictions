#!/bin/bash
#SBATCH --job-name=gru_adam_lr001_spacy_large
#SBATCH --output=logs/gru_pipeline_spacy_large/logs_%x_%j.out
#SBATCH --error=logs/gru_pipeline_spacy_large/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

mkdir -p logs/gru_pipeline_spacy_large

cd ../../../prediction_classification_experiments-v2

source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

DATE=$(date +%Y-%m-%d)
EXPERIMENT="july_2026_results_${DATE}"
EMBEDDING_MODEL="spacy_large"

echo "============================================================"
echo "GRU Experiment 1: Adam, lr=0.001, 100 epochs"
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

    python gru-experiment.py \
        --experiment_name ${EXPERIMENT} \
        --train_path classification_results/${EXPERIMENT}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_train_set.csv \
        --val_path classification_results/${EXPERIMENT}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_val_set.csv \
        --test_path classification_results/${EXPERIMENT}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_test_set.csv \
        --embedding_model ${EMBEDDING_MODEL} \
        --n_epochs 100 \
        --learning_rate 0.001 \
        --optimizer adam \
        --hidden_size 128 \
        --run_name gru_adam_lr001 \
        --seed $seed
done

echo ""
echo "✓ ADAM lr=0.001 COMPLETE (seeds: 3, 7, 33)"

# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "Averaging GRU results across seeds..."
echo "======================================"
mkdir -p ../data/classification_results/${EXPERIMENT}/averaged/

python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --embedding_model ${EMBEDDING_MODEL} \
    --model_type gru \
    --experiments seed3 seed7 seed33

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ Adam optimizer (lr=0.001, 100 epochs) completed"
echo "✓ All train / val / test samples"
echo "✓ Results averaged across seeds: 3, 7, 33"
echo ""