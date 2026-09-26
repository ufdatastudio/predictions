#!/bin/bash
#SBATCH --job-name=finetuned_bert_large_tolsam
#SBATCH --output=logs/finetuned_bert_pipeline/logs_%x_%j.out
#SBATCH --error=logs/finetuned_bert_pipeline/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

mkdir -p logs/finetuned_bert_pipeline

cd ../../../prediction_classification_experiments-v2

source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

DATE=$(date +%Y-%m-%d)
EXPERIMENT="july_2026_results_${DATE}"

# INPUT: July 2026 dataset
DATASET="../data/combined_datasets/july_2026_results/july_2026_results.csv"

# OUTPUT: save BERT experiments under NAACL results
SAVE_PATH="../data/classification_results/naacl_2026_submission"

MODEL_NAME="bert-large-cased"
FINETUNED_MODEL_NAME="tolsa-bert"
TEXT_COLUMN="Base Sentence"
LABEL_COLUMN="Ground Truth"

echo "============================================================"
echo "FINE-TUNED BERT PIPELINE — ${MODEL_NAME}"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Dataset: ${DATASET}"
echo "Save path: ${SAVE_PATH}"
echo "Current directory: $(pwd)"
echo ""

# ============================================================
# TRAIN, VALIDATE, TEST, AND EVALUATE
# ============================================================

for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: ${seed}"
    echo "============================================================"
    echo ""

    python finetune-bert-experiment.py \
        --dataset "${DATASET}" \
        --save_path "${SAVE_PATH}" \
        --text_column "${TEXT_COLUMN}" \
        --label_column "${LABEL_COLUMN}" \
        --model_name "${MODEL_NAME}" \
        --seed "${seed}" \
        --val_size 0.2 \
        --test_size 0.2 \
        --max_length 512 \
        --epochs 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --logging_steps 25 \
        --finetuned_model_name "${FINETUNED_MODEL_NAME}" \
        --sample_size 1000
done

# ============================================================
# AVERAGE BERT RESULTS ACROSS SEEDS
# ============================================================

echo ""
echo "======================================"
echo "All training complete. Aggregating results..."
echo "======================================"

mkdir -p "${SAVE_PATH}/${EXPERIMENT}/averaged"

python average_classification_results.py \
    --mode single \
    --model_type bert \
    --results_dir "${SAVE_PATH}" \
    --experiment "${EXPERIMENT}" \
    --model_name "${MODEL_NAME}" \
    --experiments seed3 seed7 seed33

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ Fine-tuned BERT completed for seeds: 3, 7, 33"
echo "✓ BERT results saved under: ${SAVE_PATH}/${EXPERIMENT}/"
echo "✓ Averaged results saved under: ${SAVE_PATH}/${EXPERIMENT}/averaged/"
echo ""