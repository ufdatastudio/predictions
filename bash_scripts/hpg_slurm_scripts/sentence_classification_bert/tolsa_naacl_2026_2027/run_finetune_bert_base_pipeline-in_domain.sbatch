#!/bin/bash
#SBATCH --job-name=finetuned_bert_base
#SBATCH --output=logs_%x_%j.out
#SBATCH --error=logs_%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sm.tran@ufl.edu
#SBATCH --account=tkahveci
#SBATCH --qos=tkahveci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=a100:1
#SBATCH --time=06:00:00


set -e

# Purge existing modules, load conda, and activate the custom environment
module purge
module load conda
conda activate /blue/tkahveci/sm.tran/envs/bert_env

# Navigate to your script and dataset directory
cd /blue/tkahveci/sm.tran/datastudio/oct_11

# Dataset files
TRAIN_FILE="x_y_train_set.csv"
VAL_FILE="x_y_val_set.csv"
TEST_FILE="x_y_test_set.csv"

# Output configuration
SAVE_PATH="./results"
MODEL_NAME="bert-base-cased"
FINETUNED_MODEL_NAME="tolsa-bert_base"
TEXT_COLUMN="Base Sentence"
LABEL_COLUMN="Ground Truth"

echo "============================================================"
echo "FINE-TUNED BERT PIPELINE — ${MODEL_NAME}"
echo "============================================================"
echo "Working directory : $(pwd)"
echo "Environment       : /blue/tkahveci/sm.tran/envs/bert_env"
echo "Train dataset     : ${TRAIN_FILE}"
echo "Val dataset       : ${VAL_FILE}"
echo "Test dataset      : ${TEST_FILE}"
echo "Save directory    : ${SAVE_PATH}"
echo ""

# Iterate over random seeds
for seed in 3 7 33; do
    echo "============================================================"
    echo "                      RUNNING SEED: ${seed}"
    echo "============================================================"

    python finetune_son.py \
        --train_file "${TRAIN_FILE}" \
        --val_file "${VAL_FILE}" \
        --test_file "${TEST_FILE}" \
        --save_path "${SAVE_PATH}" \
        --text_column "${TEXT_COLUMN}" \
        --label_column "${LABEL_COLUMN}" \
        --model_name "${MODEL_NAME}" \
        --seed "${seed}" \
        --max_length 512 \
        --epochs 3 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --logging_steps 25 \
        --finetuned_model_name "${FINETUNED_MODEL_NAME}"
done

echo ""
echo "============================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "============================================================"
