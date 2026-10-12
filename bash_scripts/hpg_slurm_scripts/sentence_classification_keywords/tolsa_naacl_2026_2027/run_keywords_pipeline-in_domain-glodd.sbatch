#!/bin/bash
#SBATCH --job-name=glodd_step0_rule
#SBATCH --output=logs/glodd_step0_rule/logs_%x_%j.out
#SBATCH --error=logs/glodd_step0_rule/logs_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -euo pipefail

# Submit this job from the directory containing this script / glodd-hristova-step0-experiment.py
# (the relative cd below assumes that location).
# Create the log directory BEFORE calling sbatch (SLURM opens logs first):
#   mkdir -p logs/glodd_step0_rule
cd ../../../../prediction_classification_experiments-v2

source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

EXPERIMENT="tolsa_naacl_2026_2027-final"
SAVE_PATH="../data/classification_results/naacl_2026_submission"
BASE_RESULTS="${SAVE_PATH}/${EXPERIMENT}"
TEXT_COLUMN="Base Sentence"
LABEL_COLUMN="Ground Truth"

echo "============================================================"
echo "GLODD & HRISTOVA STEP 0 RULE BASELINE"
echo "Experiment: $EXPERIMENT"
echo "Base results: $BASE_RESULTS"
echo "============================================================"

for SEED in 3 7 33; do
    VAL_DATASET="${BASE_RESULTS}/seed${SEED}/in_domain/splits/x_y_val_set.csv"
    TEST_DATASET="${BASE_RESULTS}/seed${SEED}/in_domain/splits/x_y_test_set.csv"

    for f in "$VAL_DATASET" "$TEST_DATASET"; do
        if [[ ! -f "$f" ]]; then
            echo "ERROR: Dataset not found: $f" >&2
            exit 1
        fi
    done

    # One spaCy model per invocation; each run applies it to val and test separately.
    for EMBEDDING_MODEL in spacy_small spacy_medium spacy_large; do
        echo
        echo "===== GLODD STEP 0 SEED=$SEED SPACY=$EMBEDDING_MODEL ====="
        python glodd-hristova-step0-experiment.py \
            --val_dataset "$VAL_DATASET" \
            --test_dataset "$TEST_DATASET" \
            --experiment "$EXPERIMENT" \
            --text_column "$TEXT_COLUMN" \
            --label_column "$LABEL_COLUMN" \
            --embedding_model "$EMBEDDING_MODEL" \
            --seed "$SEED" \
            --save_path "$SAVE_PATH"
    done
done

echo
echo "===== AVERAGING ACROSS SEEDS ====="
# No --embedding_model: averages every spaCy folder found under glodd_hristova_step0/,
# each one separately across the 3 seeds.
python average_classification_results.py \
    --mode single \
    --model_type glodd \
    --results_dir "$SAVE_PATH" \
    --experiment "$EXPERIMENT" \
    --experiments seed3 seed7 seed33

echo
echo "All Glodd & Hristova Step 0 runs completed and averaged."
echo "Averaged: ${BASE_RESULTS}/averaged/in_domain/glodd_hristova_step0/<spacy_model>/glodd/"
echo "Results: ${BASE_RESULTS}/seed*/in_domain/glodd_hristova_step0/<spacy_model>/"