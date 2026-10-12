#!/bin/bash
#SBATCH --job-name=conll_hedge_rule
#SBATCH --output=logs/conll_hedge_rule/logs_%x_%j.out
#SBATCH --error=logs/conll_hedge_rule/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -euo pipefail

# Submit this job from the directory containing this script / conll-2010-hedge-experiment.py
# (the relative cd below assumes that location).
# Create the log directory BEFORE calling sbatch (SLURM opens logs first):
#   mkdir -p logs/conll_hedge_rule
cd ../../../../prediction_classification_experiments-v2

source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

EXPERIMENT="tolsa_naacl_2026_2027-final"
SAVE_PATH="../data/classification_results/naacl_2026_submission"
BASE_RESULTS="${SAVE_PATH}/${EXPERIMENT}"
TEXT_COLUMN="Base Sentence"
LABEL_COLUMN="Ground Truth"

echo "============================================================"
echo "CoNLL-2010 HEDGE RULE BASELINE"
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

    echo
    echo "===== CoNLL HEDGE SEED=$SEED ====="
    python conll-2010-hedge-experiment.py \
        --val_dataset "$VAL_DATASET" \
        --test_dataset "$TEST_DATASET" \
        --experiment "$EXPERIMENT" \
        --text_column "$TEXT_COLUMN" \
        --label_column "$LABEL_COLUMN" \
        --seed "$SEED" \
        --save_path "$SAVE_PATH"
done

echo
echo "===== AVERAGING ACROSS SEEDS ====="
mkdir -p "${BASE_RESULTS}/averaged"

python average_classification_results.py \
    --mode single \
    --model_type conll \
    --results_dir "$SAVE_PATH" \
    --experiment "$EXPERIMENT" \
    --model_name "conll-2010-hedge" \
    --experiments seed3 seed7 seed33

echo "All CoNLL-2010 hedge runs completed and averaged."
echo "Results: ${BASE_RESULTS}/seed*/in_domain/conll_2010_hedge_rule/"
echo "Averaged: ${BASE_RESULTS}/averaged/"