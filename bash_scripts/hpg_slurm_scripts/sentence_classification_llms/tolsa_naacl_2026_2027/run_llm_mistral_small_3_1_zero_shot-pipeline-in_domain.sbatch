#!/bin/bash
#SBATCH --job-name=llm_mistral_small_3_1_zero
#SBATCH --output=logs/llm_pipeline_spacy_large/logs_%x_%j.out
#SBATCH --error=logs/llm_pipeline_spacy_large/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -euo pipefail

# Submit this job from sentence_classification_llms.
# Create the log directory BEFORE calling sbatch (SLURM opens logs first).
cd ../../../../prediction_classification_experiments-v2

source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

EXPERIMENT="tolsa_naacl_2026_2027-final"
BASE_RESULTS="../data/classification_results/naacl_2026_submission/${EXPERIMENT}"
MODEL_NAME="mistral-small-3.1"
PROMPT_TYPE="zero-shot"
LABEL_COLUMN="Ground Truth"

echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Prompt: $PROMPT_TYPE"
echo "Experiment: $EXPERIMENT"
echo "============================================================"

for SEED in 3 7 33; do
    for SPLIT in val test; do
        DATASET="${BASE_RESULTS}/seed${SEED}/in_domain/splits/x_y_${SPLIT}_set.csv"

        if [[ ! -f "$DATASET" ]]; then
            echo "ERROR: Dataset not found: $DATASET" >&2
            exit 1
        fi

        echo
        echo "===== MODEL=$MODEL_NAME SEED=$SEED SPLIT=$SPLIT ====="
        python llm-experiment.py \
            --model_name "$MODEL_NAME" \
            --test_dataset "$DATASET" \
            --label_column "$LABEL_COLUMN" \
            --seed "$SEED" \
            --prompt_type "$PROMPT_TYPE"
    done
done

echo "All zero-shot runs completed for $MODEL_NAME."
