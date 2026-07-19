#!/bin/bash
#SBATCH --job-name=llm_llama_31_8b_cot_spacy_large
#SBATCH --output=logs/llm_pipeline_spacy_large/logs_%x_%j.out
#SBATCH --error=logs/llm_pipeline_spacy_large/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

mkdir -p logs/llm_pipeline_spacy_large

# Move up 3 directories from sentence_classification_llms to reach root
cd ../../../prediction_classification_experiments-v2

source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

DATE="2026-07-10"
EXPERIMENT="july_2026_results_${DATE}"
# Path relative to prediction_classification_experiments-v2
BASE_RESULTS="../data/classification_results/${EXPERIMENT}"
EMBEDDING_MODEL="spacy_large"

# LLM Configuration
MODEL_NAME="llama-3.1-8b-instruct"
PROMPT_TYPE="chain-of-thought"

echo "============================================================"
echo "Starting LLM Pipeline (In-Domain) — ${MODEL_NAME}"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Prompt Type: ${PROMPT_TYPE}"
echo "Using test sets from: ${EMBEDDING_MODEL}"
echo ""

# ============================================================
# RUN LLM CLASSIFIER ACROSS SEEDS
# ============================================================
for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""
    
    python llm-experiment.py \
        --model_name ${MODEL_NAME} \
        --test_dataset ${BASE_RESULTS}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_test_set.csv \
        --label_column 'Ground Truth' \
        --seed $seed \
        --prompt_type ${PROMPT_TYPE}
done

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ ${MODEL_NAME} (${PROMPT_TYPE}) completed for seeds: 3, 7, 33"
echo ""