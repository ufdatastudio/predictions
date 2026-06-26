#!/bin/bash
#SBATCH --job-name=llm_llama_31_8b
#SBATCH --output=logs/llm_pipeline_llama_31_8b/logs_%x_%j.out
#SBATCH --error=logs/llm_pipeline_llama_31_8b/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

# Create log directory if it doesn't exist
mkdir -p logs/llm_pipeline_llama_31_8b

cd ../../../prediction_classification_experiments-v2

# Activate virtual environment
source /home/dj.brinkley/orange_dj.brinkley/predictions/.venv/bin/activate

EXPERIMENT="eacl_2026_results_$(date +%Y-%m-%d)"
BASE_RESULTS="../data/classification_results/${EXPERIMENT}"
EMBEDDING_MODEL="spacy_large"  # Which embedding model's test set to use

echo "============================================================"
echo "Starting LLM Pipeline (In-Domain) — llama-3.1-8b-instruct"
echo "============================================================"
echo "Using test sets from: ${EMBEDDING_MODEL}"
echo ""

for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""
    python llm-classifiers.py \
        --model_name llama-3.1-8b-instruct \
        --test_dataset ${BASE_RESULTS}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_test_set.csv \
        --label_column 'Ground Truth' \
        --seed $seed \
        --prompt_type zero-shot
done

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ Llama 3.1 8B classification completed for seeds: 3, 7, 33"
echo ""