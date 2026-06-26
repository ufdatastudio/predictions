#!/bin/bash
#SBATCH --job-name=llm_mistral_small
#SBATCH --output=logs/llm_pipeline_mistral_small/logs_%x_%j.out
#SBATCH --error=logs/llm_pipeline_mistral_small/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

# Create log directory if it doesn't exist
mkdir -p logs/llm_pipeline_mistral_small

cd ../../../prediction_classification_experiments-v2

# Activate virtual environment
source /home/dj.brinkley/orange_dj.brinkley/predictions/.venv/bin/activate

EXPERIMENT="eacl_2026_results_$(date +%Y-%m-%d)"
BASE_RESULTS="../data/classification_results/${EXPERIMENT}"
EMBEDDING_MODEL="spacy_large"  # Which embedding model's test set to use

echo "============================================================"
echo "     SENTENCE CLASSIFICATION (HPG): mistral-small-3.1"
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
        --model_name mistral-small-3.1 \
        --test_dataset ${BASE_RESULTS}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_test_set.csv \
        --label_column 'Ground Truth' \
        --seed $seed \
        --prompt_type few-shot
done

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ Mistral classification completed for seeds: 3, 7, 33"
echo ""
