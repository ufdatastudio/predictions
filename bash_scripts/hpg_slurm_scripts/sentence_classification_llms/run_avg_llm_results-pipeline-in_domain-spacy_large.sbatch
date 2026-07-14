#!/bin/bash
# NOTE: Run all LLM classifier scripts (run_llm_pipeline-in_domain-*.sbatch) before running this.
#SBATCH --job-name=avg_llm_results_spacy_large
#SBATCH --output=logs/avg_llm_results/logs_%x_%j.out
#SBATCH --error=logs/avg_llm_results/logs_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu
  
set -e
  
# Create log directory if it doesn't exist
mkdir -p logs/avg_llm_results
  
cd ../../../prediction_classification_experiments-v2
  
# Activate virtual environment
source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate
  
DATE="2026-07-10"
EXPERIMENT="july_2026_results_${DATE}"
EMBEDDING_MODEL="spacy_large"  # Which embedding model's test sets were used
  
echo "============================================================"
echo "     COMBINE & AVERAGE LLM CLASSIFICATION RESULTS"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo ""
  
# ============================================================
# COMBINE LLM OUTPUTS
# ============================================================
echo "======================================"
echo "Combining LLM classifier outputs..."
echo "======================================"
python combine_llm_classifiers_output.py \
    --experiment ${EXPERIMENT} \
    --embedding_model ${EMBEDDING_MODEL}
  
# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "Aggregating results..."
echo "======================================"
mkdir -p ../data/classification_results/${EXPERIMENT}/averaged/

python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --experiments seed3 seed7 seed33 \
    --model_type llm \
    --embedding_model ${EMBEDDING_MODEL}
  
echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ LLM outputs combined for seeds: 3, 7, 33"
echo "✓ LLM results averaged for seeds: 3, 7, 33"
echo "✓ Embedding model: ${EMBEDDING_MODEL}"
echo "✓ Results saved to: ../data/classification_results/${EXPERIMENT}/averaged/"