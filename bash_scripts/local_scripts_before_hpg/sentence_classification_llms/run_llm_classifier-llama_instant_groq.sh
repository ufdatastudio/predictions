#!/bin/bash
# run_llm_classifier-llama_instant_groq.sh - Run LLM sentence classification for all seeds
#
# Usage:
#   chmod +x run_llm_classifier-llama_instant_groq.sh
#   bash run_llm_classifier-llama_instant_groq.sh

set -e

cd ../../../prediction_classification_experiments-v2

# EXPERIMENT="eacl_2026_results_$(date +%Y-%m-%d)"
EXPERIMENT="july_2026_results_2026-07-08"
BASE_RESULTS="../data/classification_results/${EXPERIMENT}"
EMBEDDING_MODEL="spacy_large"  # Which embedding model's test set to use

echo "============================================================"
echo "     SENTENCE CLASSIFICATION (LOCAL): llama-3.1-8b-instant"
echo "============================================================"

for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""

    python llm-classifiers.py \
        --model_name llama-3.1-8b-instant \
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