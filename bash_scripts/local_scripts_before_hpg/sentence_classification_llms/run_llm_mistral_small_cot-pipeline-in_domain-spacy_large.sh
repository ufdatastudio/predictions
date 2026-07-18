#!/bin/bash
# run_llm_classifiers-mistral_small.sh - Run LLM sentence classification for all seeds
#
# Usage:
#   chmod +x run_llm_mistral_small_cot-pipeline-in_domain-spacy_large.sh
#   bash run_llm_mistral_small_cot-pipeline-in_domain-spacy_large.sh

set -e

cd ../../../prediction_classification_experiments-v2

# EXPERIMENT="eacl_2026_results_$(date +%Y-%m-%d)"
EXPERIMENT="july_2026_results_2026-07-08"
BASE_RESULTS="../data/classification_results/${EXPERIMENT}"
EMBEDDING_MODEL="spacy_large"  # Which embedding model's test set to use

echo "============================================================"
echo "     SENTENCE CLASSIFICATION (LOCAL): mistral-small-3.1"
echo "============================================================"

for seed in 3; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""

    python llm-experiment.py \
        --model_name mistral-small-3.1 \
        --test_dataset ${BASE_RESULTS}/seed${seed}/in_domain/${EMBEDDING_MODEL}/x_y_test_set.csv \
        --label_column 'Ground Truth' \
        --seed $seed \
        --prompt_type chain-of-thought
done

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ Mistral classification completed for seeds: 3, 7, 33"
echo ""