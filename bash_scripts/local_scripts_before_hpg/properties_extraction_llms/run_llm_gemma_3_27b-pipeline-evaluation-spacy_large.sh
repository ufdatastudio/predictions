#!/bin/bash
# run_llm_gemma_3_27b-pipeline-evaluation-spacy_large.sh
#
# Usage:
#   chmod +x run_llm_gemma_3_27b-pipeline-evaluation-spacy_large.sh
#   bash run_llm_gemma_3_27b-pipeline-evaluation-spacy_large.sh

set -e

cd ../../../prediction_classification_experiments-v2

MODEL_NAME="gpt-oss-120b x gemma-3-27b-it"
EMBEDDING_MODEL="spacy_large"
SEED=3

echo "Starting LLM Property Extraction Evaluation"
echo "Model: ${MODEL_NAME}"
echo "Embedding Model: ${EMBEDDING_MODEL}"
echo "Seed: ${SEED}"
echo "Current directory: $(pwd)"

START_TIME=$(date +%s)
echo "Start time: $(date)"

# ============================================================
# EVALUATE PROPERTIES
# ============================================================
echo ""
echo "======================================"
echo "Running Property Extraction Evaluation — Seed ${SEED}"
echo "======================================"

cd ../properties_extraction_experiments
echo "Current directory: $(pwd)"

echo ""
echo "============================================================"
echo "                      SEED: ${SEED}"
echo "============================================================"
echo ""

python3 evaluate_properties_extraction.py \
    --y_path extraction_results/naacl_2026_submission/ground_truth/zero-shot/seed${SEED}/gpt-oss-120b/extracted_properties.csv \
    --y_hat_path extraction_results/naacl_2026_submission/classification/zero-shot/seed${SEED}/gemma-3-27b-it/extracted_properties.csv \
    --model_name "${MODEL_NAME}" \
    --seed ${SEED} \
    --embedding_model ${EMBEDDING_MODEL}

# ============================================================
# COMPLETE
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ Property extraction evaluation completed"
echo "✓ Model: ${MODEL_NAME}"
echo "✓ Embedding Model: ${EMBEDDING_MODEL}"
echo "✓ Seed: ${SEED}"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""