#!/bin/bash

#
# Usage:
#   chmod +x run_llm_gpt_oss_120b_zero_shot-pipeline-in_domain-extraction.sh
#   bash run_llm_gpt_oss_120b_zero_shot-pipeline-in_domain-extraction.sh

set -e

cd ../../../properties_extraction_experiments

echo "Starting LLM Property Extraction Pipeline — Ground Truth"
echo "Model: gpt-oss-120b"
echo "Prompt Type: zero-shot"
echo "Seeds: 3 7 33"
echo "Current directory: $(pwd)"

START_TIME=$(date +%s)

echo "Start time: $(date)"

# ============================================================
# RUN GROUND-TRUTH PROPERTY EXTRACTION
# ============================================================

echo ""
echo "======================================"
echo "Running Ground Truth Extraction — Seeds 3, 7, 33"
echo "======================================"

for seed in 3 7 33; do
    echo ""
    echo "Running seed: $seed"

    python3 llm-experiment.py \
        --dataset_path ../data/extract_tolsa_properties_results/naacl_2026_submission/ground_truth/extracted_properties-ground_truth_only.csv \
        --model_name "gpt-oss-120b" \
        --task_name extraction \
        --prompt_type zero-shot \
        --seed $seed
done

# ============================================================
# COMPLETE
# ============================================================

END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ Ground-truth property extraction completed"
echo "✓ Model: gpt-oss-120b"
echo "✓ Prompt type: zero-shot"
echo "✓ Seeds: 3 7 33"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "======================================"
echo ""