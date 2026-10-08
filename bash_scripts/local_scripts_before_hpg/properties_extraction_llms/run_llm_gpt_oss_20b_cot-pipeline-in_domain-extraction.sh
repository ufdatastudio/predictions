#!/bin/bash
#
# Usage:
#   chmod +x run_llm_gpt_oss_20b_cot-pipeline-in_domain-extraction.sh
#   bash run_llm_gpt_oss_20b_cot-pipeline-in_domain-extraction.sh

set -e

cd ../../../properties_extraction_experiments

echo "Starting LLM Property Extraction Pipeline — Ground Truth"
echo "Model: gpt-oss-20b"
echo "Prompt Type: chain-of-thought"
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
        --model_name "gpt-oss-20b" \
        --task_name extraction \
        --prompt_type chain-of-thought \
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
echo "✓ Model: gpt-oss-20b"
echo "✓ Prompt type: chain-of-thought"
echo "✓ Seeds: 3 7 33"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "======================================"
echo ""