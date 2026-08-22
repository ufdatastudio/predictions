#!/bin/bash
# run_llm_pipeline-ground_truth.sh
#
# Usage:
#   chmod +x run_llm_pipeline-ground_truth.sh
#   bash run_llm_pipeline-ground_truth.sh

set -e

cd ../../../prediction_classification_experiments-v2

echo "Starting LLM Property Extraction Pipeline — Ground Truth"
echo "Model: openai/gpt-oss-20b"
echo "Prompt Type: zero-shot"
echo "Current directory: $(pwd)"

START_TIME=$(date +%s)
echo "Start time: $(date)"

# ============================================================
# PRE-GENERATE COMBINED DATASET
# ============================================================
echo ""
echo "======================================"
echo "Pre-generating combined dataset..."
echo "======================================"

python3 create_combined_dataset.py \
    --datasets synthetic financial_phrasebank chronicle2050 timebank yt news_api mf_climate clients_rivals_rouges forecast_bench \
    --predictions_only_datasets yt news_api mf_climate clients_rivals_rouges forecast_bench \
    --output_name naacl_2026_submission \
    --no_version

echo "Dataset ready."

# ============================================================
# EXTRACT GROUND-TRUTH PROPERTIES
# ============================================================
echo ""
echo "======================================"
echo "Running Ground Truth Extraction — Seed 7"
echo "======================================"

cd ../properties_extraction_experiments
echo "Current directory: $(pwd)"

for seed in 7; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""

    python3 llm-experiment.py \
        --dataset_path combined_datasets/naacl_2026_submission/naacl_2026_submission.csv \
        --model_name "openai/gpt-oss-20b" \
        --task_name ground_truth \
        --prompt_type zero-shot \
        --sample_fraction 0.1 \
        --seed $seed
done

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
echo "✓ Ground-truth property extraction completed"
echo "✓ Model: openai/gpt-oss-20b"
echo "✓ Prompt type: zero-shot"
echo "✓ Seeds: 7"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""