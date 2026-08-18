#!/bin/bash
# Usage:
# chmod +x run_llm_pipeline-ground_truth.sh
# bash run_llm_pipeline-ground_truth.sh

set -e

# Navigate to the correct directory 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../prediction_classification_experiments-v2"

# source .venv_predictions/bin/activate

echo "============================================================"
echo "Extract Properties: Ground Truth (openai/gpt-oss-20b)"
echo "============================================================"

START_TIME=$(date +%s)
echo "Start time: $(date)"

echo ""
echo "======================================"
echo "STEP 1: Pre-generating full combined dataset"
echo "======================================"

python3 create_combined_dataset.py \
    --datasets synthetic financial_phrasebank chronicle2050 timebank yt news_api mf_climate \
    --predictions_only_datasets yt news_api mf_climate \
    --output_name naacl_2026_submission \
    --no_version

echo "✓ Full dataset ready."
echo ""

cd "../properties_extraction_experiments"

echo "======================================"
echo "STEP 2: Running property extraction (10% Sample)"
echo "======================================"

for seed in 3; do
    echo ""
    echo "------------------------------------------------------------"
    echo "                      SEED: $seed"
    echo "------------------------------------------------------------"
    
    python3 llm-experiment.py \
        --dataset_path combined_datasets/naacl_2026_submission/naacl_2026_submission.csv \
        --model_name "openai/gpt-oss-20b" \
        --task_name ground_truth \
        --sample_fraction 0.1 \
        --seed $seed
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "✓ Finished: Ground Truth — openai/gpt-oss-20b"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "============================================================"