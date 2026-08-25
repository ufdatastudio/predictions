#!/bin/bash
#SBATCH --job-name=prop_llama_31_8b_cls
#SBATCH --output=logs/prop_extraction_llama_31_8b/logs_%x_%j.out
#SBATCH --error=logs/prop_extraction_llama_31_8b/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

mkdir -p logs/prop_extraction_llama_31_8b

cd ../../../prediction_classification_experiments-v2

source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

MODEL_NAME="llama-3.1-8b-instruct"
PROMPT_TYPE="zero-shot"
TASK_NAME="classification"

echo "============================================================"
echo "Property Extraction Pipeline — Classification"
echo "Model: ${MODEL_NAME}"
echo "Prompt Type: ${PROMPT_TYPE}"
echo "Current directory: $(pwd)"
echo "============================================================"
echo ""

# ============================================================
# PRE-GENERATE COMBINED DATASET
# ============================================================
echo ""
echo "======================================"
echo "Pre-generating combined dataset..."
echo "======================================"

python create_combined_dataset.py \
    --datasets synthetic financial_phrasebank chronicle2050 timebank yt news_api mf_climate clients_rivals_rouges forecast_bench \
    --predictions_only_datasets yt news_api mf_climate clients_rivals_rouges forecast_bench \
    --output_name naacl_2026_submission \
    --no_version

echo "Dataset ready."

# ============================================================
# EXTRACT PROPERTIES — CLASSIFICATION
# ============================================================
echo ""
echo "======================================"
echo "Running Classification Extraction — Seed 3"
echo "======================================"

cd ../properties_extraction_experiments
echo "Current directory: $(pwd)"

for seed in 3; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""

    python llm-experiment.py \
        --dataset_path combined_datasets/naacl_2026_submission/naacl_2026_submission.csv \
        --model_name ${MODEL_NAME} \
        --task_name ${TASK_NAME} \
        --prompt_type ${PROMPT_TYPE} \
        --sample_fraction 0.1 \
        --seed $seed
done

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ Classification property extraction completed"
echo "✓ Model: ${MODEL_NAME}"
echo "✓ Prompt type: ${PROMPT_TYPE}"
echo "✓ Seeds: 3"
echo ""