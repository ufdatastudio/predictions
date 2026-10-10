#!/bin/bash
#SBATCH --job-name=ml_st_minilm_l6
#SBATCH --output=logs/ml_pipeline_st_minilm_l6/logs_%x_%j.out
#SBATCH --error=logs/ml_pipeline_st_minilm_l6/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -euo pipefail

# This script assumes it is submitted from the tolsa_naacl_2026_2027
# directory, and that the logs directory exists before sbatch opens stdout/stderr.
mkdir -p logs/ml_pipeline_st_minilm_l6
cd ../../../../prediction_classification_experiments-v2

source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

EXPERIMENT="tolsa_naacl_2026_2027-final"
OUTPUT_DIR="../data/classification_results/naacl_2026_submission"
DATASET="../data/combined_datasets/tolsa_naacl_2026_2027-final/tolsa_naacl_2026_2027_main_columns_v1.csv"
EMBEDDING_MODEL="st_minilm_l6"
SEEDS=(3 7 33)

echo "============================================================"
echo "ML Pipeline (In-Domain Baseline) — ${EMBEDDING_MODEL}"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Dataset: ${DATASET}"
echo "Output root: ${OUTPUT_DIR}/${EXPERIMENT}"
echo "Current directory: $(pwd)"
echo

if [[ ! -f "${DATASET}" ]]; then
    echo "ERROR: Dataset not found: ${DATASET}" >&2
    exit 1
fi

echo "Running seeds: ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
    echo
    echo "============================================================"
    echo "SEED: ${seed}"
    echo "============================================================"

    python -u ml-experiment.py \
        --dataset "${DATASET}" \
        --output_dir "${OUTPUT_DIR}/${EXPERIMENT}" \
        --val_size 0.2 \
        --seed "${seed}" \
        --embedding_model "${EMBEDDING_MODEL}"
done

echo
echo "======================================"
echo "All training complete. Aggregating results..."
echo "======================================"

python -u average_classification_results.py \
    --mode single \
    --experiment "${EXPERIMENT}" \
    --model_type ml \
    --embedding_model "${EMBEDDING_MODEL}" \
    --experiments seed3 seed7 seed33 \
    --results_dir "${OUTPUT_DIR}"

echo
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "Model: ${EMBEDDING_MODEL}"
echo "Seeds completed: ${SEEDS[*]}"
echo "Results directory: ${OUTPUT_DIR}/${EXPERIMENT}"
echo "======================================"
