#!/bin/bash
#SBATCH --job-name=ml_spacy_medium_word
#SBATCH --output=logs/ml_pipeline_spacy_medium_word/logs_%x_%j.out
#SBATCH --error=logs/ml_pipeline_spacy_medium_word/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

mkdir -p logs/ml_pipeline_spacy_medium_word

cd ../../../prediction_classification_experiments-v2

source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

DATE=$(date +%Y-%m-%d)
EXPERIMENT="july_2026_results_${DATE}"
EMBEDDING_MODEL="spacy_medium"

echo "============================================================"
echo "ML Pipeline (In-Domain Baseline) — ${EMBEDDING_MODEL}"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Embedding Model: ${EMBEDDING_MODEL}"
echo "Current directory: $(pwd)"
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
    --output_name july_2026_results \
    --no_version

echo "Dataset ready."

# ============================================================
# TRAIN, TEST & EVALUATE
# ============================================================
echo ""
echo "Running Baseline (Standard) — Seeds 3, 7, 33"
for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""
    echo ">>> Running Baseline (Standard) — ${EMBEDDING_MODEL}"
    python ml-experiment.py \
        --dataset ../data/combined_datasets/july_2026_results/july_2026_results.csv \
        --val_size 0.2 \
        --seed $seed \
        --embedding_model ${EMBEDDING_MODEL} \
        --embedding_level word_averaged
done

# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "All training complete. Aggregating results..."
echo "======================================"
mkdir -p ../data/classification_results/${EXPERIMENT}/averaged/

python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --embedding_model ${EMBEDDING_MODEL} \
    --experiments seed3 seed7 seed33

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ ${EMBEDDING_MODEL} experiments completed for seeds: 3, 7, 33"
echo "✓ Results aggregated and saved"
echo ""