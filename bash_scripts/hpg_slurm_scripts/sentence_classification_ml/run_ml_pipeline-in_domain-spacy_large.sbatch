#!/bin/bash
#SBATCH --job-name=ml_spacy_large
#SBATCH --output=logs/ml_pipeline_spacy_large/logs_%x_%j.out
#SBATCH --error=logs/ml_pipeline_spacy_large/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

mkdir -p logs/ml_pipeline_spacy_large

cd ../../../prediction_classification_experiments-v2

source /home/dj.brinkley/orange_dj.brinkley/predictions/.venv/bin/activate

echo "Starting ML Pipeline (In-Domain Baseline) — spacy_large"
echo "Current directory: $(pwd)"

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
    echo ">>> Running Baseline (Standard) — spacy_large"
    python ml-train.py \
        --dataset ../data/combined_datasets/july_2026_results/july_2026_results.csv \
        --val_size 0.2 \
        --seed $seed \
        --embedding_model spacy_large
done

# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "All training complete. Aggregating results..."
echo "======================================"
EXPERIMENT="july_2026_results_$(date +%Y-%m-%d)"

python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --embedding_model spacy_large \
    --experiments seed3 seed7 seed33

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ spacy_large experiments completed for seeds: 3, 7, 33"
echo "✓ Results aggregated and saved"
echo ""