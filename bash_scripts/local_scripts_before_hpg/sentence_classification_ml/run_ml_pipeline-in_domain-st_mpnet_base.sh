#!/bin/bash
# run_ml_pipeline-in_domain-st_mpnet_base.sh
# Usage:
#   chmod +x run_ml_pipeline-in_domain-st_mpnet_base.sh
#   bash run_ml_pipeline-in_domain-st_mpnet_base.sh

set -e

# Fix for macOS sentence-transformers segfault
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ../../../prediction_classification_experiments-v2

echo "Starting ML Pipeline (In-Domain Baseline) — st_mpnet_base"
echo "Current directory: $(pwd)"

# ============================================================
# PRE-GENERATE DATASET
# ============================================================
echo ""
echo "======================================"
echo "Pre-generating combined dataset..."
echo "======================================"

python3 create_combined_dataset.py \
    --datasets synthetic financial_phrasebank chronicle2050 timebank yt news_api mf_climate clients_rivals_rouges forecast_bench \
    --predictions_only_datasets yt news_api mf_climate clients_rivals_rouges forecast_bench \
    --output_name eacl_2026_results \
    --no_version

echo "Dataset ready."

# ============================================================
# TRAIN / TEST
# ============================================================
echo ""
echo "Running Baseline (st_mpnet_base) — Seeds 3, 7, 33"

for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""

    python ml-train.py \
        --dataset ../data/combined_datasets/eacl_2026_results/eacl_2026_results.csv \
        --val_size 0.2 \
        --seed $seed \
        --embedding_model st_mpnet_base
done

# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "Aggregating results..."
echo "======================================"

EXPERIMENT="eacl_2026_results_$(date +%Y-%m-%d)"
mkdir -p ../data/classification_results/${EXPERIMENT}/averaged/in_dataset_comparisons/

python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --embedding_model st_mpnet_base \
    --experiments seed3 seed7 seed33

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ st_mpnet_base experiments completed"
echo ""