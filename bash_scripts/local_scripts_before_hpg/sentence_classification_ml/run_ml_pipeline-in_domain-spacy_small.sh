#!/bin/bash
# run_ml_pipeline-in_domain-spacy_small.sh - Baseline with three seeds
#
# Usage:
#   chmod +x run_ml_pipeline-in_domain-spacy_small.sh
#   bash run_ml_pipeline-in_domain-spacy_small.sh
set -e
cd ../../../prediction_classification_experiments-v2
echo "Starting ML Pipeline (In-Domain Baseline) — spacy_small"
echo "Current directory: $(pwd)"
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
    echo ">>> Running Baseline (Standard) — spacy_small"
    python ml-train.py \
        --dataset ../data/combined_datasets/july_2026_results/july_2026_results.csv \
        --val_size 0.2 \
        --seed $seed \
        --embedding_model spacy_small
done
# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "All training complete. Aggregating results..."
echo "======================================"
EXPERIMENT="july_2026_results_$(date +%Y-%m-%d)"
mkdir -p ../data/classification_results/${EXPERIMENT}/averaged/in_dataset_comparisons/
python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --embedding_model spacy_small \
    --experiments seed3 seed7 seed33
echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ spacy_small experiments completed for seeds: 3, 7, 33"
echo "✓ Results aggregated and saved"
echo ""