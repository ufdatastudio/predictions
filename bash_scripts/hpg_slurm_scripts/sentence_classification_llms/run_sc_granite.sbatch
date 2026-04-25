#!/bin/bash
#SBATCH --job-name=run_sc_granite
#SBATCH --output=../logs/sentence_classification_llms/run_sc_granite_logs/logs_%x_%j.out
#SBATCH --error=../logs/sentence_classification_llms/run_sc_granite_logs/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

cd ../../../prediction_classification_experiments-v2

EXPERIMENT="synthetic-fpb-chronicle2050-yt-news-timebank-mf_climate_$(date +%Y-%m-%d)"
BASE_RESULTS="../data/classification_results/${EXPERIMENT}"

echo "============================================================"
echo "     SENTENCE CLASSIFICATION (HPG): granite-3.3-8b-instruct"
echo "============================================================"

for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""

    python llm-classifiers.py \
        --model_name granite-3.3-8b-instruct \
        --test_dataset ${BASE_RESULTS}/seed${seed}/in_domain/x_y_test_set.csv \
        --label_column 'Ground Truth' \
        --seed $seed
done

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ LLM classification completed for seeds: 3, 7, 33"
echo ""