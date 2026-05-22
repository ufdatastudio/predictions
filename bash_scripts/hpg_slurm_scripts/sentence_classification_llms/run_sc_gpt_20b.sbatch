# NOTE: This script only classifies. Run after all LLM classifiers complete, then run run_avg_llm.sbatch to combine and average.
#!/bin/bash
#SBATCH --job-name=run_sc_gpt_20b
#SBATCH --output=../logs/sentence_classification_llms/run_sc_gpt_20b_logs/logs_%x_%j.out
#SBATCH --error=../logs/sentence_classification_llms/run_sc_gpt_20b_logs/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu
set -e
cd ../../../prediction_classification_experiments-v2

EXPERIMENT="emnlp_2026_results_$(date +%Y-%m-%d)"
BASE_RESULTS="../data/classification_results/${EXPERIMENT}"

echo "============================================================"
echo "     SENTENCE CLASSIFICATION (HPG): gpt-oss-20b"
echo "============================================================"
for seed in 3 7 33; do
    echo ""
    echo "============================================================"
    echo "                      SEED: $seed"
    echo "============================================================"
    echo ""
    python llm-classifiers.py \
        --model_name gpt-oss-20b \
        --test_dataset ${BASE_RESULTS}/seed${seed}/in_domain/x_y_test_set.csv \
        --label_column 'Ground Truth' \
        --seed $seed
done

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ GPT 20B classification completed for seeds: 3, 7, 33"
echo ""
