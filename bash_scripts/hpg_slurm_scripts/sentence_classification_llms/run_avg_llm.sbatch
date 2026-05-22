# NOTE: Run all LLM classifier scripts (run_sc_gemma.sbatch, run_sc_granite.sbatch, etc.) before running this.
#!/bin/bash
#SBATCH --job-name=run_sc_avg_llm
#SBATCH --output=../logs/sentence_classification_llms/run_sc_avg_llm_logs/logs_%x_%j.out
#SBATCH --error=../logs/sentence_classification_llms/run_sc_avg_llm_logs/logs_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu
set -e
cd ../../../prediction_classification_experiments-v2

EXPERIMENT="emnlp_2026_results_$(date +%Y-%m-%d)"

echo "============================================================"
echo "     COMBINE & AVERAGE LLM CLASSIFICATION RESULTS"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo ""

# ============================================================
# COMBINE LLM OUTPUTS
# ============================================================
echo "======================================"
echo "Combining LLM classifier outputs..."
echo "======================================"
python combine_llm_classifiers_output.py \
    --experiment ${EXPERIMENT}

# ============================================================
# AGGREGATE RESULTS
# ============================================================
echo ""
echo "======================================"
echo "Aggregating results..."
echo "======================================"
python average_classification_results.py \
    --mode single \
    --experiment ${EXPERIMENT} \
    --experiments seed3 seed7 seed33 \
    --model_type llm

echo ""
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "✓ LLM outputs combined for seeds: 3, 7, 33"
echo "✓ LLM results averaged for seeds: 3, 7, 33"
echo "✓ Results saved to: ../data/classification_results/${EXPERIMENT}/averaged/"
echo ""