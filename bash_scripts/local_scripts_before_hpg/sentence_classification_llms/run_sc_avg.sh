# NOTE: Run all LLM classifier scripts (run_sc_gemma.sh, run_sc_granite.sh, etc.) before running this.
#!/bin/bash
# run_sc_avg.sh - Combine and average LLM classification results across seeds
#
# Usage:
#   chmod +x run_sc_avg.sh
#   bash run_sc_avg.sh
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