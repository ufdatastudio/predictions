#!/bin/bash
set -euo pipefail

# Run local few-shot LLM sentence classification on the test split.

cd ../../../../prediction_classification_experiments-v2

if [[ ! -f "llm-experiment.py" ]]; then
    echo "ERROR: llm-experiment.py not found in /Users/detraviousjamaribrinkley/Documents/Development/research_labs/uf_ds/predictions/bash_scripts/local_scripts_before_hpg/sentence_classification_llms/tolsa_naacl_2026_2027" >&2
    exit 1
fi

EXPERIMENT="tolsa_naacl_2026_2027-final"
BASE_RESULTS="../data/classification_results/naacl_2026_submission/${EXPERIMENT}"
MODEL="mistral-small-3.1"
PROMPT_TYPE="few-shot"
LABEL_COLUMN="Ground Truth"

echo "============================================================"
echo "Model: ${MODEL}"
echo "Experiment: ${EXPERIMENT}"
echo "Seeds: 3 7 33"
echo "Split: test"
echo "Prompt: ${PROMPT_TYPE}"
echo "============================================================"

for SEED in 3 7 33; do
    SEED_DIR="${BASE_RESULTS}/seed${SEED}/in_domain"
    DATASET="${SEED_DIR}/splits/x_y_test_set.csv"
    FEW_SHOT_SAMPLES="${SEED_DIR}/llm/few-shot.csv"

    if [[ ! -f "${DATASET}" ]]; then
        echo "ERROR: Dataset not found: ${DATASET}" >&2
        exit 1
    fi

    if [[ ! -f "${FEW_SHOT_SAMPLES}" ]]; then
        echo "ERROR: Few-shot samples not found: ${FEW_SHOT_SAMPLES}" >&2
        exit 1
    fi

    echo
    echo "===== MODEL=${MODEL} SEED=${SEED} SPLIT=test ====="
    echo "Dataset: ${DATASET}"
    echo "Few-shot samples: ${FEW_SHOT_SAMPLES}"

    python llm-experiment.py \
        --model_name "${MODEL}" \
        --test_dataset "${DATASET}" \
        --label_column "${LABEL_COLUMN}" \
        --seed "${SEED}" \
        --prompt_type "${PROMPT_TYPE}" \
        --few_shot_samples "${FEW_SHOT_SAMPLES}"
done

echo
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "Model: ${MODEL}"
echo "Seeds completed: 3 7 33"
echo "Split completed: test"
echo "Total runs: 3"
