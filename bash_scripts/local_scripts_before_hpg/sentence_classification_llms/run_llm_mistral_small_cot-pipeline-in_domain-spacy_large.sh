#!/bin/bash

# run_llm_mistral_small_cot-pipeline-in_domain-spacy_large.sh
# Run local LLM sentence classification on validation and test splits
# for seeds 3, 7, and 33.

set -e

cd ../../../prediction_classification_experiments-v2

EXPERIMENT="tolsa_naacl_2026_2027-final"
BASE_RESULTS="../data/classification_results/naacl_2026_submission/${EXPERIMENT}"

MODEL="mistral-small-3.1"
# MODEL="gpt-oss-20b"
PROMPT_TYPE="chain-of-thought"

echo "============================================================"
echo " SENTENCE CLASSIFICATION (LOCAL): ${MODEL}"
echo " Experiment: ${EXPERIMENT}"
echo " Seeds: 3, 7, 33"
echo " Splits: validation and test"
echo "============================================================"

for SEED in 3 7 33; do

    SPLIT_DIR="${BASE_RESULTS}/seed${SEED}/in_domain/splits"

    echo ""
    echo "############################################################"
    echo " SEED: ${SEED}"
    echo "############################################################"

    for SPLIT in val test; do

        DATASET="${SPLIT_DIR}/x_y_${SPLIT}_set.csv"

        echo ""
        echo "============================================================"
        echo " SEED: ${SEED} | SPLIT: ${SPLIT}"
        echo "============================================================"
        echo "Dataset: ${DATASET}"

        if [[ ! -f "${DATASET}" ]]; then
            echo "ERROR: Dataset not found: ${DATASET}"
            exit 1
        fi

        python llm-experiment.py \
            --model_name "${MODEL}" \
            --test_dataset "${DATASET}" \
            --label_column 'Ground Truth' \
            --seed "${SEED}" \
            --prompt_type "${PROMPT_TYPE}"

    done
done

echo ""
echo "======================================"
echo " PIPELINE COMPLETE"
echo "======================================"
echo "Model: ${MODEL}"
echo "Seeds completed: 3, 7, 33"
echo "Splits completed: validation and test"
echo "Total runs: 6"
echo ""
