#!/bin/bash
# extract_prediction_properties.sh
#
# Usage:
#   bash extract_prediction_properties.sh

cd ../script_experiments


echo "============================================"
echo "Bash: Tense Extraction"
echo "============================================"

models = ["llama-3.1-8b-instant", "openai/gpt-oss-120b", "mistral-7b-instruct"]
for model in models: 
    # Combined synthetic dataset
    python3 tense_extraction.py \
        --dataset dataset_analyses/2026-02-26/combined_predictions_and_observations-v1.csv \
        --models model

    python3 tense_extraction.py \
        --dataset financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv \
        --models model

    # Chronicle 2050
    python3 tense_extraction.py \
        --dataset chronicle2050/data.csv \
        --models model