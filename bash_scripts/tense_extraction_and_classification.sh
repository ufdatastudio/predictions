#!/bin/bash
# tense_extraction_and_classification.sh
#
# Runs tense extraction on the combined synthetic dataset and real datasets,
# then classifies tense using the rule-based AUX+MD approach.
# Results are combined separately in a Jupyter notebook.
#
# Usage:
#   bash tense_extraction_and_classification.sh

cd ../script_experiments

# ============================================================
# 1. TENSE EXTRACTION
# ============================================================
echo "============================================"
echo "Bash: Tense Extraction"
echo "============================================"

# Combined synthetic dataset (predictions + observations)
python3 tense_extraction.py \
    --dataset dataset_analyses/2026-02-26/combined_predictions_and_observations-v1.csv \
    --dataset_name synthetic_combined \
    --text_column "Base Sentence" \
    --spacy_disable_components [] \
    --visualize False

# Financial phrasebank – imbalanced
python3 tense_extraction.py \
    --dataset financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv \
    --dataset_name financial_phrasebank_imbalanced \
    --text_column "Base Sentence" \
    --spacy_disable_components [] \
    --visualize False

# Chronicle 2050
python3 tense_extraction.py \
    --dataset chronicle2050/data.csv \
    --dataset_name chronicle2050 \
    --text_column "sentence" \
    --spacy_disable_components [] \
    --visualize False

# ============================================================
# 2. TENSE CLASSIFICATION
# --dataset points to the POS features CSV produced by tense_extraction.py
# --visualize is not passed – image is always saved, no pop-up window
# ============================================================
echo "============================================"
echo "Bash: Tense Classification"
echo "============================================"

# Combined synthetic dataset
python3 tense_classification.py \
    --dataset tense_extraction/synthetic_combined/pos_features-v1.csv \
    --dataset_name synthetic_combined \
    --pos_feature AUX \
    --detailed_pos_feature MD

# Financial phrasebank – imbalanced
python3 tense_classification.py \
    --dataset tense_extraction/financial_phrasebank_imbalanced/pos_features-v1.csv \
    --dataset_name financial_phrasebank_imbalanced \
    --pos_feature AUX \
    --detailed_pos_feature MD

# Chronicle 2050
python3 tense_classification.py \
    --dataset tense_extraction/chronicle2050/pos_features-v1.csv \
    --dataset_name chronicle2050 \
    --pos_feature AUX \
    --detailed_pos_feature MD

echo "============================================"
echo "Bash: Done"
echo "Results will be combined in the Jupyter notebook."
echo "============================================"