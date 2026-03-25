#!/bin/bash
# train_ml_models.sh

cd ../../../prediction_classification_experiments-v2

echo "============================================================"
echo "          SENTENCE CLASSIFICATION (LOCAL): gpt-oss-120b"
echo "============================================================"

python llm-classifiers.py \
    --model_name gpt-oss-120b \
    --label_column 'Sentence Label'
