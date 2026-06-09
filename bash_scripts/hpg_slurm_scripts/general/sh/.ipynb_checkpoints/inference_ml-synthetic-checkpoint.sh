#!/bin/bash
# inference_ml-synthetic.sh

cd ../prediction_classification_experiments-v2

# Chronicle 2050 dataset
echo "chronicle 2050 dataset!"
uv run --active python3 ml-inference.py \
    --dataset chronicle2050/data.csv \
    --experiment combined-full_synthetic-v1_2026-03-07 \
    --seed 40 \
    --text_column "sentence" \
    --output_name "chronicle2050"

# Financial Phrase Bank dataset
echo "financial phrasebank dataset!"
uv run --active python3 ml-inference.py \
    --dataset financial_phrase_bank/annotators/fpb-maya-binary-imbalanced-96d-v1.csv \
    --experiment combined-full_synthetic-v1_2026-03-07 \
    --seed 40 \
    --text_column "Base Sentence" \
    --output_name "financial_phrasebank-imbalanced"

# Sentiment140 dataset
echo "sentiment140 dataset!"
uv run --active python3 ml-inference.py \
    --dataset sentiment140/data.csv \
    --experiment combined-full_synthetic-v1_2026-03-07 \
    --seed 40 \
    --text_column "text" \
    --output_name "sentiment140"


echo ""
echo "✓ All inference complete!"
echo ""
echo "Next steps:"
echo "  - Statistical analysis: prediction_classification_experiments-v2/evaluations.ipynb"fda
echo "  - Extract properties: script_experiments/extract_prediction_properties.py"
echo ""