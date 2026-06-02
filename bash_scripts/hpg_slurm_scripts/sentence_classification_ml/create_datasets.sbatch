#!/bin/bash
set -e
cd ../../../prediction_classification_experiments-v2
module load python/3
source ../.venv/bin/activate

echo "Generating all combined datasets..."

python create_combined_dataset.py --datasets predictions non_predictions --output_name combined-full_synthetic-v1 --save_path ../data/combined_datasets/ --no_version
python create_combined_dataset.py --datasets predictions non_predictions financial_phrasebank --output_name combined-synthetic-fpb-v1 --save_path ../data/combined_datasets/ --no_version
python create_combined_dataset.py --datasets predictions non_predictions chronicle2050 --output_name combined-synthetic-chronicle2050-v1 --save_path ../data/combined_datasets/ --no_version
python create_combined_dataset.py --datasets financial_phrasebank chronicle2050 --output_name combined-fpb-chronicle2050-v1 --save_path ../data/combined_datasets/ --no_version
python create_combined_dataset.py --datasets predictions non_predictions financial_phrasebank chronicle2050 --output_name combined-synthetic_fpb_chr2050-v1 --save_path ../data/combined_datasets/ --no_version

echo "✓ All datasets created successfully!"