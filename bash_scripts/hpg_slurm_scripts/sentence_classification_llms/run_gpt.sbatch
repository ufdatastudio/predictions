#!/bin/bash
#SBATCH --job-name=run_sc_gpt
#SBATCH --output=logs/logs_%x_%j.out
#SBATCH --error=logs/logs_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -e

cd ../../../prediction_classification_experiments-v2
module load python/3
source ../.venv/bin/activate

echo "============================================================"
echo "          SENTENCE CLASSIFCATION: gpt-oss-120b"
echo "============================================================"

python llm-classifiers.py \
    --model_name gpt-oss-120b \
    --label_column 'Sentence Label'
