
#!/bin/bash
set -euo pipefail

mkdir -p logs/llm_pipeline

write_few_shot_script() {
    local MODEL="$1"
    local JOB="$2"
    local FILE="$3"

    cat > "$FILE" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB}
#SBATCH --output=logs/llm_pipeline/logs_%x_%j.out
#SBATCH --error=logs/llm_pipeline/logs_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -euo pipefail

# Submit this job from sentence_classification_llms.
cd ../../../../prediction_classification_experiments-v2
source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

EXPERIMENT="tolsa_naacl_2026_2027-final"
BASE_RESULTS="../data/classification_results/naacl_2026_submission/\${EXPERIMENT}"
MODEL_NAME="${MODEL}"
PROMPT_TYPE="few-shot"
LABEL_COLUMN="Ground Truth"

echo "============================================================"
echo "Model: \$MODEL_NAME"
echo "Prompt: \$PROMPT_TYPE"
echo "Experiment: \$EXPERIMENT"
echo "Seeds: 3 7 33"
echo "Split: test"
echo "============================================================"

for SEED in 3 7 33; do
    SEED_DIR="\${BASE_RESULTS}/seed\${SEED}/in_domain"
    DATASET="\${SEED_DIR}/splits/x_y_test_set.csv"
    FEW_SHOT_SAMPLES="\${SEED_DIR}/llm/few-shot.csv"

    if [[ ! -f "\${DATASET}" ]]; then
        echo "ERROR: Dataset not found: \${DATASET}" >&2
        exit 1
    fi

    if [[ ! -f "\${FEW_SHOT_SAMPLES}" ]]; then
        echo "ERROR: Few-shot samples not found: \${FEW_SHOT_SAMPLES}" >&2
        exit 1
    fi

    echo
    echo "===== MODEL=\${MODEL_NAME} SEED=\${SEED} SPLIT=test ====="
    echo "Dataset: \${DATASET}"
    echo "Few-shot samples: \${FEW_SHOT_SAMPLES}"

    python llm-experiment.py \\
        --model_name "\${MODEL_NAME}" \\
        --test_dataset "\${DATASET}" \\
        --label_column "\${LABEL_COLUMN}" \\
        --seed "\${SEED}" \\
        --prompt_type "\${PROMPT_TYPE}" \\
        --few_shot_samples "\${FEW_SHOT_SAMPLES}"
done

echo "All few-shot test runs completed for \$MODEL_NAME."
EOF

    chmod +x "$FILE"
    echo "Created: $FILE"
}

write_few_shot_script \
    "gpt-oss-20b" \
    "llm_gpt_oss_20b_few" \
    "run_llm_gpt_oss_20b_few_shot-pipeline-in_domain.sbatch"

write_few_shot_script \
    "llama-3.3-70b-instruct" \
    "llm_llama_3_3_70b_instruct_few" \
    "run_llm_llama_3_3_70b_instruct_few_shot-pipeline-in_domain.sbatch"

write_few_shot_script \
    "mistral-small-3.1" \
    "llm_mistral_small_3_1_few" \
    "run_llm_mistral_small_3_1_few_shot-pipeline-in_domain.sbatch"

echo
echo "All three few-shot SBATCH scripts created."
