#!/bin/bash
set -euo pipefail

write_few_shot_script() {
    local MODEL="$1"
    local FILE="$2"

    cat > "$FILE" <<EOF
#!/bin/bash
set -euo pipefail

# Run local few-shot LLM sentence classification on the test split.

cd ../../../../prediction_classification_experiments-v2

if [[ ! -f "llm-experiment.py" ]]; then
    echo "ERROR: llm-experiment.py not found in $(pwd)" >&2
    exit 1
fi

EXPERIMENT="tolsa_naacl_2026_2027-final"
BASE_RESULTS="../data/classification_results/naacl_2026_submission/\${EXPERIMENT}"
MODEL="${MODEL}"
PROMPT_TYPE="few-shot"
LABEL_COLUMN="Ground Truth"

echo "============================================================"
echo "Model: \${MODEL}"
echo "Experiment: \${EXPERIMENT}"
echo "Seeds: 3 7 33"
echo "Split: test"
echo "Prompt: \${PROMPT_TYPE}"
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
    echo "===== MODEL=\${MODEL} SEED=\${SEED} SPLIT=test ====="
    echo "Dataset: \${DATASET}"
    echo "Few-shot samples: \${FEW_SHOT_SAMPLES}"

    python llm-experiment.py \\
        --model_name "\${MODEL}" \\
        --test_dataset "\${DATASET}" \\
        --label_column "\${LABEL_COLUMN}" \\
        --seed "\${SEED}" \\
        --prompt_type "\${PROMPT_TYPE}" \\
        --few_shot_samples "\${FEW_SHOT_SAMPLES}"
done

echo
echo "======================================"
echo "PIPELINE COMPLETE"
echo "======================================"
echo "Model: \${MODEL}"
echo "Seeds completed: 3 7 33"
echo "Split completed: test"
echo "Total runs: 3"
EOF

    chmod +x "$FILE"
    echo "Created: $FILE"
}

write_few_shot_script \
    "gpt-oss-20b" \
    "run_local_gpt_oss_20b_few_shot.sh"

write_few_shot_script \
    "llama-3.3-70b-instruct" \
    "run_local_llama_3_3_70b_instruct_few_shot.sh"

write_few_shot_script \
    "mistral-small-3.1" \
    "run_local_mistral_small_3_1_few_shot.sh"

echo
echo "All three local few-shot scripts created."