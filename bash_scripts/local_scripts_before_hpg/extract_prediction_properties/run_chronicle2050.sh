#!/bin/bash
# Usage:
# chmod +x run_chronicle2050.sh
# bash run_chronicle2050.sh

echo "============================================"
echo "Extracting Properties: Chronicle2050 Dataset"
echo "============================================"

# Navigate relative to where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../script_experiments"

# Record start time
START_TIME=$(date +%s)
echo "Start time: $(date)"

python3 extract_prediction_properties.py \
    --dataset chronicle2050 \
    --models "llama-3.3-70b-versatile" \
    --text_column "Base Sentence"

# Record end time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Convert to hours, minutes, seconds
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo "✓ Finished: Chronicle2050 Dataset"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"