
#!/bin/bash
#SBATCH --job-name=avg_llm_metrics
#SBATCH --output=logs/avg_llm_results/logs_%x_%j.out
#SBATCH --error=logs/avg_llm_results/logs_%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dj.brinkley@ufl.edu

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
PREDICTION_REPO="$PROJECT_DIR/../../../../prediction_classification_experiments-v2"
RESULTS_DIR="$PROJECT_DIR/../../../../data/classification_results/naacl_2026_submission"
EXPERIMENT="tolsa_naacl_2026_2027-final"

mkdir -p "$PROJECT_DIR/logs/avg_llm_results"

cd "$PREDICTION_REPO"
source /orange/ufdatastudios/dj.brinkley/predictions/.venv/bin/activate

echo "Experiment: $EXPERIMENT"
echo "Results directory: $RESULTS_DIR/$EXPERIMENT"
echo "Seeds: 3 7 33"

python -u average_llm_metrics_by_seed.py \
    --results_dir "$RESULTS_DIR" \
    --experiment "$EXPERIMENT" \
    --seeds 3 7 33

echo "LLM metric averaging complete."