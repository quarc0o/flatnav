#!/bin/bash
#SBATCH --job-name=speed-tests
#SBATCH --output=slurm_scripts/logs/speed-tests-%j.out
#SBATCH --error=slurm_scripts/logs/speed-tests-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Usage: sbatch slurm_scripts/speed_test.sh
# Example: sbatch slurm_scripts/speed_test.sh

echo "=========================================="
echo "Run Statistical Hypothesis Tests"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 64G"
echo "Start time: $(date)"
echo "=========================================="



# Load environment - IMPORTANT: purge first to avoid conflicts
module purge
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environmet
source $HOME/code/flatnav/venv/bin/activate

# Verify Python and required packages
echo ""
echo "Verifying environment..."
python --version



# Go to experiments directory
echo ""
echo "Working directory: $HOME/code/flatnav/experiments"
cd $HOME/code/flatnav/experiments

echo ""
echo "Checking for data files..."
if [ -d "../node-access-distributions" ]; then
    echo "Distribution directory found:"
    json_count=$(ls -1 ../node-access-distributions/*.json 2>/dev/null | wc -l)
    pkl_count=$(ls -1 ../node-access-distributions/*.pkl 2>/dev/null | wc -l)
    echo "  JSON files: $json_count"
    echo "  PKL files: $pkl_count"
else
    echo "  WARNING: Distribution directory not found!"
fi

echo ""
echo "Running: make speed-test"
echo "=========================================="
make speed-test

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully completed statistical tests!"
    echo "Results saved to: ../metrics/hypothesis_tests.json"
else
    echo "✗ Failed to run statistical tests (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
