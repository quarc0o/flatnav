#!/bin/bash
#SBATCH --job-name=flatnav
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --partition=CPUQ
#SBATCH --output=logs/flatnav-%j.out
#SBATCH --error=logs/flatnav-%j.err

# Usage: sbatch slurm_run.sh mnist-bench-flatnav-hnsw
# Or: sbatch slurm_run.sh mnist-bench-flatnav-alpha

MAKE_TARGET=$1

if [ -z "$MAKE_TARGET" ]; then
    echo "Error: No make target specified"
    echo "Usage: sbatch slurm_run.sh <make_target>"
    echo "Example: sbatch slurm_run.sh mnist-bench-flatnav-hnsw"
    exit 1
fi

echo "=========================================="
echo "FlatNav Benchmark Job"
echo "=========================================="
echo "Make target: $MAKE_TARGET"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Start time: $(date)"
echo "=========================================="

# Load environment - IMPORTANT: purge first to avoid conflicts
module purge
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environmet
source $HOME/code/flatnav/venv/bin/activate

# Verify flatnav is installed
echo "Verifying flatnav installation..."
python -c "import flatnav; print(f'Flatnav version: {flatnav.__version__}')" || {
    echo "ERROR: flatnav not installed properly!"
    exit 1
}

# Go to experiments directory
cd $HOME/code/flatnav/experiments

# Run the make target
echo ""
echo "Running: make $MAKE_TARGET"
echo "=========================================="
make $MAKE_TARGET

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully completed: $MAKE_TARGET"
else
    echo "✗ Failed: $MAKE_TARGET (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE