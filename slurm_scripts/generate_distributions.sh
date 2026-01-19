#!/bin/bash
#SBATCH --job-name=generate-distributions
#SBATCH --output=slurm_scripts/logs/generate-distributions-%j.out
#SBATCH --error=slurm_scripts/logs/generate-distributions-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32

# Usage: sbatch slurm_scripts/generate_distributions.sh <ef_construction> <ef_search> <num_node_links> [k]
# Example: sbatch slurm_scripts/generate_distributions.sh 500 500 64 100




echo "=========================================="
echo "Generate Node Access Distributions"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 128G"
echo "Start time: $(date)"
echo ""
echo "Parameters:"
echo "  ef-construction: $EF_CONSTRUCTION"
echo "  ef-search: $EF_SEARCH"
echo "  num-node-links: $NUM_NODE_LINKS"
echo "  k: $K"
echo "=========================================="

# Load environment - IMPORTANT: purge first to avoid conflicts
module purge
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environmet
source $HOME/code/flatnav/venv/bin/activate

# Verify flatnav is installed
echo ""
echo "Verifying flatnav installation..."
python -c "import flatnav; print(f'Flatnav version: {flatnav.__version__}')" || {
    echo "ERROR: flatnav not installed properly!"
    exit 1
}

# Go to experiments directory
cd $HOME/code/flatnav/experiments


make generate-distributions

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully generated distributions!"
    echo "Results saved to: /root/node-access-distributions/"
else
    echo "✗ Failed to generate distributions (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
