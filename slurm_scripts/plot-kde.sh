#!/bin/bash
#SBATCH --job-name=plot-kde
#SBATCH --output=slurm_scripts/logs/plot-kde-%j.out
#SBATCH --error=slurm_scripts/logs/plot-kde-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Usage: sbatch slurm_scripts/plot-kde.sh
# Example: sbatch slurm_scripts/plot-kde.sh

echo "=========================================="
echo "Plot KDE Distributions"
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

echo "Checking required packages..."
python -c "
import sys
packages = ['numpy', 'matplotlib', 'seaborn', 'scipy']
missing = []
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  ✓ {pkg} ({version})')
    except ImportError:
        print(f'  ✗ {pkg} - MISSING!')
        missing.append(pkg)
if missing:
    print(f'\nERROR: Missing packages: {missing}')
    sys.exit(1)
" || {
    echo "ERROR: Required packages not installed!"
    exit 1
}

# Go to experiments directory
echo ""
echo "Working directory: $HOME/code/flatnav/experiments"
cd $HOME/code/flatnav/experiments

echo ""
echo "Checking for data files..."
if [ -d "../node-access-distributions" ]; then
    echo "Distribution directory found:"
    ls -lh ../node-access-distributions/*.json 2>/dev/null | wc -l | xargs echo "  JSON files found:"
else
    echo "  WARNING: Distribution directory not found!"
fi

echo ""
echo "Running: make kde-plots"
echo "=========================================="
make kde-plots

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully plotted kde!"
    echo "Results saved to: ../node-access-distributions/"
else
    echo "✗ Failed to plot kde (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
