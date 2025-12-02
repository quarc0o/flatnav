#!/bin/bash
#SBATCH --job-name=generate-datasets
#SBATCH --output=slurm_scripts/logs/generate-datasets-%j.out
#SBATCH --error=slurm_scripts/logs/generate-datasets-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4


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

make generate-datasets

echo "Dataset generation complete!"
