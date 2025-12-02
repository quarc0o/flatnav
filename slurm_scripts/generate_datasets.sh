#!/bin/bash
#SBATCH --job-name=generate-datasets
#SBATCH --output=slurm_scripts/logs/generate-datasets-%j.out
#SBATCH --error=slurm_scripts/logs/generate-datasets-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4


# Load environment
module purge
module load Python/3.11.3-GCCcore-12.3.0

# Run the generate-datasets script
cd experiments
python generate-datasets.py \
    --dataset-size 100000 \
    --num-queries 1000 \
    --dimensions 10 50 100 \
    --k 100

echo "Dataset generation complete!"
