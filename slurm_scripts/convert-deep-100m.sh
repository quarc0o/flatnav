#!/bin/bash
#SBATCH --job-name=convert-deep
#SBATCH --time=1:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=CPUQ
#SBATCH --output=logs/convert-deep-%j.out
#SBATCH --error=logs/convert-deep-%j.err

# Convert deep base.1B.fbin to train_100m.npy
# Only extracts the first 100M vectors to avoid loading entire 384GB file

echo "=========================================="
echo "Converting Deep Dataset"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Start time: $(date)"
echo "=========================================="

# Load environment
module purge
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source $HOME/code/flatnav/venv/bin/activate

cd $HOME/code/flatnav

# Input and output paths - adjust INPUT_FILE if your file is named differently
INPUT_FILE="data/deep/deep_base.1B.fbin"
OUTPUT_DIR="data/deep"

# Check if input exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Please update INPUT_FILE path in this script"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Run the conversion
python -c "
import numpy as np
import os

input_path = '$INPUT_FILE'
output_dir = '$OUTPUT_DIR'
num_vectors = 100_000_000  # 100M

print(f'Reading from: {input_path}')

with open(input_path, 'rb') as f:
    # Read header
    total_vectors = np.fromfile(f, dtype=np.uint32, count=1)[0]
    dim = np.fromfile(f, dtype=np.uint32, count=1)[0]

    print(f'Total vectors: {total_vectors:,}')
    print(f'Dimension: {dim}')
    print(f'Extracting first {num_vectors:,} vectors...')

    # Determine dtype from file extension
    dtype = np.float32 if input_path.endswith('fbin') else np.uint8

    # Read only the first 100M vectors
    data = np.fromfile(f, dtype=dtype, count=num_vectors * dim)
    data = data.reshape(num_vectors, dim)

    print(f'Data shape: {data.shape}')
    print(f'Data dtype: {data.dtype}')

# Save
output_path = os.path.join(output_dir, 'train_100m.npy')
np.save(output_path, data)
print(f'Saved to: {output_path}')

# Also create 10M version
output_path_10m = os.path.join(output_dir, 'train_10m.npy')
np.save(output_path_10m, data[:10_000_000])
print(f'Saved to: {output_path_10m}')
"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Successfully converted deep dataset"
    ls -lh $OUTPUT_DIR/train_*.npy
else
    echo "Failed (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
