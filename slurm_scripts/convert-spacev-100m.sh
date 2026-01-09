#!/bin/bash
#SBATCH --job-name=convert-spacev
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=CPUQ
#SBATCH --output=logs/convert-spacev-%j.out
#SBATCH --error=logs/convert-spacev-%j.err

# Convert SPACEV base.100m.i8bin to train_100m.npy
# SPACEV is int8 format, 100 dimensions
# Download first: wget https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/base.100m.i8bin -P data/msspacev/

echo "=========================================="
echo "Converting SPACEV Dataset"
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

# Input and output paths
INPUT_FILE="data/msspacev/base.100M.i8bin"
OUTPUT_DIR="data/msspacev"

# Check if input exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Download it first with:"
    echo "  wget https://huggingface.co/datasets/unum-cloud/ann-spacev-100m/resolve/main/base.100M.i8bin -P data/msspacev/"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Run the conversion
python -c "
import numpy as np

input_path = '$INPUT_FILE'
output_dir = '$OUTPUT_DIR'

print(f'Reading from: {input_path}')

with open(input_path, 'rb') as f:
    # Read header: num_vectors (uint32), dimension (uint32)
    num_vectors = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
    dim = int(np.fromfile(f, dtype=np.uint32, count=1)[0])

    print(f'Number of vectors: {num_vectors:,}')
    print(f'Dimension: {dim}')

    # Read all vectors (int8 format) - use int64 for count to avoid overflow
    total_elements = num_vectors * dim
    print(f'Total elements to read: {total_elements:,}')

    data = np.fromfile(f, dtype=np.int8, count=total_elements)
    data = data.reshape(num_vectors, dim)

print(f'Data shape: {data.shape}')
print(f'Data dtype: {data.dtype}')

# Save 100M version
import os
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
    echo "Successfully converted SPACEV dataset"
    ls -lh $OUTPUT_DIR/train_*.npy
else
    echo "Failed (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
