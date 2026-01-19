#!/bin/bash
#SBATCH --job-name=convert-bigann
#SBATCH --time=4:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/flatnav-%j.out
#SBATCH --error=logs/flatnav-%j.err

echo "=========================================="
echo "BigANN Dataset Conversion Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 180G"
echo "Start time: $(date)"
echo "=========================================="

# Load environment
module purge
module load Python/3.11.3-GCCcore-12.3.0

# Activate virtual environment
source $HOME/code/flatnav/venv/bin/activate

# Go to flatnav directory
cd $HOME/code/flatnav

# Convert the deep dataset
echo ""
echo "==========================================="
echo "Starting Deep Dataset Conversion"
echo "==========================================="

# Check if input files exist
QUERY_FILE="data/deep/deep_query.public.10K.fbin"
BASE_FILE="data/deep/deep_base.1B.fbin"

echo ""
echo "Checking for input files..."
if [ -f "$QUERY_FILE" ]; then
    echo "✓ Query file found: $QUERY_FILE"
    ls -lh "$QUERY_FILE"
else
    echo "✗ Query file NOT found: $QUERY_FILE"
    echo "Current directory: $(pwd)"
    echo "Contents of data/deep/:"
    ls -lh data/deep/ 2>/dev/null || echo "data/deep/ directory does not exist"
    exit 1
fi

if [ -f "$BASE_FILE" ]; then
    echo "✓ Base file found: $BASE_FILE"
    ls -lh "$BASE_FILE"
else
    echo "✗ Base file NOT found: $BASE_FILE"
    echo "Current directory: $(pwd)"
    echo "Contents of data/deep/:"
    ls -lh data/deep/ 2>/dev/null || echo "data/deep/ directory does not exist"
    exit 1
fi

# Convert queries
echo ""
echo "-------------------------------------------"
echo "Step 1/2: Converting queries (10K vectors)..."
echo "Input:  $QUERY_FILE"
echo "Output: data/deep/queries.npy"
echo "Start time: $(date)"
echo "-------------------------------------------"

python convert_bigann_datasets.py "$QUERY_FILE" queries
QUERY_EXIT_CODE=$?

if [ $QUERY_EXIT_CODE -eq 0 ]; then
    echo "✓ Query conversion completed successfully"
    if [ -f "data/deep/queries.npy" ]; then
        echo "Created file:"
        ls -lh data/deep/queries.npy
    else
        echo "⚠ Warning: queries.npy not found despite successful exit code"
    fi
else
    echo "✗ Query conversion failed (exit code: $QUERY_EXIT_CODE)"
    exit $QUERY_EXIT_CODE
fi

# Convert base dataset
echo ""
echo "-------------------------------------------"
echo "Step 2/2: Converting base dataset (1B vectors)..."
echo "Input:  $BASE_FILE"
echo "Output: data/deep/train_100m.npy, data/deep/train_10m.npy"
echo "Start time: $(date)"
echo "-------------------------------------------"

python convert_bigann_datasets.py "$BASE_FILE" train
BASE_EXIT_CODE=$?

if [ $BASE_EXIT_CODE -eq 0 ]; then
    echo "✓ Base dataset conversion completed successfully"
    echo "Created files:"
    ls -lh data/deep/train_*.npy 2>/dev/null || echo "⚠ Warning: train files not found"
else
    echo "✗ Base dataset conversion failed (exit code: $BASE_EXIT_CODE)"
    exit $BASE_EXIT_CODE
fi

EXIT_CODE=0

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Successfully converted Deep dataset"
    echo ""
    echo "All created files:"
    ls -lh data/deep/*.npy 2>/dev/null || echo "Warning: no .npy files found in data/deep/"
    echo ""
    echo "Summary:"
    echo "  - queries.npy: 10K query vectors"
    echo "  - train_10m.npy: First 10M vectors from base"
    echo "  - train_100m.npy: First 100M vectors from base"
else
    echo "✗ Conversion failed (exit code: $EXIT_CODE)"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
