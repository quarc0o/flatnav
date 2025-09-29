#!/bin/bash
#SBATCH --job-name=nytimes
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/flatnav-%j.out
#SBATCH --error=logs/flatnav-%j.err

# Dataset configuration - CHANGE THESE FOR EACH DATASET
DATASET_NAME="nytimes"
DATASET_FULL="nytimes-256-angular"
METRIC="angular"

# Load environment
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/code/flatnav/venv/bin/activate

cd $HOME/code/flatnav

# Download dataset
echo "Downloading $DATASET_NAME dataset..."
./bin/download_ann_benchmarks_datasets.sh $DATASET_FULL

# Run experiments
cd experiments

# FlatNav benchmark
poetry run python run-benchmark.py \
    --dataset ../data/$DATASET_FULL/$DATASET_FULL.train.npy \
    --queries ../data/$DATASET_FULL/$DATASET_FULL.test.npy \
    --gtruth ../data/$DATASET_FULL/$DATASET_FULL.gtruth.npy \
    --dataset-name $DATASET_NAME \
    --index-type flatnav \
    --use-hnsw-base-layer \
    --hnsw-base-layer-filename nytimes.mtx \
    --num-node-links 32 \
    --ef-construction 30 40 50 100 200 300 \
	--ef-search 100 200 300 500 1000 3000 \
    --metric $METRIC \
    --metrics-file $HOME/code/flatnav/metrics/metrics.json

# HNSW benchmark
poetry run python run-benchmark.py \
    --dataset ../data/$DATASET_FULL/$DATASET_FULL.train.npy \
    --queries ../data/$DATASET_FULL/$DATASET_FULL.test.npy \
    --gtruth ../data/$DATASET_FULL/$DATASET_FULL.gtruth.npy \
    --dataset-name $DATASET_NAME \
    --index-type hnsw \
    --num-node-links 32 \
    --ef-construction 30 40 50 100 200 300 \
	--ef-search 100 200 300 500 1000 3000 \
    --metric $METRIC \
    --metrics-file $HOME/code/flatnav/metrics/metrics.json

# Clean up dataset to save space
echo "Cleaning up dataset..."
rm -rf ../data/$DATASET_FULL

echo "$DATASET_NAME experiments completed!"