#!/bin/bash
#SBATCH --job-name=bigann-100m
#SBATCH --partition=GPUQ
#SBATCH --time=24:00:00
#SBATCH --mem=1TB       
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/bigann-100m-%j.out
#SBATCH --error=logs/bigann-100m-%j.err
#SBATCH --gres=gpu:0 

module load Python/3.11.3-GCCcore-12.3.0
source $HOME/code/flatnav/venv/bin/activate

cd $HOME/code/flatnav

# Download BigANN dataset (this will take a while)
echo "Downloading BigANN dataset..."
./bin/download_bigann_datasets.sh bigann

# The script converts files differently than ANN benchmarks
# Check actual file paths after download
echo "Checking downloaded files:"
ls -la data/bigann/

# Adjust paths based on what the conversion creates
cd experiments

# FlatNav with HNSW base layer
poetry run python run-benchmark.py \
    --dataset ../data/bigann/train_100m.npy \
    --queries ../data/bigann/queries.npy \
    --gtruth ../data/bigann/ground_truth_100m.npy \
    --dataset-name bigann-100m \
    --index-type flatnav \
    --use-hnsw-base-layer \
    --hnsw-base-layer-filename bigann-100m.mtx \
    --num-node-links 32 \
    --ef-construction 50 100 200 \
    --ef-search 100 200 300 500 1000 3000 \
    --metric l2 \
    --num-build-threads 32 \
    --num-search-threads 1 \
    --metrics-file $HOME/code/flatnav/metrics/bigann_metrics.json

# HNSW (note: second one should be index-type hnsw, not flatnav)
poetry run python run-benchmark.py \
    --dataset ../data/bigann/train_100m.npy \
    --queries ../data/bigann/queries.npy \
    --gtruth ../data/bigann/ground_truth_100m.npy \
    --dataset-name bigann-100m \
    --index-type hnsw \
    --num-node-links 32 \
    --ef-construction 50 100 200 \
    --ef-search 100 200 300 500 1000 3000 \
    --metric l2 \
    --num-build-threads 32 \
    --num-search-threads 1 \
    --metrics-file $HOME/code/flatnav/metrics/bigann_metrics.json

# Don't delete BigANN - it's too big to re-download
echo "BigANN 100M experiments completed!"