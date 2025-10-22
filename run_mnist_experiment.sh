#!/bin/bash
# run_experiment.sh - Run experiments manually

# Load environment
module load Python/3.11.3-GCCcore-12.3.0
source $HOME/code/flatnav/venv/bin/activate

cd $HOME/code/flatnav/experiments

# Example: Run MNIST benchmark for FlatNav
poetry run python run-benchmark.py \
    --dataset $HOME/code/flatnav/data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
    --queries $HOME/code/flatnav/data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
    --gtruth $HOME/code/flatnav/data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
    --dataset-name mnist \
    --index-type flatnav \
    --num-node-links 32 \
    --ef-construction 100 200 \
    --ef-search 100 200 300 \
    --metric l2 \
    --metrics-file $HOME/code/flatnav/metrics/metrics.json

# Run MNIST benchmark for HNSW
poetry run python run-benchmark.py \
    --dataset $HOME/code/flatnav/data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
    --queries $HOME/code/flatnav/data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
    --gtruth $HOME/code/flatnav/data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
    --dataset-name mnist \
    --index-type hnsw \
    --num-node-links 32 \
    --ef-construction 100 200 \
    --ef-search 100 200 300 \
    --metric l2 \
    --metrics-file $HOME/code/flatnav/metrics/metrics.json