# This file contains the benchmarking script configurations for various near neighbor datasets
# The main configurable parameter is the type of index: --index-type (hnsw, flatnav)
# We use dataset-name for the name of the plots.


################################
# Begin Billion-Scale Benchmarks
################################

# Billion-scale benchmark for the Microsoft SpaceV1B dataset
# Reference: https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B

space-v1b-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name msft-spacev-1b \
		--dataset /root/data/SPACEV1B/collection.npy \
		--queries /root/data/SPACEV1B/queries.npy \
		--gtruth /root/data/SPACEV1B/ground_truth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 50 100 \
		--ef-search 100 200 300 500 1000 \
		--num-build-threads 32 \
		--num-search-threads 1 \
		--metric l2


space-v1b-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name msft-spacev-1b \
		--dataset /root/data/SPACEV1B/collection.npy \
		--queries /root/data/SPACEV1B/queries.npy \
		--gtruth /root/data/SPACEV1B/ground_truth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename space-v1b.mtx \
		--num-node-links 32 \
		--ef-construction 30 50 100 \
		--ef-search 100 200 300 500 1000 \
		--num-build-threads 45 \
		--num-search-threads 1 \
		--metric l2

################################
# End Billion-Scale Benchmarks
################################

################################
# Begin 100-Million Scale Benchmarks
# ################################
bigann-100m-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name bigann-100m \
		--dataset ../data/bigann/train_100m.npy \
		--queries ../data/bigann/queries.npy \
		--gtruth ../data/bigann/ground_truth_100m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename bigann-100m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1

bigann-100m-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name bigann-100m \
		--dataset ../data/bigann/train_100m.npy \
		--queries ../data/bigann/queries.npy \
		--gtruth ../data/bigann/ground_truth_100m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1

yandex-deep-100m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name yandex-deep-100m \
		--dataset ../data/deep/train_100m.npy \
		--queries ../data/deep/queries.npy \
		--gtruth ../data/deep/ground_truth_100m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename yandex-deep-100m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


yandex-deep-100m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name yandex-deep-100m \
		--dataset ../data/deep/train_100m.npy \
		--queries ../data/deep/queries.npy \
		--gtruth ../data/deep/ground_truth_100m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


spacev-100m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name spacev-100m \
		--dataset ../data/msspacev/train_100m.npy \
		--queries ../data/msspacev/queries.npy \
		--gtruth ../data/msspacev/ground_truth_100m.npy \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename spacev-100m.mtx \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1

spacev-100m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name spacev-100m \
		--dataset ../data/msspacev/train_100m.npy \
		--queries ../data/msspacev/queries.npy \
		--gtruth ../data/msspacev/ground_truth_100m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


tti-100m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name tti-100m \
		--dataset ../data/text2image/train_100m.npy \
		--queries ../data/text2image/queries.npy \
		--gtruth ../data/text2image/ground_truth_100m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename tti-100m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric angular \
		--num-build-threads 32 \
		--num-search-threads 1

tti-100m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name tti-100m \
		--dataset ../data/text2image/train_100m.npy \
		--queries ../data/text2image/queries.npy \
		--gtruth ../data/text2image/ground_truth_100m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric angular \
		--num-build-threads 32 \
		--num-search-threads 1

################################
# End 100-Million Scale Benchmarks
# # ################################

################################
# Begin 10-Million Scale Benchmarks
# # ################################
bigann-10m-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name bigann-10m \
		--dataset ../data/bigann/train_10m.npy \
		--queries ../data/bigann/queries.npy \
		--gtruth ../data/bigann/ground_truth_10m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename bigann-10m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


bigann-10m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name bigann-10m \
		--dataset ../data/bigann/train_10m.npy \
		--queries ../data/bigann/queries.npy \
		--gtruth ../data/bigann/ground_truth_10m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


yandex-deep-10m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name yandex-deep-10m \
		--dataset ../data/deep/train_10m.npy \
		--queries ../data/deep/queries.npy \
		--gtruth ../data/deep/ground_truth_10m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename yandex-deep-10m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


yandex-deep-10m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name yandex-deep-10m \
		--dataset ../data/deep/train_10m.npy \
		--queries ../data/deep/queries.npy \
		--gtruth ../data/deep/ground_truth_10m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


spacev-10m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name spacev-10m \
		--dataset ../data/msspacev/train_10m.npy \
		--queries ../data/msspacev/queries.npy \
		--gtruth ../data/msspacev/ground_truth_10m.npy \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename spacev-10m.mtx \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


spacev-10m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name spacev-10m \
		--dataset ../data/msspacev/train_10m.npy \
		--queries ../data/msspacev/queries.npy \
		--gtruth ../data/msspacev/ground_truth_10m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


tti-10m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name tti-10m \
		--dataset ../data/text2image/train_10m.npy \
		--queries ../data/text2image/queries.npy \
		--gtruth ../data/text2image/ground_truth_10m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename tti-10m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric angular \
		--num-build-threads 32 \
		--num-search-threads 1

tti-10m--bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name tti-10m \
		--dataset ../data/text2image/train_10m.npy \
		--queries ../data/text2image/queries.npy \
		--gtruth ../data/text2image/ground_truth_10m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric angular \
		--num-build-threads 32 \
		--num-search-threads 1

################################
# End 10-Million Scale Benchmarks
# # ################################

################################
# Begin ANN Benchmarks
# Ref: https://ann-benchmarks.com/
################################

deep-image-bench-flatnav: 
	poetry run python run_benchmark.py \
		--dataset-name deep-image-96-random \
		--dataset ../data/deep-image-96-angular/deep-image-96-angular.train.npy \
		--queries ../data/deep-image-96-angular/deep-image-96-angular.test.npy \
		--gtruth ../data/deep-image-96-angular/deep-image-96-angular.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename deep-image.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
        --num-search-threads 1

deep-image-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name deep-image-96 \
		--dataset ../data/deep-image-96-angular/deep-image-96-angular.train.npy \
		--queries ../data/deep-image-96-angular/deep-image-96-angular.test.npy \
		--gtruth ../data/deep-image-96-angular/deep-image-96-angular.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
        --num-search-threads 1

mnist-bench-flatnav: 
	poetry run python run_benchmark.py \
		--dataset-name mnist-784-random \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename mnist.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 

mnist-bench-flatnav-base: 
	poetry run python run-benchmark.py \
		--dataset-name mnist-784-base \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 

mnist-bench-hnsw: 
	poetry run python run_benchmark.py \
		--dataset-name mnist-784-non-random \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 

gist-bench-flatnav: 
	poetry run python run_benchmark.py \
		--dataset-name gist-pruning \
		--dataset ../data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries ../data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth ../data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename gist.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-percentages 1 2 5 \
		--alpha 1.0 \
		--hub-identification indegree


gist-bench-hnsw: 
	poetry run python run_benchmark.py \
		--dataset-name gist \
		--dataset ../data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries ../data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth ../data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


sift-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name sift-random \
		--dataset ../data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries ../data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth ../data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename sift.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


sift-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name sift \
		--dataset ../data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries ../data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth ../data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1

hubness-test:
	poetry run python hubness.py \
		--datasets normal-1-angular normal-2-angular normal-4-angular normal-8-angular normal-16-angular normal-32-angular normal-64-angular normal-128-angular normal-256-angular normal-1024-angular normal-1536-angular normal-1-euclidean normal-2-euclidean normal-4-euclidean normal-8-euclidean normal-16-euclidean normal-32-euclidean normal-64-euclidean normal-128-euclidean normal-256-euclidean normal-1024-euclidean normal-1536-euclidean \
		--k 100 \
		--ef-construction 100 \
		--ef-search 200 \
		--num-node-links 32 \
		--hubness-scores hubness-scores.json


generate-datasets:
	poetry run python generate-datasets.py \
		--dataset-size 1010000 \
		--num-queries 10000 \
		--dimensions 1 2 4 8 16 32 64 128 1536 \
		--k 100

# Generate hubness scores for each desired dataset
# generate-hubness-scores:
# 	poetry run python generate-hubness-scores.py \
# 		--dataset-names normal-1-angular normal-2-angular normal-4-angular normal-8-angular normal-16-angular normal-32-angular normal-64-angular normal-128-angular normal-256-angular normal-1024-angular normal-1536-angular normal-2048-angular normal-1-euclidean normal-2-euclidean normal-4-euclidean normal-8-euclidean normal-16-euclidean normal-32-euclidean normal-64-euclidean normal-128-euclidean normal-256-euclidean normal-1024-euclidean normal-1536-euclidean normal-2048-euclidean \
# 		--base-path /media/scratch/ \
# 		--k 100 \
# 		--save-file hubness-scores-full.json


glove50-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset ../data/glove-50-angular/glove-50-angular.train.npy \
		--queries ../data/glove-50-angular/glove-50-angular.test.npy \
		--gtruth ../data/glove-50-angular/glove-50-angular.gtruth.npy \
		--dataset-name glove-50-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
        --hnsw-base-layer-filename glove50.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove50-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset ../data/glove-50-angular/glove-50-angular.train.npy \
		--queries ../data/glove-50-angular/glove-50-angular.test.npy \
		--gtruth ../data/glove-50-angular/glove-50-angular.gtruth.npy \
		--dataset-name glove-50-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove100-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset ../data/glove-100-angular/glove-100-angular.train.npy \
		--queries ../data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth ../data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove-100-unpruned \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename glove100.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove100-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset ../data/glove-100-angular/glove-100-angular.train.npy \
		--queries ../data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth ../data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove-100-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove200-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset ../data/glove-200-angular/glove-200-angular.train.npy \
		--queries ../data/glove-200-angular/glove-200-angular.test.npy \
		--gtruth ../data/glove-200-angular/glove-200-angular.gtruth.npy \
		--dataset-name glove-200-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
        --hnsw-base-layer-filename glove25.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove200-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset ../data/glove-200-angular/glove-200-angular.train.npy \
		--queries ../data/glove-200-angular/glove-200-angular.test.npy \
		--gtruth ../data/glove-200-angular/glove-200-angular.gtruth.npy \
		--dataset-name glove-200-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular


nytimes-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset ../data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries ../data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth ../data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256-angular-non-random \
		--index-type flatnav \
		--use-hnsw-base-layer \
        --hnsw-base-layer-filename nytimes.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

nytimes-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset ../data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries ../data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth ../data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

node-access-test:
	poetry run python node-access-patterns.py \
		--datasets mnist-784-euclidean sift-128-euclidean glove-25-angular glove-100-angular \
		--k 100 \
		--metrics l2 l2 angular angular \
		--ef-construction 100 \
		--ef-search 200 \
		--num-node-links 32


################################################################
# Begin ANN Benchmarks - FlatNav Base (without HNSW base layer)
# These targets run FlatNav without using HNSW as the base layer
################################

deep-image-bench-flatnav-base:
	poetry run python run-benchmark.py \
		--dataset-name deep-image-96-base \
		--dataset ../data/deep-image-96-angular/deep-image-96-angular.train.npy \
		--queries ../data/deep-image-96-angular/deep-image-96-angular.test.npy \
		--gtruth ../data/deep-image-96-angular/deep-image-96-angular.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1

gist-bench-flatnav-base:
	poetry run python run-benchmark.py \
		--dataset-name gist-base \
		--dataset ../data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries ../data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth ../data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1

sift-bench-flatnav-base:
	poetry run python run-benchmark.py \
		--dataset-name sift-base \
		--dataset ../data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries ../data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth ../data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1

nytimes-bench-flatnav-base:
	poetry run python run-benchmark.py \
		--dataset ../data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries ../data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth ../data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256-angular-base \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

################################
# Hubness Test targets 
################################################################

		
generate-hubness-scores:
	poetry run python generate-hubness-scores.py \
		--base-path ../data \
		--dataset-names normal-256-angular normal-256-euclidean \
		--k 100 \
		--save-file hubness-scores.json 

# sift-128-euclidean glove-100-angular normal-10-angular normal-10-euclidean normal-256-angular normal-256-euclidean normal-1024-angular normal-1024-euclidean
# l2 angular angular l2 angular l2 angular l2
generate-distributions:
	poetry run python generate-distributions.py \
		--k 100 \
		--ef-construction 100 \
		--ef-search 200 \
		--num-node-links 32

kde-plots:
	poetry run python kde_plots.py

# Run statistical tests for hub nodes connectivity. 
run-hypothesis-test:
	poetry run python statistical_test.py

speed-test:
	poetry run python hubness_speed_test.py --hubness-percentile-threshold 99

plot-speed-test-results:
	poetry run python plot_speed_test_results.py

plot-results:
	poetry run python plot_stats_test_results.py

################################
# Hub Pruning Experiments
################################


hub-pruning-bigann-100m: 
	poetry run python hub_pruning_experiment.py \
		--dataset-name bigann-100m \
		--dataset ../data/bigann/train_100m.npy \
		--queries ../data/bigann/queries.npy \
		--gtruth ../data/bigann/ground_truth_100m.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 100 \
		--ef-search 100 200 300 500 1000 3000 \
		--pruning-percentages 5 \
		--metric l2 \
		--alpha 1.0 \
		--hub-identification indegree \
		--num-build-threads 16 \
		--num-search-threads 1

# Run hub pruning experiment on glove-100-angular
hub-pruning-glove100:
	poetry run python hub_pruning_experiment.py \
		--dataset ../data/glove-100-angular/glove-100-angular.train.npy \
		--queries ../data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth ../data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove-100-angular \
		--metric angular \
		--num-node-links 32 \
		--ef-construction 100 \
		--ef-search 100 200 300 500 1000 3000 \
		--pruning-percentages 5 \
		--alpha 1.0 \
		--hub-identification indegree \
		--use-hnsw-base-layer \
		--num-build-threads 16 \
		--num-search-threads 1

# Run hub pruning experiment on sift-128-euclidean
hub-pruning-sift:
	poetry run python hub_pruning_experiment.py \
		--dataset ../data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries ../data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth ../data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--dataset-name sift-128 \
		--metric l2 \
		--num-node-links 32 \
		--ef-construction 100 \
		--ef-search 100 200 300 500 1000 3000 \
		--pruning-percentages 5 \
		--alpha 1.0 \
		--hub-identification indegree \
		--use-hnsw-base-layer \
		--num-build-threads 16 \
		--num-search-threads 1

hub-pruning-nytimes:
	poetry run python hub_pruning_experiment.py \
		--dataset ../data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries ../data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth ../data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256 \
		--metric angular \
		--num-node-links 32 \
		--ef-construction 100 \
		--ef-search 100 200 300 500 1000 3000 \
		--pruning-percentages 5 \
		--alpha 1.0 \
		--hub-identification indegree \
		--use-hnsw-base-layer \
		--num-build-threads 16 \
		--num-search-threads 1

hub-pruning-gist:
	poetry run python hub_pruning_experiment.py \
		--dataset-name gist-960 \
		--dataset ../data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries ../data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth ../data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--metric l2 \
		--num-node-links 32 \
		--ef-construction 100 \
		--ef-search 100 200 300 500 1000 3000 \
		--pruning-percentages 5 \
		--alpha 1.0 \
		--hub-identification indegree \
		--num-build-threads 16 \
		--num-search-threads 1

hub-pruning-mnist:
	poetry run python hub_pruning_experiment.py \
		--dataset-name mnist-784 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--metric l2 \
		--num-node-links 32 \
		--ef-construction 100 \
		--ef-search 100 200 300 500 1000 3000 \
		--pruning-percentages 5 \
		--alpha 1.0 \
		--hub-identification indegree \
		--num-build-threads 16 \
		--num-search-threads 1

# Run hub pruning experiment on deep-image-96-angular
hub-pruning-deep-image:
	poetry run python hub_pruning_experiment.py \
		--dataset ../data/deep-image-96-angular/deep-image-96-angular.train.npy \
		--queries ../data/deep-image-96-angular/deep-image-96-angular.test.npy \
		--gtruth ../data/deep-image-96-angular/deep-image-96-angular.gtruth.npy \
		--dataset-name deep-image-96 \
		--metric angular \
		--num-node-links 32 \
		--ef-construction 100 \
		--ef-search 100 200 300 500 1000 3000 \
		--pruning-percentages 5 \
		--alpha 1.0 \
		--hub-identification indegree \
		--use-hnsw-base-layer \
		--num-build-threads 16 \
		--num-search-threads 1

# Run hub pruning experiment with access-count based hub identification
hub-pruning-access-count:
	poetry run python hub_pruning_experiment.py \
		--dataset ../data/glove-100-angular/glove-100-angular.train.npy \
		--queries ../data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth ../data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove-100-angular-access \
		--metric angular \
		--num-node-links 32 \
		--ef-construction 200 \
		--ef-search 100 \
		--pruning-percentages 1 2 5 \
		--alpha 1.0 \
		--hub-identification access_count \
		--use-hnsw-base-layer \
		--num-build-threads 16 \
		--num-search-threads 1

# Plot hub pruning results (old format)
plot-hub-pruning:
	poetry run python plot_hub_pruning_results.py \
		--results-file ../metrics/glove-100-angular_hub_pruning_experiment.json \
		--output-dir ../plots

# Plot pruning metrics (new format with pruning percentages)
plot-pruning-metrics:
	poetry run python plot_pruning_metrics.py \
		--metrics-file ../metrics/new_pruning_metrics.json \
		--output-dir ../plots

################################
# End Hub Pruning Experiments
################################

setup: install-flatnav install-hnswlib

install-hnswlib:
	if [ ! -d "hnswlib-original" ]; then \
		git clone https://github.com/BlaiseMuhirwa/hnswlib-original.git; \
	fi
	cd hnswlib-original/python_bindings && \
	poetry install --no-root && \
	poetry run python setup.py bdist_wheel 
	poetry add hnswlib-original/python_bindings/dist/*.whl

cleanup:
	rm -rf hnswlib-original 

# If passed an invalid argument, print help message
%:
	@echo "Invalid argument: $@"
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  setup: install all dependencies including flatnav"
	@echo "  install-flatnav: install flatnav"
	@echo "  install-hnswlib: install hnswlib"
	@echo "  cleanup: remove hnswlib-original"
	@echo "  generate-wheel: generate wheel for flatnav"
	@echo "  yandex-deep-bench: run yandex-deep benchmark"
	@echo "  sift-bench: run sift benchmark"
	
