# This file contains the benchmarking script configurations for various near neighbor datasets
# The main configurable parameter is the type of index: --index-type (hnsw, flatnav)
# We use dataset-name for the name of the plots.

################################
# Begin 100-Million Scale Benchmarks
# ################################
bigann-100m-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name bigann-100m \
		--dataset /root/data/bigann/train_100m.npy \
		--queries /root/data/bigann/queries.npy \
		--gtruth /root/data/bigann/ground_truth_100m.npy \
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
		--dataset /root/data/bigann/train_100m.npy \
		--queries /root/data/bigann/queries.npy \
		--gtruth /root/data/bigann/ground_truth_100m.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1

yandex-deep-100m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name yandex-deep-100m \
		--dataset /root/data/deep/train_100m.npy \
		--queries /root/data/deep/queries.npy \
		--gtruth /root/data/deep/ground_truth_100m.npy \
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
		--dataset /root/data/deep/train_100m.npy \
		--queries /root/data/deep/queries.npy \
		--gtruth /root/data/deep/ground_truth_100m.npy \
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
		--dataset /root/data/msspacev/train_100m.npy \
		--queries /root/data/msspacev/queries.npy \
		--gtruth /root/data/msspacev/ground_truth_100m.npy \
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
		--dataset /root/data/msspacev/train_100m.npy \
		--queries /root/data/msspacev/queries.npy \
		--gtruth /root/data/msspacev/ground_truth_100m.npy \
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
		--dataset /root/data/text2image/train_100m.npy \
		--queries /root/data/text2image/queries.npy \
		--gtruth /root/data/text2image/ground_truth_100m.npy \
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
		--dataset /root/data/text2image/train_100m.npy \
		--queries /root/data/text2image/queries.npy \
		--gtruth /root/data/text2image/ground_truth_100m.npy \
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
		--dataset /root/data/bigann/train_10m.npy \
		--queries /root/data/bigann/queries.npy \
		--gtruth /root/data/bigann/ground_truth_10m.npy \
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
		--dataset /root/data/bigann/train_10m.npy \
		--queries /root/data/bigann/queries.npy \
		--gtruth /root/data/bigann/ground_truth_10m.npy \
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
		--dataset /root/data/deep/train_10m.npy \
		--queries /root/data/deep/queries.npy \
		--gtruth /root/data/deep/ground_truth_10m.npy \
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
		--dataset /root/data/deep/train_10m.npy \
		--queries /root/data/deep/queries.npy \
		--gtruth /root/data/deep/ground_truth_10m.npy \
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
		--dataset /root/data/msspacev/train_10m.npy \
		--queries /root/data/msspacev/queries.npy \
		--gtruth /root/data/msspacev/ground_truth_10m.npy \
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
		--dataset /root/data/msspacev/train_10m.npy \
		--queries /root/data/msspacev/queries.npy \
		--gtruth /root/data/msspacev/ground_truth_10m.npy \
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
		--dataset /root/data/text2image/train_10m.npy \
		--queries /root/data/text2image/queries.npy \
		--gtruth /root/data/text2image/ground_truth_10m.npy \
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
		--dataset /root/data/text2image/train_10m.npy \
		--queries /root/data/text2image/queries.npy \
		--gtruth /root/data/text2image/ground_truth_10m.npy \
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
	poetry run python run-benchmark.py \
		--dataset-name deep-image-96 \
		--dataset /root/data/deep-image-96-angular/deep-image-96-angular.train.npy \
		--queries /root/data/deep-image-96-angular/deep-image-96-angular.test.npy \
		--gtruth /root/data/deep-image-96-angular/deep-image-96-angular.gtruth.npy \
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
		--dataset /root/data/deep-image-96-angular/deep-image-96-angular.train.npy \
		--queries /root/data/deep-image-96-angular/deep-image-96-angular.test.npy \
		--gtruth /root/data/deep-image-96-angular/deep-image-96-angular.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
        --num-search-threads 1

mnist-bench-flatnav-query-aware-80-32:
	python run-benchmark.py \
		--dataset-name mnist-784-query-aware-80-32 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy hnsw \
		--query-aware-pruning \
		--query-pruning-keep-ratio 0.8 \
		--num-training-queries 10000

mnist-bench-flatnav-query-aware-80-16:
	python run-benchmark.py \
		--dataset-name mnist-784-query-aware-80-16 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 16 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy hnsw \
		--query-aware-pruning \
		--query-pruning-keep-ratio 0.8 \
		--num-training-queries 10000

mnist-bench-flatnav-query-aware-80-16-hnsw-base:
	python run-benchmark.py \
		--dataset-name mnist-784-query-aware-80-16 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 16 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename mnist.mtx \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy hnsw \
		--query-aware-pruning \
		--query-pruning-keep-ratio 0.8 \
		--num-training-queries 10000


mnist-bench-flatnav-query-aware-80-16-hnsw-base:
	python run-benchmark.py \
		--dataset-name mnist-784-query-aware-80-16 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 16 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename mnist.mtx \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy hnsw \
		--query-aware-pruning \
		--query-pruning-keep-ratio 0.8 \
		--num-training-queries 10000

mnist-bench-flatnav-query-aware-60:
	python run-benchmark.py \
		--dataset-name mnist-784-query-aware-60 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy hnsw \
		--query-aware-pruning \
		--query-pruning-keep-ratio 0.6 \
		--num-training-queries 10000

mnist-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name mnist-flatnav-base \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 16 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000  \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy hnsw

mnist-bench-flatnav-link-16: 
	poetry run python run-benchmark.py \
		--dataset-name mnist-flatnav-link-16 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 16 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000  \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy hnsw

mnist-bench-flatnav-link-64: 
	poetry run python run-benchmark.py \
		--dataset-name mnist-flatnav-link-64 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 64 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000  \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy hnsw

mnist-bench-flatnav-rng:
	python run-benchmark.py \
		--dataset-name mnist-rng \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 16 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy rng

mnist-bench-flatnav-rng-link-16:
	python run-benchmark.py \
		--dataset-name mnist-rng-link-16 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 16 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy rng

mnist-bench-flatnav-rng-link-64:
	python run-benchmark.py \
		--dataset-name mnist-rng-link-64 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 64 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy rng

mnist-bench-flatnav-ssg-110:
	python run-benchmark.py \
		--dataset-name mnist-784-ssg-110 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy ssg \
		--angle-threshold 110.0

mnist-bench-flatnav-ssg-110-link-16:
	python run-benchmark.py \
		--dataset-name mnist-784-ssg-110-link-16 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 16 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy ssg \
		--angle-threshold 110.0

mnist-bench-flatnav-ssg-110-link-64:
	python run-benchmark.py \
		--dataset-name mnist-784-ssg-110-link-64 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 64 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy ssg \
		--angle-threshold 110.0

mnist-bench-flatnav-ssg-30:
	python run-benchmark.py \
		--dataset-name mnist-784-ssg-30 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy ssg \
		--angle-threshold 30.0

mnist-bench-flatnav-ssg-10:
	python run-benchmark.py \
		--dataset-name mnist-784-ssg-10 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy ssg \
		--angle-threshold 10.0

mnist-bench-flatnav-alpha:
	poetry run python run-benchmark.py \
		--dataset-name mnist-784-alpha-2.0 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy alpha_diversity \
		--alpha 2.0

mnist-bench-flatnav-alpha-1.5:
	poetry run python run-benchmark.py \
		--dataset-name mnist-784-alpha-1.5 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy alpha_diversity \
		--alpha 1.5

mnist-bench-flatnav-alpha-0.5:
	poetry run python run-benchmark.py \
		--dataset-name mnist-784-alpha-0.5 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy alpha_diversity \
		--alpha 0.5

mnist-bench-flatnav-alpha-0.1:
	poetry run python run-benchmark.py \
		--dataset-name mnist-784-alpha-0.1 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy alpha_diversity \
		--alpha 0.1

mnist-bench-flatnav-alpha-0.005:
	poetry run python run-benchmark.py \
		--dataset-name mnist-784-alpha-0.005 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy alpha_diversity \
		--alpha 0.005

mnist-bench-flatnav-alpha-0.0005:
	poetry run python run-benchmark.py \
		--dataset-name mnist-784-alpha-0.0005 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy alpha_diversity \
		--alpha 0.0005


mnist-bench-flatnav-alpha-0.000000000000005:
	poetry run python run-benchmark.py \
		--dataset-name mnist-784-alpha-0.000000000000005 \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy alpha_diversity \
		--alpha 0.000000000000005

mnist-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name mnist-784 \
		--dataset /root/data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries /root/data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth /root/data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 

gist-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name gist \
		--dataset /root/data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries /root/data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth /root/data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename gist.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


gist-bench-flatnav-query-aware-80-32-hnsw-base:
	python run-benchmark.py \
		--dataset-name gist-query-aware-80-32 \
		--dataset /root/data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries /root/data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth /root/data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename gist.mtx \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy hnsw \
		--query-aware-pruning \
		--query-pruning-keep-ratio 0.8 \
		--num-training-queries 10000

gist-bench-flatnav-query-aware-80-16-hnsw-base:
	python run-benchmark.py \
		--dataset-name gist-query-aware-80-16 \
		--dataset /root/data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries /root/data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth /root/data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 16 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename gist.mtx \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--pruning-strategy hnsw \
		--query-aware-pruning \
		--query-pruning-keep-ratio 0.8 \
		--num-training-queries 10000

gist-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name gist \
		--dataset /root/data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries /root/data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth /root/data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


sift-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name sift \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


sift-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name sift \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1


glove25-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-25-angular/glove-25-angular.train.npy \
		--queries /root/data/glove-25-angular/glove-25-angular.test.npy \
		--gtruth /root/data/glove-25-angular/glove-25-angular.gtruth.npy \
		--dataset-name glove-25-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
        --hnsw-base-layer-filename glove25.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove25-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-25-angular/glove-25-angular.train.npy \
		--queries /root/data/glove-25-angular/glove-25-angular.test.npy \
		--gtruth /root/data/glove-25-angular/glove-25-angular.gtruth.npy \
		--dataset-name glove-25-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove50-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-50-angular/glove-50-angular.train.npy \
		--queries /root/data/glove-50-angular/glove-50-angular.test.npy \
		--gtruth /root/data/glove-50-angular/glove-50-angular.gtruth.npy \
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
		--dataset /root/data/glove-50-angular/glove-50-angular.train.npy \
		--queries /root/data/glove-50-angular/glove-50-angular.test.npy \
		--gtruth /root/data/glove-50-angular/glove-50-angular.gtruth.npy \
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
		--dataset /root/data/glove-100-angular/glove-100-angular.train.npy \
		--queries /root/data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth /root/data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove-100-angular \
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
		--dataset /root/data/glove-100-angular/glove-100-angular.train.npy \
		--queries /root/data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth /root/data/glove-100-angular/glove-100-angular.gtruth.npy \
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
		--dataset /root/data/glove-200-angular/glove-200-angular.train.npy \
		--queries /root/data/glove-200-angular/glove-200-angular.test.npy \
		--gtruth /root/data/glove-200-angular/glove-200-angular.gtruth.npy \
		--dataset-name glove-200-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename glove200.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove200-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-200-angular/glove-200-angular.train.npy \
		--queries /root/data/glove-200-angular/glove-200-angular.test.npy \
		--gtruth /root/data/glove-200-angular/glove-200-angular.gtruth.npy \
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
		--dataset /root/data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries /root/data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth /root/data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256-angular \
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
		--dataset /root/data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries /root/data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth /root/data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

################################
# End ANN Benchmarks
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
	
