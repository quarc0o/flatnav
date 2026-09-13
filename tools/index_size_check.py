#!/usr/bin/env python3
"""
Measure the on-disk size of a saved flatnav index and verify that saving and
loading is lossless.

Two modes:

  make-legacy   Build float32 / uint8 / int8 indexes with the *current* code,
                save them under --legacy-dir together with the queries used and
                the exact search results they produce. Run this once, before
                changing the serialization format, to freeze reference files.

  check         Build fresh indexes, save them, report the file sizes, then load
                them back and verify that the loaded index returns exactly the
                same results as the in-memory one and still accepts new vectors
                up to its original capacity. Also loads every legacy file from
                --legacy-dir and verifies it against the frozen results.

The metric for the optimization task is the "float32 index file" line printed
by `check`. Everything else printed is a constraint that must keep holding.

Example:
  python tools/index_size_check.py make-legacy \
      --data-dir data/mnist-784-euclidean --legacy-dir data/legacy
  python tools/index_size_check.py check \
      --data-dir data/mnist-784-euclidean --legacy-dir data/legacy
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

import flatnav
from flatnav.data_type import DataType

# Fixed experiment parameters. Capacity is deliberately twice the number of
# vectors inserted: users typically preallocate more room than they end up using.
NUM_VECTORS = 20_000
CAPACITY = 40_000
NUM_EXTRA = 1_000  # vectors added after load() to prove capacity survived
NUM_QUERIES = 1_000
M = 16
EF_CONSTRUCTION = 100
EF_SEARCH = 100
K = 10
BUILD_THREADS = 4

DTYPES = {
    "float32": (DataType.float32, flatnav.index.IndexL2Float),
    "uint8": (DataType.uint8, flatnav.index.IndexL2Uint8),
    "int8": (DataType.int8, flatnav.index.IndexL2Int8),
}


def load_mnist(data_dir):
    train = np.load(os.path.join(data_dir, "mnist-784-euclidean.train.npy"))
    test = np.load(os.path.join(data_dir, "mnist-784-euclidean.test.npy"))
    if len(train) < NUM_VECTORS + NUM_EXTRA:
        sys.exit(f"need at least {NUM_VECTORS + NUM_EXTRA} training vectors")
    return train, test[:NUM_QUERIES]


def cast(arr, dtype_name):
    # MNIST pixel values are integers in [0, 255] stored as float32, so these
    # casts are exact.
    if dtype_name == "float32":
        return np.ascontiguousarray(arr, dtype=np.float32)
    if dtype_name == "uint8":
        return np.ascontiguousarray(arr, dtype=np.uint8)
    if dtype_name == "int8":
        return np.ascontiguousarray(arr.astype(np.int32) - 128, dtype=np.int8)
    raise ValueError(dtype_name)


def build_index(dtype_name, vectors):
    data_type, _ = DTYPES[dtype_name]
    index = flatnav.index.create(
        distance_type="l2",
        dim=vectors.shape[1],
        dataset_size=CAPACITY,
        max_edges_per_node=M,
        index_data_type=data_type,
        verbose=False,
    )
    index.set_num_threads(BUILD_THREADS)
    index.add(data=vectors, ef_construction=EF_CONSTRUCTION)
    return index


def run_queries(index, queries):
    index.set_num_threads(BUILD_THREADS)
    distances, labels = index.search(queries=queries, K=K, ef_search=EF_SEARCH)
    return np.asarray(distances), np.asarray(labels)


def brute_force_recall(vectors, queries, labels):
    v = vectors.astype(np.float32)
    q = queries.astype(np.float32)
    d = (q * q).sum(1)[:, None] - 2.0 * q @ v.T + (v * v).sum(1)[None, :]
    truth = np.argpartition(d, K, axis=1)[:, :K]
    hits = sum(len(set(truth[i]) & set(labels[i])) for i in range(len(q)))
    return hits / (len(q) * K)


def assert_same_results(name, expected, actual):
    exp_d, exp_l = expected
    act_d, act_l = actual
    if not np.array_equal(exp_l, act_l):
        diff = int((exp_l != act_l).sum())
        sys.exit(f"FAIL [{name}]: {diff} of {exp_l.size} result labels differ")
    if not np.array_equal(exp_d, act_d):
        sys.exit(f"FAIL [{name}]: result distances differ (max abs diff "
                 f"{np.abs(exp_d - act_d).max():.6g})")


def assert_capacity_survived(name, index, extra):
    try:
        index.add(data=extra, ef_construction=EF_CONSTRUCTION)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"FAIL [{name}]: adding {len(extra)} vectors after load raised: {e}")


def make_legacy(args):
    train, queries = load_mnist(args.data_dir)
    os.makedirs(args.legacy_dir, exist_ok=True)
    np.save(os.path.join(args.legacy_dir, "queries.npy"), queries)
    with open(os.path.join(args.legacy_dir, "baseline_sizes.txt"), "w") as f:
        for dtype_name in DTYPES:
            vectors = cast(train[:NUM_VECTORS], dtype_name)
            q = cast(queries, dtype_name)
            t0 = time.time()
            index = build_index(dtype_name, vectors)
            path = os.path.join(args.legacy_dir, f"legacy_{dtype_name}.index")
            index.save(path)
            d, l = run_queries(index, q)
            np.save(os.path.join(args.legacy_dir, f"legacy_{dtype_name}_dists.npy"), d)
            np.save(os.path.join(args.legacy_dir, f"legacy_{dtype_name}_labels.npy"), l)
            size = os.path.getsize(path)
            print(f"{dtype_name}: built in {time.time() - t0:.1f}s, "
                  f"saved {size} bytes -> {path}")
            f.write(f"{dtype_name} {size}\n")
    print("legacy files written to", args.legacy_dir)


def check(args):
    train, queries = load_mnist(args.data_dir)
    extra_src = train[NUM_VECTORS:NUM_VECTORS + NUM_EXTRA]
    tmpdir = tempfile.mkdtemp(prefix="flatnav-size-")
    sizes = {}

    for dtype_name, (_, cls) in DTYPES.items():
        vectors = cast(train[:NUM_VECTORS], dtype_name)
        q = cast(queries, dtype_name)
        extra = cast(extra_src, dtype_name)

        index = build_index(dtype_name, vectors)
        expected = run_queries(index, q)
        if dtype_name == "float32":
            print(f"info float32: recall@{K} vs brute force on first 200 queries: "
                  f"{brute_force_recall(vectors, q[:200], expected[1][:200]):.3f}")

        path = os.path.join(tmpdir, f"{dtype_name}.index")
        t0 = time.time()
        index.save(path)
        save_s = time.time() - t0
        sizes[dtype_name] = os.path.getsize(path)

        t0 = time.time()
        loaded = cls.load_index(path)
        load_s = time.time() - t0
        assert_same_results(f"{dtype_name} save/load round-trip", expected,
                            run_queries(loaded, q))
        assert_capacity_survived(f"{dtype_name} save/load round-trip", loaded, extra)
        print(f"OK  {dtype_name}: round-trip lossless, capacity preserved "
              f"(save {save_s:.2f}s, load {load_s:.2f}s)")

    if args.legacy_dir and os.path.isdir(args.legacy_dir):
        lq = np.load(os.path.join(args.legacy_dir, "queries.npy"))
        for dtype_name, (_, cls) in DTYPES.items():
            path = os.path.join(args.legacy_dir, f"legacy_{dtype_name}.index")
            if not os.path.exists(path):
                continue
            expected = (
                np.load(os.path.join(args.legacy_dir, f"legacy_{dtype_name}_dists.npy")),
                np.load(os.path.join(args.legacy_dir, f"legacy_{dtype_name}_labels.npy")),
            )
            try:
                loaded = cls.load_index(path)
            except Exception as e:  # noqa: BLE001
                sys.exit(f"FAIL [legacy {dtype_name}]: load_index raised: {e}")
            assert_same_results(f"legacy {dtype_name}", expected,
                                run_queries(loaded, cast(lq, dtype_name)))
            assert_capacity_survived(f"legacy {dtype_name}", loaded,
                                     cast(extra_src, dtype_name))
            print(f"OK  legacy {dtype_name}: loads and matches frozen results")
    else:
        print("no legacy dir found, skipping backward-compatibility check")

    print()
    print(f"float32 index file: {sizes['float32']} bytes   <-- metric")
    print(f"uint8   index file: {sizes['uint8']} bytes")
    print(f"int8    index file: {sizes['int8']} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("make-legacy", "check"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", default="data/mnist-784-euclidean")
        p.add_argument("--legacy-dir", default="data/legacy")
    args = parser.parse_args()
    if args.mode == "make-legacy":
        make_legacy(args)
    else:
        check(args)


if __name__ == "__main__":
    main()
