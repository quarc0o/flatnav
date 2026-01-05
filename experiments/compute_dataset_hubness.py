#!/usr/bin/env python3
"""
Compute the intrinsic hubness/skewness of a dataset using brute-force exact k-NN.

Hubness is a phenomenon in high-dimensional data where some points (hubs) appear
frequently as nearest neighbors of many other points, while others (anti-hubs)
rarely appear. This script computes the N_k distribution and its skewness.

The N_k score for a point is the number of times it appears in the k-nearest
neighbors of all other points.

Usage:
    python compute_dataset_hubness.py \
        --dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
        --k 10 \
        --output ../metrics/hubness_mnist.json

    # For large datasets, use sampling
    python compute_dataset_hubness.py \
        --dataset ../data/gist-960-euclidean/gist-960-euclidean.train.npy \
        --k 10 \
        --sample-size 10000 \
        --output ../metrics/hubness_gist.json
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist


def load_dataset(path: str, sample_size: Optional[int] = None) -> np.ndarray:
    """Load dataset from numpy file, optionally sampling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path).astype(np.float32)
    logging.info(f"Loaded dataset: {data.shape}")

    if sample_size and sample_size < len(data):
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(data), sample_size, replace=False)
        data = data[indices]
        logging.info(f"Sampled to {len(data)} points")

    return data


def compute_knn_bruteforce_batched(
    data: np.ndarray,
    k: int,
    metric: str = "euclidean",
    batch_size: int = 1000,
) -> np.ndarray:
    """
    Compute exact k-NN for all points using brute force, with batching for memory efficiency.

    Args:
        data: Dataset array of shape (n, d)
        k: Number of nearest neighbors
        metric: Distance metric ('euclidean' or 'cosine')
        batch_size: Number of query points per batch

    Returns:
        Array of shape (n, k) containing indices of k nearest neighbors for each point
    """
    n = len(data)
    knn_indices = np.zeros((n, k), dtype=np.int32)

    logging.info(f"Computing {k}-NN for {n} points using brute force...")
    start_time = time.time()

    num_batches = (n + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, n)
        batch_queries = data[batch_start:batch_end]

        # Compute distances from batch to all points
        distances = cdist(batch_queries, data, metric=metric)

        # For each query, find k+1 nearest (including self), then exclude self
        for i, dist_row in enumerate(distances):
            global_idx = batch_start + i
            # Get indices of k+1 smallest distances
            nearest_indices = np.argpartition(dist_row, k + 1)[:k + 1]
            # Sort them by distance
            nearest_indices = nearest_indices[np.argsort(dist_row[nearest_indices])]
            # Exclude self (first one should be self with distance 0)
            nearest_indices = nearest_indices[nearest_indices != global_idx][:k]
            knn_indices[global_idx] = nearest_indices

        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            progress = (batch_idx + 1) / num_batches * 100
            logging.info(f"  Progress: {progress:.1f}% ({batch_end}/{n}), elapsed: {elapsed:.1f}s")

    total_time = time.time() - start_time
    logging.info(f"k-NN computation completed in {total_time:.1f}s")

    return knn_indices


def compute_nk_scores(knn_indices: np.ndarray) -> np.ndarray:
    """
    Compute N_k scores (occurrence counts) for each point.

    N_k(x) = number of times point x appears in the k-NN lists of all other points.

    Args:
        knn_indices: Array of shape (n, k) with k-NN indices

    Returns:
        Array of shape (n,) with N_k scores
    """
    n = len(knn_indices)
    nk_scores = np.zeros(n, dtype=np.int32)

    # Count occurrences
    for neighbor_idx in knn_indices.flatten():
        nk_scores[neighbor_idx] += 1

    return nk_scores


def compute_hubness_statistics(nk_scores: np.ndarray, k: int) -> Dict:
    """
    Compute hubness statistics from N_k scores.

    Args:
        nk_scores: Array of N_k scores for each point
        k: The k value used for k-NN

    Returns:
        Dictionary with hubness statistics
    """
    n = len(nk_scores)

    # Basic statistics
    mean_nk = np.mean(nk_scores)
    std_nk = np.std(nk_scores)
    median_nk = np.median(nk_scores)
    min_nk = np.min(nk_scores)
    max_nk = np.max(nk_scores)

    # Skewness - the key hubness measure
    # Positive skewness indicates hubness (some points appear very frequently)
    skewness = stats.skew(nk_scores)
    kurtosis = stats.kurtosis(nk_scores)

    # Hub and anti-hub counts
    # Hubs: points with N_k > mean + 2*std
    hub_threshold = mean_nk + 2 * std_nk
    num_hubs = np.sum(nk_scores > hub_threshold)

    # Anti-hubs: points that never appear as neighbors (N_k = 0)
    num_antihubs = np.sum(nk_scores == 0)

    # Robin Hood index (concentration measure)
    # Measures inequality in the distribution
    sorted_scores = np.sort(nk_scores)
    cumsum = np.cumsum(sorted_scores)
    robin_hood = 1 - 2 * np.sum(cumsum) / (n * np.sum(nk_scores))

    # Gini coefficient
    gini = 2 * np.sum((np.arange(1, n + 1) - 0.5) * sorted_scores) / (n * np.sum(sorted_scores)) - 1

    # Percentiles
    percentiles = {
        f"p{p}": float(np.percentile(nk_scores, p))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }

    # Distribution (for plotting)
    unique_scores, counts = np.unique(nk_scores, return_counts=True)
    distribution = {
        int(score): float(count / n * 100)
        for score, count in zip(unique_scores, counts)
    }

    return {
        "k": k,
        "n_points": n,
        "mean_nk": float(mean_nk),
        "std_nk": float(std_nk),
        "median_nk": float(median_nk),
        "min_nk": int(min_nk),
        "max_nk": int(max_nk),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "hub_threshold": float(hub_threshold),
        "num_hubs": int(num_hubs),
        "hub_percentage": float(num_hubs / n * 100),
        "num_antihubs": int(num_antihubs),
        "antihub_percentage": float(num_antihubs / n * 100),
        "robin_hood_index": float(robin_hood),
        "gini_coefficient": float(gini),
        "percentiles": percentiles,
        "distribution": distribution,
    }


def print_hubness_report(stats: Dict, dataset_name: str) -> None:
    """Print a formatted hubness report."""
    print("\n" + "=" * 70)
    print(f"HUBNESS ANALYSIS: {dataset_name}")
    print("=" * 70)
    print(f"Dataset size: {stats['n_points']} points")
    print(f"k value: {stats['k']}")
    print("-" * 70)

    print("\nN_k SCORE STATISTICS (occurrence counts):")
    print(f"  Mean:     {stats['mean_nk']:.2f}")
    print(f"  Std Dev:  {stats['std_nk']:.2f}")
    print(f"  Median:   {stats['median_nk']:.1f}")
    print(f"  Min:      {stats['min_nk']}")
    print(f"  Max:      {stats['max_nk']}")

    print("\nHUBNESS MEASURES:")
    print(f"  Skewness:           {stats['skewness']:.4f}  (>0 indicates hubness)")
    print(f"  Kurtosis:           {stats['kurtosis']:.4f}")
    print(f"  Gini coefficient:   {stats['gini_coefficient']:.4f}")
    print(f"  Robin Hood index:   {stats['robin_hood_index']:.4f}")

    print("\nHUBS AND ANTI-HUBS:")
    print(f"  Hub threshold:      N_k > {stats['hub_threshold']:.1f}")
    print(f"  Number of hubs:     {stats['num_hubs']} ({stats['hub_percentage']:.2f}%)")
    print(f"  Number of anti-hubs (N_k=0): {stats['num_antihubs']} ({stats['antihub_percentage']:.2f}%)")

    print("\nPERCENTILES:")
    for key, value in stats['percentiles'].items():
        print(f"  {key}: {value:.1f}")

    print("=" * 70 + "\n")


def save_results(
    stats: Dict,
    dataset_name: str,
    output_path: str,
    nk_scores: Optional[np.ndarray] = None,
) -> None:
    """Save hubness statistics to JSON file."""
    result = {
        "dataset_name": dataset_name,
        "statistics": stats,
    }

    # Optionally include raw scores
    if nk_scores is not None:
        result["nk_scores"] = nk_scores.tolist()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logging.info(f"Results saved to {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute intrinsic hubness/skewness of a dataset using brute-force k-NN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute hubness for MNIST
  python compute_dataset_hubness.py \\
    --dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \\
    --k 10 \\
    --output ../metrics/hubness_mnist.json

  # With sampling for large datasets
  python compute_dataset_hubness.py \\
    --dataset ../data/gist-960-euclidean/gist-960-euclidean.train.npy \\
    --k 10 \\
    --sample-size 10000 \\
    --output ../metrics/hubness_gist_sampled.json

  # Multiple k values
  python compute_dataset_hubness.py \\
    --dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \\
    --k 5 10 20 50 \\
    --output ../metrics/hubness_mnist_multi_k.json
        """
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset (.npy file)"
    )

    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Name for the dataset (default: inferred from filename)"
    )

    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[10],
        help="k value(s) for k-NN (default: 10)"
    )

    parser.add_argument(
        "--metric",
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Distance metric (default: euclidean)"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for large datasets (default: use all data)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for distance computation (default: 1000)"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path"
    )

    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Save raw N_k scores in output (increases file size)"
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_arguments()

    # Infer dataset name if not provided
    dataset_name = args.dataset_name
    if not dataset_name:
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
        # Clean up common suffixes
        for suffix in [".train", ".test", "-euclidean", "-angular"]:
            dataset_name = dataset_name.replace(suffix, "")

    logging.info(f"Dataset: {dataset_name}")

    # Load data
    data = load_dataset(args.dataset, args.sample_size)

    all_results = []

    for k in args.k:
        logging.info(f"\n{'='*50}")
        logging.info(f"Computing hubness for k={k}")
        logging.info(f"{'='*50}")

        # Compute k-NN
        knn_indices = compute_knn_bruteforce_batched(
            data,
            k=k,
            metric=args.metric,
            batch_size=args.batch_size,
        )

        # Compute N_k scores
        logging.info("Computing N_k scores...")
        nk_scores = compute_nk_scores(knn_indices)

        # Compute statistics
        stats = compute_hubness_statistics(nk_scores, k)

        # Print report
        print_hubness_report(stats, f"{dataset_name} (k={k})")

        all_results.append({
            "k": k,
            "statistics": stats,
            "nk_scores": nk_scores.tolist() if args.save_scores else None,
        })

    # Save results
    output_data = {
        "dataset_name": dataset_name,
        "dataset_path": args.dataset,
        "metric": args.metric,
        "sample_size": args.sample_size,
        "n_points": len(data),
        "dimensionality": data.shape[1],
        "results": all_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    logging.info(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
