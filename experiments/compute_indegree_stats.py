#!/usr/bin/env python3
"""
Compute in-degree distribution and hub statistics for FlatNav indices.

This script builds FlatNav indices and computes graph statistics without running
search benchmarks. It's useful for analyzing the graph structure properties.

Usage:
    python compute_indegree_stats.py \
        --dataset /path/to/data.npy \
        --dataset-name mnist \
        --metric l2 \
        --num-node-links 32 \
        --ef-construction 100 200

    # With HNSW base layer
    python compute_indegree_stats.py \
        --dataset /path/to/data.npy \
        --dataset-name mnist \
        --metric l2 \
        --use-hnsw-base-layer \
        --num-node-links 32 \
        --ef-construction 100 200
"""

import time
import json
import argparse
import os
import logging
from typing import List, Dict, Union, Optional
import numpy as np
import hnswlib
import flatnav
from flatnav.data_type import DataType


FLATNAV_DATA_TYPES = {
    "float32": DataType.float32,
    "uint8": DataType.uint8,
    "int8": DataType.int8,
}


def compute_indegree_distribution(
    index: Union[flatnav.index.IndexL2Float, flatnav.index.IndexIPFloat],
) -> Dict[int, float]:
    """
    Compute in-degree distribution for a FlatNav index.

    :param index: FlatNav index to analyze.
    :return: Dictionary mapping in-degree values to percentage of nodes.
    """
    try:
        distribution = index.get_indegree_distribution()

        logging.info("=" * 80)
        logging.info("In-Degree Distribution")
        logging.info("=" * 80)

        sorted_distribution = sorted(distribution.items())

        logging.info(f"{'In-Degree':<15} {'Percentage':<15} {'Bar'}")
        logging.info("-" * 80)

        for indegree, percentage in sorted_distribution[:30]:  # Show first 30
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            logging.info(f"{indegree:<15} {percentage:>10.4f}%    {bar}")

        if len(sorted_distribution) > 30:
            logging.info(f"... ({len(sorted_distribution) - 30} more entries)")

        logging.info("=" * 80)
        logging.info(f"Total unique in-degree values: {len(distribution)}")
        logging.info("=" * 80)

        return distribution
    except Exception as e:
        logging.error(f"Error computing in-degree distribution: {e}", exc_info=True)
        return {}


def compute_hub_stats(
    index: Union[flatnav.index.IndexL2Float, flatnav.index.IndexIPFloat],
    hub_percentile: float = 10.0,
) -> Dict[str, float]:
    """
    Compute hub node statistics for a FlatNav index.

    :param index: FlatNav index to analyze.
    :param hub_percentile: Percentile threshold for hub classification.
    :return: Dictionary containing hub statistics.
    """
    try:
        stats = index.get_hub_statistics(hub_percentile)

        logging.info("=" * 80)
        logging.info("Hub Node Statistics")
        logging.info("=" * 80)
        logging.info(f"Hub Percentile: {hub_percentile}%")
        logging.info("-" * 80)

        logging.info("NODE CLASSIFICATION:")
        logging.info(f"  Number of Hubs: {int(stats['num_hubs'])} ({hub_percentile}%)")
        logging.info(f"  Number of Regular Nodes: {int(stats['num_regular'])} ({100-hub_percentile}%)")
        logging.info(f"  Hub Threshold (min in-degree): {int(stats['hub_threshold'])}")
        logging.info("-" * 80)

        logging.info("IN-DEGREE STATISTICS:")
        logging.info(f"  Average Hub In-Degree: {stats['avg_hub_indegree']:.2f}")
        logging.info(f"  Average Regular In-Degree: {stats['avg_regular_indegree']:.2f}")
        logging.info(f"  Overall Average In-Degree: {stats['avg_indegree']:.2f}")
        logging.info(f"  Median In-Degree: {stats['median_indegree']:.2f}")
        logging.info(f"  Max In-Degree: {int(stats['max_indegree'])}")
        logging.info(f"  Min In-Degree: {int(stats['min_indegree'])}")
        logging.info("-" * 80)

        logging.info("CONNECTIVITY PATTERNS:")
        logging.info(f"  Hub → Hub: {stats['hub_to_hub_pct']:.2f}%")
        logging.info(f"  Hub → Regular: {stats['hub_to_regular_pct']:.2f}%")
        logging.info(f"  Regular → Hub: {stats['regular_to_hub_pct']:.2f}%")
        logging.info(f"  Regular → Regular: {stats['regular_to_regular_pct']:.2f}%")
        logging.info("=" * 80)

        return stats
    except Exception as e:
        logging.error(f"Error computing hub statistics: {e}", exc_info=True)
        return {}


def create_hnsw_base_layer(
    data: np.ndarray,
    space: str,
    dim: int,
    dataset_size: int,
    ef_construction: int,
    max_edges_per_node: int,
    num_threads: int,
    base_layer_filename: str,
) -> None:
    """
    Create HNSW index and save its base layer graph.
    """
    hnsw_index = hnswlib.Index(space=space, dim=dim)
    hnsw_index.init_index(
        max_elements=dataset_size,
        ef_construction=ef_construction,
        M=max_edges_per_node
    )
    hnsw_index.set_num_threads(num_threads)

    start = time.time()
    hnsw_index.add_items(data=data, ids=np.arange(dataset_size))
    end = time.time()
    logging.info(f"HNSW indexing time = {end - start:.2f} seconds")

    hnsw_index.save_base_layer_graph(filename=base_layer_filename)
    logging.info(f"Saved HNSW base layer to {base_layer_filename}")


def build_flatnav_index(
    train_dataset: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
    data_type: str = "float32",
    use_hnsw_base_layer: bool = False,
    num_build_threads: int = 1,
) -> Union[flatnav.index.IndexL2Float, flatnav.index.IndexIPFloat]:
    """
    Build a FlatNav index for graph analysis.

    :param train_dataset: Training data array.
    :param distance_type: Distance metric ('l2' or 'angular').
    :param max_edges_per_node: Maximum edges per node (M parameter).
    :param ef_construction: ef_construction parameter.
    :param data_type: Data type for the index.
    :param use_hnsw_base_layer: Whether to use HNSW's base layer connectivity.
    :param num_build_threads: Number of threads for construction.
    :return: Built FlatNav index.
    """
    dataset_size = train_dataset.shape[0]
    dim = train_dataset.shape[1]

    if use_hnsw_base_layer:
        # Create temporary file for HNSW base layer
        temp_filename = f"temp_hnsw_base_{os.getpid()}.mtx"

        _distance_type = distance_type if distance_type == "l2" else "ip"
        create_hnsw_base_layer(
            data=train_dataset,
            space=_distance_type,
            dim=dim,
            dataset_size=dataset_size,
            ef_construction=ef_construction,
            max_edges_per_node=max_edges_per_node // 2,  # HNSW uses M*2 in base layer
            num_threads=num_build_threads,
            base_layer_filename=temp_filename,
        )

        index = flatnav.index.create(
            distance_type=distance_type,
            index_data_type=FLATNAV_DATA_TYPES[data_type],
            dim=dim,
            dataset_size=dataset_size,
            max_edges_per_node=max_edges_per_node,
            verbose=False,
            collect_stats=True,
        )

        index.allocate_nodes(data=train_dataset).build_graph_links(
            mtx_filename=temp_filename
        )

        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            logging.info(f"Removed temporary file: {temp_filename}")

    else:
        index = flatnav.index.create(
            distance_type=distance_type,
            index_data_type=FLATNAV_DATA_TYPES[data_type],
            dim=dim,
            dataset_size=dataset_size,
            max_edges_per_node=max_edges_per_node,
            verbose=True,
            collect_stats=True,
        )
        index.set_num_threads(num_build_threads)

        start = time.time()
        index.add(
            data=train_dataset,
            ef_construction=ef_construction,
            num_initializations=100
        )
        end = time.time()
        logging.info(f"FlatNav indexing time = {end - start:.2f} seconds")

    return index


def generate_experiment_key(
    dataset_name: str,
    node_links: int,
    ef_construction: int,
    use_hnsw_base_layer: bool,
) -> str:
    """
    Generate a unique experiment key that includes all relevant parameters.
    """
    base_type = "hnsw-base" if use_hnsw_base_layer else "flatnav-native"
    return f"{dataset_name}_{base_type}_M{node_links}_ef{ef_construction}"


def save_distribution_data(
    output_file: str,
    experiment_key: str,
    node_links: int,
    ef_construction: int,
    distribution: Dict[int, float],
    hub_stats: Dict[str, float],
) -> None:
    """
    Save in-degree distribution and hub statistics to JSON file.
    """
    distribution_data = {
        "experiment_key": experiment_key,
        "node_links": node_links,
        "ef_construction": ef_construction,
        "distribution": {str(k): v for k, v in distribution.items()},
        "hub_stats": hub_stats,
    }

    # Load existing data or create new
    all_distributions = []
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r") as f:
            try:
                all_distributions = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Could not read {output_file}, starting fresh")

    # Check if this experiment already exists and update it
    found = False
    for i, existing in enumerate(all_distributions):
        if (existing.get("experiment_key") == experiment_key and
            existing.get("node_links") == node_links and
            existing.get("ef_construction") == ef_construction):
            all_distributions[i] = distribution_data
            found = True
            logging.info(f"Updated existing entry for {experiment_key}")
            break

    if not found:
        all_distributions.append(distribution_data)
        logging.info(f"Added new entry for {experiment_key}")

    with open(output_file, "w") as f:
        json.dump(all_distributions, f, indent=4)

    logging.info(f"Saved to {output_file}")


def run_indegree_analysis(
    train_dataset: np.ndarray,
    dataset_name: str,
    distance_type: str,
    node_links_list: List[int],
    ef_construction_list: List[int],
    output_file: str,
    data_type: str = "float32",
    use_hnsw_base_layer: bool = False,
    num_build_threads: int = 1,
    hub_percentile: float = 10.0,
) -> None:
    """
    Run in-degree analysis for all parameter combinations.
    """
    total_runs = len(node_links_list) * len(ef_construction_list)
    current_run = 0

    for node_links in node_links_list:
        for ef_construction in ef_construction_list:
            current_run += 1
            experiment_key = generate_experiment_key(
                dataset_name=dataset_name,
                node_links=node_links,
                ef_construction=ef_construction,
                use_hnsw_base_layer=use_hnsw_base_layer,
            )

            logging.info("=" * 80)
            logging.info(f"Run {current_run}/{total_runs}: {experiment_key}")
            logging.info(f"  M={node_links}, ef_construction={ef_construction}")
            logging.info("=" * 80)

            # Build index
            index = build_flatnav_index(
                train_dataset=train_dataset,
                distance_type=distance_type,
                max_edges_per_node=node_links,
                ef_construction=ef_construction,
                data_type=data_type,
                use_hnsw_base_layer=use_hnsw_base_layer,
                num_build_threads=num_build_threads,
            )

            # Compute statistics
            logging.info("\nComputing in-degree distribution...")
            distribution = compute_indegree_distribution(index)

            logging.info("\nComputing hub statistics...")
            hub_stats = compute_hub_stats(index, hub_percentile=hub_percentile)

            # Save results
            save_distribution_data(
                output_file=output_file,
                experiment_key=experiment_key,
                node_links=node_links,
                ef_construction=ef_construction,
                distribution=distribution,
                hub_stats=hub_stats,
            )

            # Free memory
            del index
            logging.info(f"Completed {experiment_key}\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute in-degree distribution and hub statistics for FlatNav indices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with FlatNav native construction
  python compute_indegree_stats.py \\
    --dataset /path/to/train.npy \\
    --dataset-name mnist \\
    --metric l2 \\
    --num-node-links 32 \\
    --ef-construction 100 200

  # Using HNSW base layer
  python compute_indegree_stats.py \\
    --dataset /path/to/train.npy \\
    --dataset-name mnist \\
    --metric l2 \\
    --use-hnsw-base-layer \\
    --num-node-links 32 \\
    --ef-construction 100 200

  # Multiple M values
  python compute_indegree_stats.py \\
    --dataset /path/to/train.npy \\
    --dataset-name gist \\
    --metric l2 \\
    --num-node-links 16 32 64 \\
    --ef-construction 100 200 400

  # Custom output file
  python compute_indegree_stats.py \\
    --dataset /path/to/train.npy \\
    --dataset-name sift \\
    --metric l2 \\
    --output ../metrics/sift_indegree_stats.json
        """
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the training dataset (.npy file)"
    )

    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Name identifier for the dataset (used in experiment keys)"
    )

    parser.add_argument(
        "--metric",
        required=True,
        choices=["l2", "angular"],
        help="Distance metric to use"
    )

    parser.add_argument(
        "--num-node-links",
        nargs="+",
        type=int,
        default=[32],
        help="Number of edges per node (M parameter). Can specify multiple values."
    )

    parser.add_argument(
        "--ef-construction",
        nargs="+",
        type=int,
        default=[100, 200],
        help="ef_construction parameter. Can specify multiple values."
    )

    parser.add_argument(
        "--data-type",
        default="float32",
        choices=["float32", "uint8", "int8"],
        help="Data type for the index"
    )

    parser.add_argument(
        "--use-hnsw-base-layer",
        action="store_true",
        help="Use HNSW's base layer connectivity instead of FlatNav native construction"
    )

    parser.add_argument(
        "--num-build-threads",
        type=int,
        default=1,
        help="Number of threads for index construction"
    )

    parser.add_argument(
        "--hub-percentile",
        type=float,
        default=10.0,
        help="Percentile threshold for hub classification (default: 10.0)"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: ../metrics/indegree_stats_{dataset_name}.json)"
    )

    parser.add_argument(
        "--train-dataset-range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Range of training data to use [start, end)"
    )

    return parser.parse_args()


def load_dataset(path: str, data_range: Optional[List[int]] = None) -> np.ndarray:
    """
    Load dataset from numpy file.

    :param path: Path to the .npy file
    :param data_range: Optional [start, end] range to load subset
    :return: Dataset as numpy array
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path)

    if data_range:
        start, end = data_range
        data = data[start:end]

    return data.astype(np.float32, copy=False)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_arguments()

    # Load dataset
    logging.info(f"Loading dataset from {args.dataset}")
    train_data = load_dataset(args.dataset, args.train_dataset_range)
    logging.info(f"Dataset shape: {train_data.shape}")

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = f"../metrics/indegree_stats_{args.dataset_name}.json"

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info(f"Output file: {output_file}")
    logging.info(f"Parameters:")
    logging.info(f"  Dataset: {args.dataset_name}")
    logging.info(f"  Metric: {args.metric}")
    logging.info(f"  M values: {args.num_node_links}")
    logging.info(f"  ef_construction values: {args.ef_construction}")
    logging.info(f"  Use HNSW base layer: {args.use_hnsw_base_layer}")
    logging.info(f"  Hub percentile: {args.hub_percentile}%")

    # Run analysis
    run_indegree_analysis(
        train_dataset=train_data,
        dataset_name=args.dataset_name,
        distance_type=args.metric,
        node_links_list=args.num_node_links,
        ef_construction_list=args.ef_construction,
        output_file=output_file,
        data_type=args.data_type,
        use_hnsw_base_layer=args.use_hnsw_base_layer,
        num_build_threads=args.num_build_threads,
        hub_percentile=args.hub_percentile,
    )

    logging.info("Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
