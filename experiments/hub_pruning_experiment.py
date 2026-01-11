"""
Hub vs Non-Hub Node Pruning Experiment

This experiment tests the hub-highway hypothesis by comparing search performance
when pruning edges from:
1. Hub nodes (top X% most connected nodes by in-degree)
2. Random nodes (X% randomly selected nodes)
3. Anti-hub nodes (bottom X% least connected nodes by in-degree)

The hypothesis is that hub nodes are critical for graph traversal performance,
so pruning their edges should degrade performance more than pruning random nodes
or anti-hub nodes.
"""

import json
import time
import argparse
import logging
import os
import copy
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import flatnav
from flatnav.data_type import DataType
from data_loader import get_data_loader
from run_benchmark import compute_metrics, train_index, FLATNAV_DATA_TYPES

logging.basicConfig(level=logging.INFO)


def compute_indegree_distribution(outdegree_table: List[List[int]]) -> Dict[int, int]:
    """
    Compute the in-degree for each node in the graph.

    :param outdegree_table: List where outdegree_table[i] contains the neighbors of node i.
    :return: Dictionary mapping node_id -> in-degree count.
    """
    num_nodes = len(outdegree_table)
    indegree_counts = {i: 0 for i in range(num_nodes)}

    for node_id, neighbors in enumerate(outdegree_table):
        for neighbor in neighbors:
            if neighbor != node_id:  # Exclude self-loops
                indegree_counts[neighbor] += 1

    return indegree_counts


def find_hub_nodes_by_indegree(
    indegree_counts: Dict[int, int],
    percentile: float
) -> List[int]:
    """
    Find hub nodes based on in-degree (nodes that are pointed to most frequently).

    :param indegree_counts: Dictionary mapping node_id -> in-degree count.
    :param percentile: Percentile threshold (e.g., 95 means top 5% most connected).
    :return: List of node IDs that are hub nodes.
    """
    indegree_values = np.array(list(indegree_counts.values()))
    threshold = np.percentile(indegree_values, percentile)

    hub_nodes = [
        node_id for node_id, indegree in indegree_counts.items()
        if indegree >= threshold
    ]
    return hub_nodes


def find_anti_hub_nodes_by_indegree(
    indegree_counts: Dict[int, int],
    percentile: float
) -> List[int]:
    """
    Find anti-hub nodes based on in-degree (nodes that are pointed to least frequently).

    :param indegree_counts: Dictionary mapping node_id -> in-degree count.
    :param percentile: Percentile threshold (e.g., 5 means bottom 5% least connected).
    :return: List of node IDs that are anti-hub nodes.
    """
    num_nodes = len(indegree_counts)
    num_to_select = int(num_nodes * percentile / 100.0)

    # Sort nodes by in-degree (ascending) and take the bottom X%
    sorted_nodes = sorted(indegree_counts.items(), key=lambda x: x[1])
    anti_hub_nodes = [node_id for node_id, _ in sorted_nodes[:num_to_select]]

    return anti_hub_nodes


def find_hub_nodes_by_access_count(
    node_access_counts: Dict[int, int],
    percentile: float
) -> List[int]:
    """
    Find hub nodes based on access frequency during search.

    :param node_access_counts: Dictionary mapping node_id -> access count.
    :param percentile: Percentile threshold (e.g., 95 means top 5% most accessed).
    :return: List of node IDs that are hub nodes.
    """
    access_values = np.array(list(node_access_counts.values()))
    threshold = np.percentile(access_values, percentile)

    hub_nodes = [
        node_id for node_id, count in node_access_counts.items()
        if count >= threshold
    ]
    return hub_nodes


def select_random_nodes(
    num_nodes: int,
    percentage: float,
    seed: int = 42
) -> List[int]:
    """
    Randomly select a percentage of nodes.

    :param num_nodes: Total number of nodes in the graph.
    :param percentage: Percentage of nodes to select (e.g., 5.0 for 5%).
    :param seed: Random seed for reproducibility.
    :return: List of randomly selected node IDs.
    """
    np.random.seed(seed)
    num_to_select = int(num_nodes * percentage / 100.0)
    return list(np.random.choice(num_nodes, size=num_to_select, replace=False))


def run_search_benchmark(
    index: Union[flatnav.index.IndexL2Float, flatnav.index.IndexIPFloat],
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef_search: int,
    k: int = 100,
) -> Dict[str, float]:
    """
    Run search benchmark and return metrics.
    """
    requested_metrics = [
        "recall",
        "qps",
        "latency_p50",
        "latency_p95",
        "latency_p99",
        "distance_computations",
    ]

    metrics = compute_metrics(
        requested_metrics=requested_metrics,
        index=index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
    )
    return metrics


def run_pruning_experiment(
    train_dataset: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
    ef_search_params: List[int],
    pruning_percentages: List[float],
    alpha: float,
    hub_identification_method: str = "indegree",
    data_type: str = "float32",
    use_hnsw_base_layer: bool = True,
    hnsw_base_layer_filename: str = "hnsw_base_layer.mtx",
    num_build_threads: int = 1,
    num_search_threads: int = 1,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Run the hub vs random node pruning experiment.

    :param train_dataset: Training vectors.
    :param queries: Query vectors.
    :param ground_truth: Ground truth neighbor IDs.
    :param distance_type: Distance metric ('l2' or 'angular').
    :param max_edges_per_node: Maximum edges per node (M parameter).
    :param ef_construction: Construction beam width.
    :param ef_search_params: List of ef_search values to test.
    :param pruning_percentages: List of percentages to prune (e.g., [1, 2, 5]).
    :param alpha: Fraction of edges to remove from selected nodes.
    :param hub_identification_method: 'indegree' or 'access_count'.
    :param data_type: Data type for the index.
    :param use_hnsw_base_layer: Whether to use HNSW base layer for construction.
    :param hnsw_base_layer_filename: Filename for HNSW base layer.
    :param num_build_threads: Number of build threads.
    :param num_search_threads: Number of search threads.
    :param seed: Random seed.

    :return: Dictionary with experiment results.
    """
    dataset_size, dim = train_dataset.shape
    results = {
        "baseline": [],
        "hub_pruned": [],
        "random_pruned": [],
        "anti_hub_pruned": [],
    }

    # Helper to build an index with common parameters
    def build_index():
        return train_index(
            index_type="flatnav",
            data_type=data_type,
            train_dataset=train_dataset,
            max_edges_per_node=max_edges_per_node,
            ef_construction=ef_construction,
            dataset_size=dataset_size,
            dim=dim,
            distance_type=distance_type,
            use_hnsw_base_layer=use_hnsw_base_layer,
            hnsw_base_layer_filename=hnsw_base_layer_filename,
            num_build_threads=num_build_threads,
        )

    for prune_pct in pruning_percentages:
        logging.info(f"\n{'='*60}")
        logging.info(f"Running experiment with {prune_pct}% pruning")
        logging.info(f"{'='*60}")

        # Build the baseline index (also used to compute hub statistics)
        logging.info("Building baseline index...")
        baseline_index = build_index()
        baseline_index.set_num_threads(num_search_threads)

        # Get graph structure and compute in-degrees
        outdegree_table = baseline_index.get_graph_outdegree_table()
        indegree_counts = compute_indegree_distribution(outdegree_table)

        # Identify hub nodes (top X% by in-degree)
        hub_percentile = 100 - prune_pct  # e.g., 5% pruning means 95th percentile
        # Identify anti-hub nodes (bottom X% by in-degree)
        anti_hub_percentile = prune_pct  # e.g., 5% pruning means 5th percentile

        if hub_identification_method == "indegree":
            hub_nodes = find_hub_nodes_by_indegree(indegree_counts, hub_percentile)
            anti_hub_nodes = find_anti_hub_nodes_by_indegree(indegree_counts, anti_hub_percentile)
        else:
            # Run a warmup search to collect access counts
            logging.info("Running warmup search to collect node access counts...")
            _ = run_search_benchmark(baseline_index, queries[:100], ground_truth[:100], ef_search_params[0])
            node_access_counts = baseline_index.get_node_access_counts()
            hub_nodes = find_hub_nodes_by_access_count(node_access_counts, hub_percentile)
            # For anti-hubs with access count, find least accessed nodes
            anti_hub_nodes = find_anti_hub_nodes_by_indegree(indegree_counts, anti_hub_percentile)
            baseline_index.reset_node_access_distribution()

        # Select random nodes (same count as hub nodes for fair comparison)
        random_nodes = select_random_nodes(dataset_size, prune_pct, seed=seed)

        logging.info(f"Number of hub nodes: {len(hub_nodes)}")
        logging.info(f"Number of anti-hub nodes: {len(anti_hub_nodes)}")
        logging.info(f"Number of random nodes: {len(random_nodes)}")

        # Log statistics about selected nodes
        hub_indegrees = [indegree_counts[n] for n in hub_nodes]
        anti_hub_indegrees = [indegree_counts[n] for n in anti_hub_nodes]
        random_indegrees = [indegree_counts[n] for n in random_nodes]
        logging.info(f"Hub nodes avg in-degree: {np.mean(hub_indegrees):.2f}")
        logging.info(f"Anti-hub nodes avg in-degree: {np.mean(anti_hub_indegrees):.2f}")
        logging.info(f"Random nodes avg in-degree: {np.mean(random_indegrees):.2f}")

        # Run baseline searches for all ef_search values
        logging.info("\nRunning baseline searches (no pruning)...")
        baseline_results_for_pct = []
        for ef_search in ef_search_params:
            logging.info(f"  ef_search={ef_search}")
            baseline_metrics = run_search_benchmark(
                baseline_index, queries, ground_truth, ef_search
            )
            baseline_metrics["pruning_percentage"] = prune_pct
            baseline_metrics["ef_search"] = ef_search
            baseline_metrics["experiment_type"] = "baseline"
            results["baseline"].append(baseline_metrics)
            baseline_results_for_pct.append(baseline_metrics)
            logging.info(f"    Recall: {baseline_metrics['recall']:.4f}, QPS: {baseline_metrics['qps']:.2f}")

        # Free baseline index memory before building next index
        del baseline_index

        # Build and test hub-pruned index
        logging.info("\nBuilding hub-pruned index...")
        hub_pruned_index = build_index()
        hub_pruned_index.set_num_threads(num_search_threads)
        hub_pruned_index.reprune_graph(hub_nodes=hub_nodes, alpha=alpha)

        logging.info("Running hub-pruned searches...")
        hub_results_for_pct = []
        for ef_search in ef_search_params:
            logging.info(f"  ef_search={ef_search}")
            hub_metrics = run_search_benchmark(
                hub_pruned_index, queries, ground_truth, ef_search
            )
            hub_metrics["pruning_percentage"] = prune_pct
            hub_metrics["ef_search"] = ef_search
            hub_metrics["experiment_type"] = "hub_pruned"
            hub_metrics["num_pruned_nodes"] = len(hub_nodes)
            hub_metrics["avg_indegree_pruned"] = np.mean(hub_indegrees)
            results["hub_pruned"].append(hub_metrics)
            hub_results_for_pct.append(hub_metrics)
            logging.info(f"    Recall: {hub_metrics['recall']:.4f}, QPS: {hub_metrics['qps']:.2f}")

        del hub_pruned_index

        # Build and test random-pruned index
        logging.info("\nBuilding random-pruned index...")
        random_pruned_index = build_index()
        random_pruned_index.set_num_threads(num_search_threads)
        random_pruned_index.reprune_graph(hub_nodes=random_nodes, alpha=alpha)

        logging.info("Running random-pruned searches...")
        random_results_for_pct = []
        for ef_search in ef_search_params:
            logging.info(f"  ef_search={ef_search}")
            random_metrics = run_search_benchmark(
                random_pruned_index, queries, ground_truth, ef_search
            )
            random_metrics["pruning_percentage"] = prune_pct
            random_metrics["ef_search"] = ef_search
            random_metrics["experiment_type"] = "random_pruned"
            random_metrics["num_pruned_nodes"] = len(random_nodes)
            random_metrics["avg_indegree_pruned"] = np.mean(random_indegrees)
            results["random_pruned"].append(random_metrics)
            random_results_for_pct.append(random_metrics)
            logging.info(f"    Recall: {random_metrics['recall']:.4f}, QPS: {random_metrics['qps']:.2f}")

        del random_pruned_index

        # Build and test anti-hub-pruned index
        logging.info("\nBuilding anti-hub-pruned index...")
        anti_hub_pruned_index = build_index()
        anti_hub_pruned_index.set_num_threads(num_search_threads)
        anti_hub_pruned_index.reprune_graph(hub_nodes=anti_hub_nodes, alpha=alpha)

        logging.info("Running anti-hub-pruned searches...")
        anti_hub_results_for_pct = []
        for ef_search in ef_search_params:
            logging.info(f"  ef_search={ef_search}")
            anti_hub_metrics = run_search_benchmark(
                anti_hub_pruned_index, queries, ground_truth, ef_search
            )
            anti_hub_metrics["pruning_percentage"] = prune_pct
            anti_hub_metrics["ef_search"] = ef_search
            anti_hub_metrics["experiment_type"] = "anti_hub_pruned"
            anti_hub_metrics["num_pruned_nodes"] = len(anti_hub_nodes)
            anti_hub_metrics["avg_indegree_pruned"] = np.mean(anti_hub_indegrees)
            results["anti_hub_pruned"].append(anti_hub_metrics)
            anti_hub_results_for_pct.append(anti_hub_metrics)
            logging.info(f"    Recall: {anti_hub_metrics['recall']:.4f}, QPS: {anti_hub_metrics['qps']:.2f}")

        del anti_hub_pruned_index

        # Log recall degradation summary for this pruning percentage
        logging.info(f"\n{'='*60}")
        logging.info(f"Recall degradation summary for {prune_pct}% pruning:")
        logging.info(f"{'='*60}")
        for i, ef_search in enumerate(ef_search_params):
            baseline_recall = baseline_results_for_pct[i]['recall']
            hub_recall = hub_results_for_pct[i]['recall']
            random_recall = random_results_for_pct[i]['recall']
            anti_hub_recall = anti_hub_results_for_pct[i]['recall']

            hub_drop = baseline_recall - hub_recall
            random_drop = baseline_recall - random_recall
            anti_hub_drop = baseline_recall - anti_hub_recall

            logging.info(f"\nef_search={ef_search}:")
            logging.info(f"  Hub pruning: {hub_drop:.4f} ({hub_drop/baseline_recall*100:.2f}%)")
            logging.info(f"  Random pruning: {random_drop:.4f} ({random_drop/baseline_recall*100:.2f}%)")
            logging.info(f"  Anti-hub pruning: {anti_hub_drop:.4f} ({anti_hub_drop/baseline_recall*100:.2f}%)")

    return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hub vs Non-Hub Node Pruning Experiment for Hub-Highway Hypothesis"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to training dataset file.",
    )

    parser.add_argument(
        "--queries",
        required=True,
        help="Path to query dataset file.",
    )

    parser.add_argument(
        "--gtruth",
        required=True,
        help="Path to ground truth file.",
    )

    parser.add_argument(
        "--metric",
        required=True,
        default="l2",
        help="Distance type. Options: 'l2' or 'angular'.",
    )

    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Name of the dataset (for output files).",
    )

    parser.add_argument(
        "--num-node-links",
        type=int,
        default=32,
        help="Maximum number of edges per node (M parameter).",
    )

    parser.add_argument(
        "--ef-construction",
        type=int,
        default=200,
        help="ef_construction parameter.",
    )

    parser.add_argument(
        "--ef-search",
        nargs="+",
        type=int,
        default=[100, 200, 400],
        help="ef_search parameters to test.",
    )

    parser.add_argument(
        "--pruning-percentages",
        nargs="+",
        type=float,
        default=[1, 2, 5],
        help="Percentages of nodes to prune (e.g., 1 2 5 for 1%%, 2%%, 5%%).",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Fraction of edges to remove from selected nodes (0.0-1.0).",
    )

    parser.add_argument(
        "--hub-identification",
        type=str,
        default="indegree",
        choices=["indegree", "access_count"],
        help="Method to identify hub nodes: 'indegree' or 'access_count'.",
    )

    parser.add_argument(
        "--data-type",
        type=str,
        default="float32",
        choices=["float32", "uint8", "int8"],
        help="Data type for index.",
    )

    parser.add_argument(
        "--use-hnsw-base-layer",
        action="store_true",
        default=True,
        help="Use HNSW base layer for index construction.",
    )

    parser.add_argument(
        "--num-build-threads",
        type=int,
        default=1,
        help="Number of threads for index construction.",
    )

    parser.add_argument(
        "--num-search-threads",
        type=int,
        default=1,
        help="Number of threads for search.",
    )

    parser.add_argument(
        "--metrics-file",
        type=str,
        default="../metrics/test_building_pruning_metrics.json",
        help="Path to the pruning metrics file to append results to.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--train-dataset-range",
        nargs="+",
        type=int,
        default=None,
        help="Range of training data to use [start, end].",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load data
    data_loader = get_data_loader(
        train_dataset_path=args.dataset,
        queries_path=args.queries,
        ground_truth_path=args.gtruth,
        range=args.train_dataset_range,
    )
    train_data, queries, ground_truth = data_loader.load_data()

    logging.info(f"Dataset: {args.dataset_name}")
    logging.info(f"Train data shape: {train_data.shape}")
    logging.info(f"Queries shape: {queries.shape}")
    logging.info(f"Ground truth shape: {ground_truth.shape}")

    # Use a unique mtx filename per dataset to avoid conflicts when running in parallel
    hnsw_base_layer_filename = f"hnsw_base_layer_{args.dataset_name}.mtx"

    # Run experiment
    results = run_pruning_experiment(
        train_dataset=train_data,
        queries=queries,
        ground_truth=ground_truth,
        distance_type=args.metric.lower(),
        max_edges_per_node=args.num_node_links,
        ef_construction=args.ef_construction,
        ef_search_params=args.ef_search,
        pruning_percentages=args.pruning_percentages,
        alpha=args.alpha,
        hub_identification_method=args.hub_identification,
        data_type=args.data_type,
        use_hnsw_base_layer=args.use_hnsw_base_layer,
        hnsw_base_layer_filename=hnsw_base_layer_filename,
        num_build_threads=args.num_build_threads,
        num_search_threads=args.num_search_threads,
        seed=args.seed,
    )

    # Save results to the main metrics file in the same format as run_benchmark.py
    metrics_file = args.metrics_file
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

    # Load existing metrics if file exists
    all_metrics = {}
    if os.path.exists(metrics_file) and os.path.getsize(metrics_file) > 0:
        with open(metrics_file, "r") as f:
            try:
                all_metrics = json.load(f)
            except json.JSONDecodeError:
                logging.error(f"Error reading {metrics_file}")

    # Format results in the same structure as run_benchmark.py
    # Keys are: {dataset_name}_{experiment_type}_{pruning_pct}pct
    for experiment_type, experiment_results in results.items():
        for result in experiment_results:
            prune_pct = int(result["pruning_percentage"])
            experiment_key = f"{args.dataset_name}_{experiment_type}_{prune_pct}pct"

            if experiment_key not in all_metrics:
                all_metrics[experiment_key] = []

            # Format each result to match run_benchmark.py format
            formatted_result = {
                "node_links": args.num_node_links,
                "ef_construction": args.ef_construction,
                "recall": result["recall"],
                "qps": result["qps"],
                "latency_p50": result["latency_p50"],
                "latency_p95": result["latency_p95"],
                "latency_p99": result["latency_p99"],
                "distance_computations": result["distance_computations"],
                "distance_type": args.metric.lower(),
                "ef_search": result["ef_search"],
                "pruning_percentage": result["pruning_percentage"],
            }

            # Add pruning-specific fields if present
            if "num_pruned_nodes" in result:
                formatted_result["num_pruned_nodes"] = result["num_pruned_nodes"]
            if "avg_indegree_pruned" in result:
                formatted_result["avg_indegree_pruned"] = result["avg_indegree_pruned"]

            all_metrics[experiment_key].append(formatted_result)

    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)

    logging.info(f"\nResults saved to: {metrics_file}")

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    for prune_pct in args.pruning_percentages:
        print(f"\nPruning {prune_pct}% of nodes:")
        print("-" * 40)

        baseline_results = [r for r in results["baseline"] if r["pruning_percentage"] == prune_pct]
        hub_results = [r for r in results["hub_pruned"] if r["pruning_percentage"] == prune_pct]
        random_results = [r for r in results["random_pruned"] if r["pruning_percentage"] == prune_pct]
        anti_hub_results = [r for r in results["anti_hub_pruned"] if r["pruning_percentage"] == prune_pct]

        for i, ef in enumerate(args.ef_search):
            if i < len(baseline_results):
                base_recall = baseline_results[i]["recall"]
                hub_recall = hub_results[i]["recall"]
                random_recall = random_results[i]["recall"]
                anti_hub_recall = anti_hub_results[i]["recall"]

                hub_drop = (base_recall - hub_recall) / base_recall * 100
                random_drop = (base_recall - random_recall) / base_recall * 100
                anti_hub_drop = (base_recall - anti_hub_recall) / base_recall * 100

                print(f"  ef_search={ef}:")
                print(f"    Baseline recall: {base_recall:.4f}")
                print(f"    Hub-pruned recall: {hub_recall:.4f} (drop: {hub_drop:.2f}%)")
                print(f"    Random-pruned recall: {random_recall:.4f} (drop: {random_drop:.2f}%)")
                print(f"    Anti-hub-pruned recall: {anti_hub_recall:.4f} (drop: {anti_hub_drop:.2f}%)")

                # Analyze results
                drops = {"Hub": hub_drop, "Random": random_drop, "Anti-hub": anti_hub_drop}
                max_drop = max(drops, key=drops.get)
                min_drop = min(drops, key=drops.get)
                print(f"    -> {max_drop} pruning causes MOST degradation ({drops[max_drop]:.2f}%)")
                print(f"    -> {min_drop} pruning causes LEAST degradation ({drops[min_drop]:.2f}%)")
                if hub_drop > random_drop and hub_drop > anti_hub_drop:
                    print(f"    => SUPPORTS hub-highway hypothesis")


if __name__ == "__main__":
    main()
