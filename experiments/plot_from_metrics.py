#!/usr/bin/env python3
"""
Generate plots from existing metrics JSON file without re-running experiments.

Usage:
    python plot_from_metrics.py --metrics metrics.json --datasets mnist-784 mnist-784-alpha-1.0 --output my_comparison.png
    python plot_from_metrics.py --metrics metrics.json --pattern "mnist.*alpha.*" --output alpha_comparison.png
    python plot_from_metrics.py --metrics metrics.json --list  # List available datasets
"""

import json
import argparse
import os
import re
from typing import List, Dict, Optional
from plotting.plot import create_plot, create_linestyles
from plotting.metrics import metric_manager


def load_metrics(metrics_file: str) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def list_available_datasets(metrics: Dict) -> None:
    """List all available dataset names in the metrics file."""
    print("\nAvailable datasets in metrics file:")
    print("=" * 60)
    for key in sorted(metrics.keys()):
        num_runs = len(metrics[key])
        print(f"  {key:<50} ({num_runs} runs)")
    print("=" * 60)


def filter_datasets(metrics: Dict, pattern: str) -> Dict:
    """Filter datasets by regex pattern."""
    regex = re.compile(pattern)
    return {k: v for k, v in metrics.items() if regex.search(k)}


def select_datasets(metrics: Dict, dataset_names: List[str]) -> Dict:
    """Select specific datasets by exact name."""
    selected = {}
    for name in dataset_names:
        if name in metrics:
            selected[name] = metrics[name]
        else:
            print(f"Warning: Dataset '{name}' not found in metrics file")
    return selected


def compute_recall_at_percentile(runs: List[Dict], percentile: int, target_recall: float) -> Optional[float]:
    """
    Find the latency at the given percentile for runs that achieve at least target_recall.
    
    Args:
        runs: List of metric runs
        percentile: Latency percentile (50, 95, 99, 999)
        target_recall: Minimum recall threshold
    
    Returns:
        Latency at percentile for best run meeting recall threshold, or None
    """
    latency_key = f"latency_p{percentile}"
    
    # Filter runs that meet recall threshold and have the latency metric
    valid_runs = [
        run for run in runs 
        if run.get('recall', 0) >= target_recall and latency_key in run
    ]
    
    if not valid_runs:
        return None
    
    # Return the best (lowest) latency among valid runs
    return min(run[latency_key] for run in valid_runs)


def create_recall_percentile_data(
    selected_metrics: Dict,
    percentile: int,
    recall_points: List[float]
) -> Dict:
    """
    Create data for recall vs latency percentile plot.
    
    Args:
        selected_metrics: Selected dataset metrics
        percentile: Latency percentile to use
        recall_points: List of recall values to evaluate at
    
    Returns:
        Dictionary mapping dataset names to (recall, latency) tuples
    """
    experiment_runs = {}
    
    for dataset_name, runs in selected_metrics.items():
        points = []
        for recall_target in recall_points:
            latency = compute_recall_at_percentile(runs, percentile, recall_target)
            if latency is not None:
                points.append((dataset_name, recall_target, latency))
        
        if points:
            experiment_runs[dataset_name] = points
    
    return experiment_runs


def create_comparison_plot(
    metrics_file: str,
    dataset_names: Optional[List[str]] = None,
    pattern: Optional[str] = None,
    output_file: str = "comparison_plot.png",
    x_metric: str = "recall",
    y_metric: str = "qps",
    x_scale: str = "linear",
    y_scale: str = "linear",
    raw: bool = False,
    friendly_names: Optional[Dict[str, str]] = None,
    recall_percentile: Optional[int] = None,
) -> None:
    """
    Create a comparison plot from metrics file.
    
    Args:
        metrics_file: Path to metrics JSON file
        dataset_names: List of exact dataset names to plot
        pattern: Regex pattern to match dataset names
        output_file: Output plot filename (will be saved in metrics folder)
        x_metric: Metric for x-axis (default: recall)
        y_metric: Metric for y-axis (default: qps)
        x_scale: Scale for x-axis (linear, log, logit)
        y_scale: Scale for y-axis (linear, log)
        raw: Whether to show all points or just Pareto frontier
        friendly_names: Optional dict to rename datasets in legend
        recall_percentile: If set, create recall vs latency_pXX plot
    """
    
    # Load metrics
    all_metrics = load_metrics(metrics_file)
    
    # Select datasets
    if pattern:
        selected_metrics = filter_datasets(all_metrics, pattern)
        print(f"Found {len(selected_metrics)} datasets matching pattern '{pattern}'")
    elif dataset_names:
        selected_metrics = select_datasets(all_metrics, dataset_names)
        print(f"Selected {len(selected_metrics)} datasets")
    else:
        print("Error: Must provide either --datasets or --pattern")
        return
    
    if not selected_metrics:
        print("No datasets selected. Nothing to plot.")
        return
    
    # Apply friendly names if provided
    if friendly_names:
        selected_metrics = {
            friendly_names.get(k, k): v 
            for k, v in selected_metrics.items()
        }
    
    # Convert to experiment runs format expected by create_plot
    if recall_percentile:
        # Special handling for recall @ percentile plots
        recall_points = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
        experiment_runs = create_recall_percentile_data(
            selected_metrics, 
            recall_percentile, 
            recall_points
        )
        actual_y_metric = f"latency_p{recall_percentile}"
        print(f"Creating recall @ p{recall_percentile} plot")
    else:
        experiment_runs = {}
        for dataset_name, runs in selected_metrics.items():
            experiment_runs[dataset_name] = [
                (dataset_name, run[x_metric], run[y_metric])
                for run in runs
                if x_metric in run and y_metric in run
            ]
        actual_y_metric = y_metric
    
    # Create linestyles
    linestyles = create_linestyles(unique_algorithms=set(experiment_runs.keys()))
    
    # Determine output path - save in metrics folder
    metrics_dir = os.path.dirname(os.path.abspath(metrics_file))
    output_path = os.path.join(metrics_dir, output_file)
    
    # Create plot
    print(f"Creating plot: {output_path}")
    print(f"  X-axis: {x_metric}")
    print(f"  Y-axis: {actual_y_metric}")
    print(f"  Datasets: {list(experiment_runs.keys())}")
    
    create_plot(
        experiment_runs=experiment_runs,
        raw=raw,
        x_scale=x_scale,
        y_scale=y_scale,
        x_axis_metric=x_metric,
        y_axis_metric=actual_y_metric,
        linestyles=linestyles,
        plot_name=output_path,
    )
    
    print(f"✓ Plot saved to: {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from metrics JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python plot_from_metrics.py --metrics ../metrics/metrics.json --list
  
  # Compare specific datasets (QPS vs Recall)
  python plot_from_metrics.py \\
    --metrics ../metrics/metrics.json \\
    --datasets mnist-784_flatnav mnist-784-alpha-1.0_flatnav \\
    --output mnist_comparison.png
  
  # Compare all alpha variants using regex
  python plot_from_metrics.py \\
    --metrics ../metrics/metrics.json \\
    --pattern "mnist.*alpha.*" \\
    --output alpha_sweep.png
  
  # Compare with custom names in legend
  python plot_from_metrics.py \\
    --metrics ../metrics/metrics.json \\
    --datasets mnist-784_flatnav mnist-784-alpha-1.0_flatnav \\
    --friendly-names "HNSW Heuristic" "Alpha-Diversity (α=1.0)" \\
    --output comparison.png
  
  # Recall @ p99 latency
  python plot_from_metrics.py \\
    --metrics ../metrics/metrics.json \\
    --datasets mnist-784_flatnav mnist-784-alpha-0.5_flatnav \\
    --recall-percentile 99 \\
    --output recall_at_p99.png
  
  # Recall @ p50 latency (median)
  python plot_from_metrics.py \\
    --metrics ../metrics/metrics.json \\
    --pattern "mnist.*" \\
    --recall-percentile 50 \\
    --output recall_at_p50.png
  
  # Standard latency plot (recall on x-axis, latency on y-axis)
  python plot_from_metrics.py \\
    --metrics ../metrics/metrics.json \\
    --datasets mnist-784_flatnav mnist-784-alpha-0.5_flatnav \\
    --x-metric recall \\
    --y-metric latency_p99 \\
    --output latency_comparison.png
        """
    )
    
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics JSON file"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Exact dataset names to plot (e.g., mnist-784_flatnav mnist-784-alpha-1.0_flatnav)"
    )
    
    parser.add_argument(
        "--pattern",
        help="Regex pattern to match dataset names (e.g., 'mnist.*alpha.*')"
    )
    
    parser.add_argument(
        "--output",
        default="comparison_plot.png",
        help="Output plot filename (will be saved in metrics folder, default: comparison_plot.png)"
    )
    
    parser.add_argument(
        "--x-metric",
        default="recall",
        help="Metric for x-axis (default: recall)"
    )
    
    parser.add_argument(
        "--y-metric",
        default="qps",
        help="Metric for y-axis (default: qps)"
    )
    
    parser.add_argument(
        "--recall-percentile",
        type=int,
        choices=[50, 95, 99, 999],
        help="Create recall @ latency percentile plot (e.g., 50, 95, 99, 999)"
    )
    
    parser.add_argument(
        "--x-scale",
        default="linear",
        choices=["linear", "log", "logit"],
        help="Scale for x-axis (default: linear)"
    )
    
    parser.add_argument(
        "--y-scale",
        default="linear",
        choices=["linear", "log"],
        help="Scale for y-axis (default: linear)"
    )
    
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show all data points, not just Pareto frontier"
    )
    
    parser.add_argument(
        "--friendly-names",
        nargs="+",
        help="Friendly names for datasets in legend (same order as --datasets)"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Check if metrics file exists
    if not os.path.exists(args.metrics):
        print(f"Error: Metrics file not found: {args.metrics}")
        return 1
    
    # Load metrics
    metrics = load_metrics(args.metrics)
    
    # List datasets if requested
    if args.list:
        list_available_datasets(metrics)
        return 0
    
    # Check that either datasets or pattern is provided
    if not args.datasets and not args.pattern:
        print("Error: Must provide either --datasets or --pattern")
        print("Use --list to see available datasets")
        return 1
    
    # Parse friendly names if provided
    friendly_names = None
    if args.friendly_names:
        if not args.datasets:
            print("Error: --friendly-names requires --datasets")
            return 1
        if len(args.friendly_names) != len(args.datasets):
            print("Error: Number of friendly names must match number of datasets")
            return 1
        friendly_names = dict(zip(args.datasets, args.friendly_names))
    
    # Create plot
    create_comparison_plot(
        metrics_file=args.metrics,
        dataset_names=args.datasets,
        pattern=args.pattern,
        output_file=args.output,
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
        raw=args.raw,
        friendly_names=friendly_names,
        recall_percentile=args.recall_percentile,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())