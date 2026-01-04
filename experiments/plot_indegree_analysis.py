#!/usr/bin/env python3
"""
Analyze and visualize in-degree distributions from graph index metrics.

This script generates various plots to understand the in-degree distribution
characteristics of graph-based nearest neighbor indices.

Usage:
    python plot_indegree_analysis.py --input ../metrics/metrics_indegree_distribution.json
    python plot_indegree_analysis.py --input ../metrics/metrics_indegree_distribution.json --output-dir ../metrics/indegree_plots
    python plot_indegree_analysis.py --input ../metrics/metrics_indegree_distribution.json --experiments gist_flatnav
"""

import json
import argparse
import os
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats


def load_distribution_data(filepath: str) -> List[Dict]:
    """Load in-degree distribution data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_multiple_files(filepaths: List[str]) -> List[Dict]:
    """Load and combine distribution data from multiple JSON files."""
    all_data = []
    for filepath in filepaths:
        if os.path.exists(filepath):
            all_data.extend(load_distribution_data(filepath))
        else:
            print(f"Warning: File not found: {filepath}")
    return all_data


def filter_experiments(
    data: List[Dict],
    experiment_keys: Optional[List[str]] = None,
    pattern: Optional[str] = None,
) -> List[Dict]:
    """Filter experiments by keys or regex pattern."""
    if pattern:
        regex = re.compile(pattern)
        return [d for d in data if regex.search(d['experiment_key'])]
    elif experiment_keys:
        return [d for d in data if d['experiment_key'] in experiment_keys]
    return data


def compute_statistics(distribution: Dict[str, float], hub_stats: Optional[Dict] = None) -> Dict:
    """
    Compute summary statistics from an in-degree distribution.

    If hub_stats is provided (from compute_indegree_stats.py), use those values
    for more accurate statistics. Otherwise, reconstruct from distribution.

    Returns dict with: mean, median, std, min, max, mode, skewness, kurtosis
    """
    # If we have pre-computed hub stats, use them
    if hub_stats:
        return {
            'mean': hub_stats.get('avg_indegree', 0),
            'median': hub_stats.get('median_indegree', 0),
            'std': 0,  # Not available in hub_stats
            'min': int(hub_stats.get('min_indegree', 0)),
            'max': int(hub_stats.get('max_indegree', 0)),
            'mode': 0,  # Compute from distribution below
            'skewness': 0,
            'kurtosis': 0,
            'p95': 0,
            'p99': 0,
            'num_hubs': int(hub_stats.get('num_hubs', 0)),
            'hub_threshold': int(hub_stats.get('hub_threshold', 0)),
            'avg_hub_indegree': hub_stats.get('avg_hub_indegree', 0),
            'avg_regular_indegree': hub_stats.get('avg_regular_indegree', 0),
        }

    # Reconstruct data from distribution
    indegrees = []
    for degree_str, percentage in distribution.items():
        degree = int(degree_str)
        # Convert percentage to count (use 10000 as base for precision)
        count = int(percentage * 100)
        indegrees.extend([degree] * max(1, count))

    if not indegrees:
        return {}

    indegrees = np.array(indegrees)

    # Find mode (most frequent in-degree)
    sorted_dist = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    mode = int(sorted_dist[0][0]) if sorted_dist else 0

    return {
        'mean': np.mean(indegrees),
        'median': np.median(indegrees),
        'std': np.std(indegrees),
        'min': int(np.min(indegrees)),
        'max': int(np.max(indegrees)),
        'mode': mode,
        'skewness': stats.skew(indegrees),
        'kurtosis': stats.kurtosis(indegrees),
        'p95': np.percentile(indegrees, 95),
        'p99': np.percentile(indegrees, 99),
    }


def get_experiment_label(exp: Dict) -> str:
    """Create a readable label for an experiment."""
    return f"{exp['experiment_key']} (M={exp['node_links']}, ef={exp['ef_construction']})"


def plot_distribution_comparison(
    experiments: List[Dict],
    output_path: str,
    title: str = "In-Degree Distribution Comparison",
    log_scale: bool = False,
    max_degree: Optional[int] = None,
) -> None:
    """
    Plot multiple in-degree distributions on the same axes for comparison.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for idx, exp in enumerate(experiments):
        distribution = exp['distribution']
        if not distribution:
            continue

        sorted_items = sorted([(int(k), v) for k, v in distribution.items()])
        indegrees = np.array([item[0] for item in sorted_items])
        percentages = np.array([item[1] for item in sorted_items])

        if max_degree:
            mask = indegrees <= max_degree
            indegrees = indegrees[mask]
            percentages = percentages[mask]

        label = get_experiment_label(exp)
        ax.plot(indegrees, percentages, marker='.', markersize=4,
                label=label, color=colors[idx], linewidth=1.5, alpha=0.8)

    ax.set_xlabel("In-Degree", fontsize=12)
    ax.set_ylabel("Percentage of Nodes (%)", fontsize=12)
    ax.set_title(title, fontsize=14)

    if log_scale:
        ax.set_yscale('log')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cdf(
    experiments: List[Dict],
    output_path: str,
    title: str = "Cumulative In-Degree Distribution (CDF)",
) -> None:
    """
    Plot cumulative distribution function for in-degree.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for idx, exp in enumerate(experiments):
        distribution = exp['distribution']
        if not distribution:
            continue

        sorted_items = sorted([(int(k), v) for k, v in distribution.items()])
        indegrees = np.array([item[0] for item in sorted_items])
        percentages = np.array([item[1] for item in sorted_items])

        # Compute CDF
        cdf = np.cumsum(percentages)

        label = get_experiment_label(exp)
        ax.plot(indegrees, cdf, label=label, color=colors[idx], linewidth=2, alpha=0.8)

    ax.set_xlabel("In-Degree", fontsize=12)
    ax.set_ylabel("Cumulative Percentage (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Median (50%)')
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5, label='95th percentile')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_log_log(
    experiments: List[Dict],
    output_path: str,
    title: str = "In-Degree Distribution (Log-Log Scale)",
) -> None:
    """
    Plot in-degree distribution on log-log scale to check for power-law behavior.
    A straight line on log-log scale indicates power-law distribution.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for idx, exp in enumerate(experiments):
        distribution = exp['distribution']
        if not distribution:
            continue

        sorted_items = sorted([(int(k), v) for k, v in distribution.items()])
        # Filter out zero degrees and zero percentages for log scale
        filtered = [(d, p) for d, p in sorted_items if d > 0 and p > 0]
        if not filtered:
            continue

        indegrees = np.array([item[0] for item in filtered])
        percentages = np.array([item[1] for item in filtered])

        label = get_experiment_label(exp)
        ax.scatter(indegrees, percentages, s=20, label=label, color=colors[idx], alpha=0.7)

    ax.set_xlabel("In-Degree (log scale)", fontsize=12)
    ax.set_ylabel("Percentage of Nodes (log scale)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_statistics_comparison(
    experiments: List[Dict],
    output_path: str,
    title: str = "In-Degree Distribution Statistics",
) -> None:
    """
    Create a bar chart comparing key statistics across experiments.
    """
    stats_list = []
    labels = []

    for exp in experiments:
        if not exp['distribution']:
            continue
        stats_data = compute_statistics(exp['distribution'])
        if stats_data:
            stats_list.append(stats_data)
            labels.append(f"{exp['experiment_key']}\nM={exp['node_links']}, ef={exp['ef_construction']}")

    if not stats_list:
        print("No valid statistics to plot")
        return

    # Create subplots for different statistics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    metrics = ['mean', 'median', 'std', 'mode', 'p95', 'max']
    metric_labels = ['Mean In-Degree', 'Median In-Degree', 'Std Dev',
                     'Mode (Most Common)', '95th Percentile', 'Max In-Degree']

    x = np.arange(len(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for ax, metric, metric_label in zip(axes, metrics, metric_labels):
        values = [s.get(metric, 0) for s in stats_list]
        bars = ax.bar(x, values, color=colors, alpha=0.8)
        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def extract_base_experiment_key(experiment_key: str) -> str:
    """
    Extract the base experiment key by removing ef_construction suffix.
    E.g., 'mnist_flatnav-native_M32_ef100' -> 'mnist_flatnav-native_M32'
    """
    # Try to remove _efXXX suffix
    match = re.match(r'^(.+)_ef\d+$', experiment_key)
    if match:
        return match.group(1)
    return experiment_key


def plot_ef_construction_effect(
    experiments: List[Dict],
    output_path: str,
    title: str = "Effect of ef_construction on In-Degree Distribution",
) -> None:
    """
    For experiments with same dataset but different ef_construction,
    show how the distribution changes.
    """
    # Group by base experiment key (without ef suffix) and node_links
    groups = {}
    for exp in experiments:
        base_key = extract_base_experiment_key(exp['experiment_key'])
        key = (base_key, exp['node_links'])
        if key not in groups:
            groups[key] = []
        groups[key].append(exp)

    # Filter groups with multiple ef_construction values
    multi_ef_groups = {k: v for k, v in groups.items() if len(v) > 1}

    if not multi_ef_groups:
        print("No experiments with varying ef_construction found")
        return

    n_groups = len(multi_ef_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 6))
    if n_groups == 1:
        axes = [axes]

    for ax, ((base_key, node_links), exps) in zip(axes, multi_ef_groups.items()):
        # Sort by ef_construction
        exps_sorted = sorted(exps, key=lambda x: x['ef_construction'])
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(exps_sorted)))

        for exp, color in zip(exps_sorted, colors):
            distribution = exp['distribution']
            if not distribution:
                continue

            sorted_items = sorted([(int(k), v) for k, v in distribution.items()])
            indegrees = np.array([item[0] for item in sorted_items])
            percentages = np.array([item[1] for item in sorted_items])

            # Limit x-axis for clarity
            mask = indegrees <= 100
            indegrees = indegrees[mask]
            percentages = percentages[mask]

            label = f"ef={exp['ef_construction']}"
            ax.plot(indegrees, percentages, label=label, color=color,
                   linewidth=2, alpha=0.8)

        ax.set_xlabel("In-Degree", fontsize=11)
        ax.set_ylabel("Percentage of Nodes (%)", fontsize=11)
        ax.set_title(f"{base_key}\n(M={node_links})", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(left=0)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def extract_dataset_name(experiment_key: str) -> str:
    """
    Extract the dataset name from experiment key.
    E.g., 'mnist_flatnav-native_M32_ef100' -> 'mnist'
    E.g., 'gist_flatnav-native_M32_ef200' -> 'gist'
    """
    # Dataset name is typically the first part before _flatnav or _hnsw
    match = re.match(r'^([^_]+)', experiment_key)
    if match:
        return match.group(1)
    return experiment_key


def plot_dataset_comparison(
    experiments: List[Dict],
    output_path: str,
    title: str = "In-Degree Distribution Comparison Across Datasets",
) -> None:
    """
    For experiments with same ef_construction but different datasets,
    show how the distributions compare.
    Creates one subplot per ef_construction value.
    """
    # Group by ef_construction and node_links
    groups = {}
    for exp in experiments:
        key = (exp['ef_construction'], exp['node_links'])
        if key not in groups:
            groups[key] = []
        groups[key].append(exp)

    # Filter groups with multiple datasets
    multi_dataset_groups = {k: v for k, v in groups.items() if len(v) > 1}

    if not multi_dataset_groups:
        print("No experiments with multiple datasets at same ef_construction found")
        return

    n_groups = len(multi_dataset_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 6))
    if n_groups == 1:
        axes = [axes]

    for ax, ((ef_construction, node_links), exps) in zip(axes, sorted(multi_dataset_groups.items())):
        # Sort by dataset name for consistent ordering
        exps_sorted = sorted(exps, key=lambda x: extract_dataset_name(x['experiment_key']))
        colors = plt.cm.tab10(np.linspace(0, 1, len(exps_sorted)))

        for exp, color in zip(exps_sorted, colors):
            distribution = exp['distribution']
            if not distribution:
                continue

            sorted_items = sorted([(int(k), v) for k, v in distribution.items()])
            indegrees = np.array([item[0] for item in sorted_items])
            percentages = np.array([item[1] for item in sorted_items])

            # Limit x-axis for clarity
            mask = indegrees <= 100
            indegrees = indegrees[mask]
            percentages = percentages[mask]

            dataset_name = extract_dataset_name(exp['experiment_key'])
            ax.plot(indegrees, percentages, label=dataset_name, color=color,
                   linewidth=2, alpha=0.8)

        ax.set_xlabel("In-Degree", fontsize=11)
        ax.set_ylabel("Percentage of Nodes (%)", fontsize=11)
        ax.set_title(f"ef_construction={ef_construction}\n(M={node_links})", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(left=0)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_tail_analysis(
    experiments: List[Dict],
    output_path: str,
    percentile_threshold: float = 95,
    title: str = "Tail Distribution Analysis (High In-Degree Nodes)",
) -> None:
    """
    Focus on the tail of the distribution - nodes with unusually high in-degree.
    These are potential hub nodes that may affect search performance.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    tail_percentages = []
    labels = []
    max_degrees = []

    for idx, exp in enumerate(experiments):
        distribution = exp['distribution']
        if not distribution:
            continue

        sorted_items = sorted([(int(k), v) for k, v in distribution.items()])
        indegrees = np.array([item[0] for item in sorted_items])
        percentages = np.array([item[1] for item in sorted_items])

        # Compute cumulative to find threshold
        cdf = np.cumsum(percentages)

        # Find where CDF crosses threshold
        threshold_idx = np.searchsorted(cdf, percentile_threshold)
        if threshold_idx < len(indegrees):
            threshold_degree = indegrees[threshold_idx]

            # Plot tail distribution (beyond threshold)
            tail_mask = indegrees >= threshold_degree
            if np.any(tail_mask):
                ax1.plot(indegrees[tail_mask], percentages[tail_mask],
                        marker='.', markersize=4, label=get_experiment_label(exp),
                        color=colors[idx], linewidth=1.5, alpha=0.8)

        # Collect tail statistics
        stats_data = compute_statistics(distribution)
        if stats_data:
            tail_pct = 100 - cdf[threshold_idx] if threshold_idx < len(cdf) else 0
            tail_percentages.append(tail_pct)
            labels.append(f"{exp['experiment_key']}\nM={exp['node_links']}")
            max_degrees.append(stats_data['max'])

    ax1.set_xlabel("In-Degree", fontsize=11)
    ax1.set_ylabel("Percentage of Nodes (%)", fontsize=11)
    ax1.set_title(f"Tail Distribution (>{percentile_threshold}th percentile)", fontsize=12)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='upper right')

    # Bar chart of max degrees
    if labels:
        x = np.arange(len(labels))
        ax2.bar(x, max_degrees, color=colors[:len(labels)], alpha=0.8)
        ax2.set_ylabel("Maximum In-Degree", fontsize=11)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax2.set_title("Maximum In-Degree (Hub Nodes)", fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, v in enumerate(max_degrees):
            ax2.text(i, v + max(max_degrees)*0.01, str(v), ha='center', fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_histogram_overlay(
    experiments: List[Dict],
    output_path: str,
    max_degree: int = 80,
    title: str = "In-Degree Histogram Comparison",
) -> None:
    """
    Plot overlapping histograms for direct visual comparison.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for idx, exp in enumerate(experiments):
        distribution = exp['distribution']
        if not distribution:
            continue

        # Create bins
        bins = np.arange(0, max_degree + 2) - 0.5
        heights = np.zeros(max_degree + 1)

        for degree_str, pct in distribution.items():
            degree = int(degree_str)
            if degree <= max_degree:
                heights[degree] = pct

        label = get_experiment_label(exp)
        ax.bar(np.arange(max_degree + 1), heights, width=0.8,
               label=label, color=colors[idx], alpha=0.5, edgecolor=colors[idx])

    ax.set_xlabel("In-Degree", fontsize=12)
    ax.set_ylabel("Percentage of Nodes (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, max_degree + 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_statistics_table(experiments: List[Dict]) -> None:
    """Print a formatted table of statistics for all experiments."""
    print("\n" + "=" * 110)
    print("IN-DEGREE DISTRIBUTION STATISTICS")
    print("=" * 110)

    # Check if any experiment has hub_stats
    has_hub_stats = any(exp.get('hub_stats') for exp in experiments)

    if has_hub_stats:
        header = f"{'Experiment':<40} {'M':>4} {'ef':>4} {'Mean':>8} {'Median':>8} {'Max':>6} {'Hubs':>6} {'HubThresh':>9}"
    else:
        header = f"{'Experiment':<40} {'M':>4} {'ef':>4} {'Mean':>8} {'Median':>8} {'Std':>8} {'Mode':>6} {'Max':>6}"
    print(header)
    print("-" * 110)

    for exp in experiments:
        if not exp['distribution']:
            continue

        hub_stats = exp.get('hub_stats')
        stats_data = compute_statistics(exp['distribution'], hub_stats)
        if stats_data:
            if has_hub_stats and hub_stats:
                row = (
                    f"{exp['experiment_key']:<40} "
                    f"{exp['node_links']:>4} "
                    f"{exp['ef_construction']:>4} "
                    f"{stats_data['mean']:>8.2f} "
                    f"{stats_data['median']:>8.1f} "
                    f"{stats_data['max']:>6} "
                    f"{stats_data.get('num_hubs', 0):>6} "
                    f"{stats_data.get('hub_threshold', 0):>9}"
                )
            else:
                row = (
                    f"{exp['experiment_key']:<40} "
                    f"{exp['node_links']:>4} "
                    f"{exp['ef_construction']:>4} "
                    f"{stats_data['mean']:>8.2f} "
                    f"{stats_data['median']:>8.1f} "
                    f"{stats_data['std']:>8.2f} "
                    f"{stats_data['mode']:>6} "
                    f"{stats_data['max']:>6}"
                )
            print(row)

    print("=" * 110 + "\n")


def generate_all_plots_from_data(
    experiments: List[Dict],
    output_dir: str,
) -> None:
    """Generate all analysis plots from pre-loaded experiment data."""
    if not experiments:
        print("No valid experiments found with distribution data")
        return

    print(f"Analyzing {len(experiments)} experiments...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Print statistics table
    print_statistics_table(experiments)

    # Generate plots
    plot_distribution_comparison(
        experiments,
        os.path.join(output_dir, "distribution_comparison.png"),
        max_degree=80
    )

    plot_distribution_comparison(
        experiments,
        os.path.join(output_dir, "distribution_comparison_log.png"),
        log_scale=True
    )

    plot_cdf(
        experiments,
        os.path.join(output_dir, "cdf.png")
    )

    plot_log_log(
        experiments,
        os.path.join(output_dir, "log_log_distribution.png")
    )

    plot_statistics_comparison(
        experiments,
        os.path.join(output_dir, "statistics_comparison.png")
    )

    plot_ef_construction_effect(
        experiments,
        os.path.join(output_dir, "ef_construction_effect.png")
    )

    plot_dataset_comparison(
        experiments,
        os.path.join(output_dir, "dataset_comparison.png")
    )

    plot_tail_analysis(
        experiments,
        os.path.join(output_dir, "tail_analysis.png")
    )

    plot_histogram_overlay(
        experiments,
        os.path.join(output_dir, "histogram_overlay.png"),
        max_degree=60
    )

    print(f"\nAll plots saved to: {output_dir}")


def generate_all_plots(
    input_file: str,
    output_dir: str,
    experiment_keys: Optional[List[str]] = None,
    pattern: Optional[str] = None,
) -> None:
    """Generate all analysis plots (legacy function for single file)."""
    all_data = load_distribution_data(input_file)
    experiments = filter_experiments(all_data, experiment_keys, pattern)
    experiments = [e for e in experiments if e['distribution']]
    generate_all_plots_from_data(experiments, output_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize in-degree distributions from graph indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots from a single file
  python plot_indegree_analysis.py -i ../metrics/indegree_stats_mnist.json

  # Combine multiple dataset files for comparison
  python plot_indegree_analysis.py \\
    -i ../metrics/indegree_stats_mnist.json \\
       ../metrics/indegree_stats_gist.json \\
       ../metrics/indegree_stats_sift.json \\
    -o ../metrics/all_datasets_comparison

  # List all experiments across multiple files
  python plot_indegree_analysis.py \\
    -i ../metrics/indegree_stats_*.json --list

  # Filter by pattern (e.g., only ef=200 experiments)
  python plot_indegree_analysis.py \\
    -i ../metrics/indegree_stats_mnist.json \\
       ../metrics/indegree_stats_gist.json \\
    --pattern ".*ef200.*"

  # Just print statistics without generating plots
  python plot_indegree_analysis.py \\
    -i ../metrics/indegree_stats_mnist.json --stats-only

  # Compare specific experiments by key
  python plot_indegree_analysis.py \\
    -i ../metrics/indegree_stats_mnist.json \\
       ../metrics/indegree_stats_gist.json \\
    -e mnist_flatnav-native_M32_ef100 gist_flatnav-native_M32_ef100
        """
    )

    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Path(s) to in-degree distribution JSON file(s). Multiple files can be specified."
    )

    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for plots (default: same as input file with _plots suffix)"
    )

    parser.add_argument(
        "--experiments", "-e",
        nargs="+",
        help="Specific experiment keys to analyze"
    )

    parser.add_argument(
        "--pattern", "-p",
        help="Regex pattern to filter experiments"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics table, don't generate plots"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Handle single or multiple input files
    input_files = args.input if isinstance(args.input, list) else [args.input]

    # Check if files exist
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        for f in missing_files:
            print(f"Error: Input file not found: {f}")
        return 1

    # Load data from all input files
    if len(input_files) == 1:
        all_data = load_distribution_data(input_files[0])
    else:
        all_data = load_multiple_files(input_files)
        print(f"Loaded data from {len(input_files)} files")

    # List experiments
    if args.list:
        print("\nAvailable experiments:")
        print("=" * 90)
        for exp in all_data:
            n_degrees = len(exp['distribution'])
            print(f"  {exp['experiment_key']:<45} M={exp['node_links']:<4} ef={exp['ef_construction']:<4} ({n_degrees} unique degrees)")
        print("=" * 90)
        return 0

    # Filter experiments
    experiments = filter_experiments(all_data, args.experiments, args.pattern)
    experiments = [e for e in experiments if e['distribution']]

    if not experiments:
        print("No valid experiments found")
        return 1

    # Stats only mode
    if args.stats_only:
        print_statistics_table(experiments)
        return 0

    # Determine output directory
    output_dir = args.output_dir
    if not output_dir:
        base_dir = os.path.dirname(os.path.abspath(input_files[0]))
        output_dir = os.path.join(base_dir, "indegree_plots")

    # Generate all plots
    generate_all_plots_from_data(
        experiments,
        output_dir,
    )

    return 0


if __name__ == "__main__":
    exit(main())
