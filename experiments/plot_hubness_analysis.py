#!/usr/bin/env python3
"""
Plot and analyze hubness/skewness metrics from dataset analysis.

Usage:
    python plot_hubness_analysis.py -i ../metrics/hubness_mnist.json
    python plot_hubness_analysis.py -i ../metrics/hubness_*.json -o ../metrics/hubness_plots
    python plot_hubness_analysis.py -i ../metrics/hubness_mnist.json --stats-only
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_hubness_data(filepath: str) -> Dict:
    """Load hubness data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_multiple_files(filepaths: List[str]) -> List[Dict]:
    """Load hubness data from multiple JSON files."""
    all_data = []
    for filepath in filepaths:
        if os.path.exists(filepath):
            data = load_hubness_data(filepath)
            all_data.append(data)
        else:
            print(f"Warning: File not found: {filepath}")
    return all_data


def plot_nk_distribution(
    data_list: List[Dict],
    output_path: str,
    title: str = "N_k Score Distribution (Hubness)",
    max_nk: Optional[int] = None,
) -> None:
    """
    Plot N_k score distributions for one or more datasets.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))

    for idx, data in enumerate(data_list):
        dataset_name = data['dataset_name']

        for result in data['results']:
            k = result['k']
            stats = result['statistics']
            distribution = stats['distribution']

            # Convert to arrays
            sorted_items = sorted([(int(nk), pct) for nk, pct in distribution.items()])
            nk_values = np.array([item[0] for item in sorted_items])
            percentages = np.array([item[1] for item in sorted_items])

            if max_nk:
                mask = nk_values <= max_nk
                nk_values = nk_values[mask]
                percentages = percentages[mask]

            label = f"{dataset_name} (k={k}, skew={stats['skewness']:.2f})"
            ax.plot(nk_values, percentages, marker='.', markersize=4,
                    label=label, color=colors[idx], linewidth=1.5, alpha=0.8)

    ax.set_xlabel("N_k Score (Occurrence Count)", fontsize=12)
    ax.set_ylabel("Percentage of Points (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_nk_distribution_log(
    data_list: List[Dict],
    output_path: str,
    title: str = "N_k Score Distribution (Log Scale)",
) -> None:
    """
    Plot N_k score distributions with log scale on y-axis.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))

    for idx, data in enumerate(data_list):
        dataset_name = data['dataset_name']

        for result in data['results']:
            k = result['k']
            stats = result['statistics']
            distribution = stats['distribution']

            sorted_items = sorted([(int(nk), pct) for nk, pct in distribution.items()])
            nk_values = np.array([item[0] for item in sorted_items])
            percentages = np.array([item[1] for item in sorted_items])

            label = f"{dataset_name} (k={k})"
            ax.plot(nk_values, percentages, marker='.', markersize=4,
                    label=label, color=colors[idx], linewidth=1.5, alpha=0.8)

    ax.set_xlabel("N_k Score (Occurrence Count)", fontsize=12)
    ax.set_ylabel("Percentage of Points (log scale)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_cdf(
    data_list: List[Dict],
    output_path: str,
    title: str = "Cumulative N_k Distribution",
) -> None:
    """
    Plot cumulative distribution of N_k scores.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))

    for idx, data in enumerate(data_list):
        dataset_name = data['dataset_name']

        for result in data['results']:
            k = result['k']
            stats = result['statistics']
            distribution = stats['distribution']

            sorted_items = sorted([(int(nk), pct) for nk, pct in distribution.items()])
            nk_values = np.array([item[0] for item in sorted_items])
            percentages = np.array([item[1] for item in sorted_items])

            cdf = np.cumsum(percentages)

            label = f"{dataset_name} (k={k})"
            ax.plot(nk_values, cdf, label=label, color=colors[idx], linewidth=2, alpha=0.8)

    ax.set_xlabel("N_k Score", fontsize=12)
    ax.set_ylabel("Cumulative Percentage (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Median')
    ax.axhline(y=95, color='gray', linestyle=':', alpha=0.5, label='95th percentile')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_statistics_comparison(
    data_list: List[Dict],
    output_path: str,
    title: str = "Hubness Statistics Comparison",
) -> None:
    """
    Create bar charts comparing hubness statistics across datasets.
    """
    # Collect statistics
    labels = []
    skewness_vals = []
    gini_vals = []
    hub_pct_vals = []
    antihub_pct_vals = []

    for data in data_list:
        dataset_name = data['dataset_name']
        for result in data['results']:
            k = result['k']
            stats = result['statistics']

            labels.append(f"{dataset_name}\n(k={k})")
            skewness_vals.append(stats['skewness'])
            gini_vals.append(stats['gini_coefficient'])
            hub_pct_vals.append(stats['hub_percentage'])
            antihub_pct_vals.append(stats['antihub_percentage'])

    if not labels:
        print("No data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    x = np.arange(len(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    # Skewness
    bars = axes[0].bar(x, skewness_vals, color=colors, alpha=0.8)
    axes[0].set_ylabel("Skewness")
    axes[0].set_title("Skewness (Higher = More Hubness)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, skewness_vals):
        axes[0].annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    # Gini coefficient
    bars = axes[1].bar(x, gini_vals, color=colors, alpha=0.8)
    axes[1].set_ylabel("Gini Coefficient")
    axes[1].set_title("Gini Coefficient (Inequality in N_k)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, gini_vals):
        axes[1].annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    # Hub percentage
    bars = axes[2].bar(x, hub_pct_vals, color=colors, alpha=0.8)
    axes[2].set_ylabel("Hub Percentage (%)")
    axes[2].set_title("Percentage of Hubs (N_k > mean + 2σ)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, hub_pct_vals):
        axes[2].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    # Anti-hub percentage
    bars = axes[3].bar(x, antihub_pct_vals, color=colors, alpha=0.8)
    axes[3].set_ylabel("Anti-hub Percentage (%)")
    axes[3].set_title("Percentage of Anti-hubs (N_k = 0)")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    axes[3].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, antihub_pct_vals):
        axes[3].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_k_effect(
    data_list: List[Dict],
    output_path: str,
    title: str = "Effect of k on Hubness Metrics",
) -> None:
    """
    Plot how hubness metrics change with different k values.
    Only for datasets with multiple k values.
    """
    # Find datasets with multiple k values
    multi_k_datasets = [d for d in data_list if len(d['results']) > 1]

    if not multi_k_datasets:
        print("No datasets with multiple k values found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(multi_k_datasets)))

    for idx, data in enumerate(multi_k_datasets):
        dataset_name = data['dataset_name']

        k_vals = [r['k'] for r in data['results']]
        skewness_vals = [r['statistics']['skewness'] for r in data['results']]
        gini_vals = [r['statistics']['gini_coefficient'] for r in data['results']]
        antihub_vals = [r['statistics']['antihub_percentage'] for r in data['results']]

        # Sort by k
        sorted_indices = np.argsort(k_vals)
        k_vals = np.array(k_vals)[sorted_indices]
        skewness_vals = np.array(skewness_vals)[sorted_indices]
        gini_vals = np.array(gini_vals)[sorted_indices]
        antihub_vals = np.array(antihub_vals)[sorted_indices]

        axes[0].plot(k_vals, skewness_vals, marker='o', label=dataset_name,
                    color=colors[idx], linewidth=2)
        axes[1].plot(k_vals, gini_vals, marker='o', label=dataset_name,
                    color=colors[idx], linewidth=2)
        axes[2].plot(k_vals, antihub_vals, marker='o', label=dataset_name,
                    color=colors[idx], linewidth=2)

    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Skewness")
    axes[0].set_title("Skewness vs k")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Gini Coefficient")
    axes[1].set_title("Gini Coefficient vs k")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Anti-hub %")
    axes[2].set_title("Anti-hub Percentage vs k")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_statistics_table(data_list: List[Dict]) -> None:
    """Print a formatted statistics table."""
    print("\n" + "=" * 110)
    print("HUBNESS STATISTICS")
    print("=" * 110)

    header = f"{'Dataset':<25} {'k':>4} {'Skewness':>10} {'Gini':>8} {'Hubs %':>8} {'Anti-hubs %':>12} {'Max N_k':>8}"
    print(header)
    print("-" * 110)

    for data in data_list:
        dataset_name = data['dataset_name']
        for result in data['results']:
            k = result['k']
            stats = result['statistics']

            row = (
                f"{dataset_name:<25} "
                f"{k:>4} "
                f"{stats['skewness']:>10.4f} "
                f"{stats['gini_coefficient']:>8.4f} "
                f"{stats['hub_percentage']:>8.2f} "
                f"{stats['antihub_percentage']:>12.2f} "
                f"{stats['max_nk']:>8}"
            )
            print(row)

    print("=" * 110 + "\n")


def generate_all_plots(
    data_list: List[Dict],
    output_dir: str,
) -> None:
    """Generate all hubness analysis plots."""
    if not data_list:
        print("No data to plot")
        return

    print(f"Generating plots for {len(data_list)} dataset(s)...")

    os.makedirs(output_dir, exist_ok=True)

    # Print statistics table
    print_statistics_table(data_list)

    # Generate plots
    plot_nk_distribution(
        data_list,
        os.path.join(output_dir, "nk_distribution.png"),
        max_nk=60
    )

    plot_nk_distribution_log(
        data_list,
        os.path.join(output_dir, "nk_distribution_log.png")
    )

    plot_cdf(
        data_list,
        os.path.join(output_dir, "nk_cdf.png")
    )

    plot_statistics_comparison(
        data_list,
        os.path.join(output_dir, "statistics_comparison.png")
    )

    plot_k_effect(
        data_list,
        os.path.join(output_dir, "k_effect.png")
    )

    print(f"\nAll plots saved to: {output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot and analyze hubness metrics from dataset analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset
  python plot_hubness_analysis.py -i ../metrics/hubness_mnist.json

  # Multiple datasets
  python plot_hubness_analysis.py \\
    -i ../metrics/hubness_mnist.json \\
       ../metrics/hubness_gist.json \\
       ../metrics/hubness_sift.json \\
    -o ../metrics/hubness_comparison

  # Just print statistics
  python plot_hubness_analysis.py \\
    -i ../metrics/hubness_mnist.json --stats-only

  # List available data
  python plot_hubness_analysis.py \\
    -i ../metrics/hubness_*.json --list
        """
    )

    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="Path(s) to hubness JSON file(s)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for plots (default: same as first input file)"
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics table, don't generate plots"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load all input files
    input_files = args.input if isinstance(args.input, list) else [args.input]

    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        for f in missing_files:
            print(f"Error: File not found: {f}")
        return 1

    data_list = load_multiple_files(input_files)

    if not data_list:
        print("No valid data found")
        return 1

    # List mode
    if args.list:
        print("\nAvailable datasets:")
        print("=" * 80)
        for data in data_list:
            print(f"  {data['dataset_name']:<30} dim={data['dimensionality']:<5} n={data['n_points']}")
            for result in data['results']:
                k = result['k']
                skew = result['statistics']['skewness']
                print(f"    k={k}: skewness={skew:.4f}")
        print("=" * 80)
        return 0

    # Stats only mode
    if args.stats_only:
        print_statistics_table(data_list)
        return 0

    # Determine output directory
    output_dir = args.output_dir
    if not output_dir:
        base_dir = os.path.dirname(os.path.abspath(input_files[0]))
        output_dir = os.path.join(base_dir, "hubness_plots")

    # Generate all plots
    generate_all_plots(data_list, output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
