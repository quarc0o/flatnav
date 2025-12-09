"""
Plot Hub Pruning Experiment Results

This script generates visualizations comparing the impact of pruning
hub nodes vs random nodes on search performance.
"""

import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def load_results(filepath: str) -> Dict:
    """Load experiment results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def plot_recall_degradation(
    results: Dict,
    metadata: Dict,
    output_dir: str,
    dataset_name: str
):
    """
    Plot recall degradation comparison between hub and random pruning.
    """
    fig, axes = plt.subplots(1, len(metadata["pruning_percentages"]),
                             figsize=(5 * len(metadata["pruning_percentages"]), 4),
                             sharey=True)

    if len(metadata["pruning_percentages"]) == 1:
        axes = [axes]

    ef_search_values = sorted(set(r["ef_search"] for r in results["baseline"]))

    for idx, prune_pct in enumerate(metadata["pruning_percentages"]):
        ax = axes[idx]

        baseline_recalls = []
        hub_recalls = []
        random_recalls = []

        for ef in ef_search_values:
            baseline = [r for r in results["baseline"]
                       if r["pruning_percentage"] == prune_pct and r["ef_search"] == ef]
            hub = [r for r in results["hub_pruned"]
                  if r["pruning_percentage"] == prune_pct and r["ef_search"] == ef]
            random = [r for r in results["random_pruned"]
                     if r["pruning_percentage"] == prune_pct and r["ef_search"] == ef]

            if baseline and hub and random:
                baseline_recalls.append(baseline[0]["recall"])
                hub_recalls.append(hub[0]["recall"])
                random_recalls.append(random[0]["recall"])

        x = np.arange(len(ef_search_values))
        width = 0.25

        ax.bar(x - width, baseline_recalls, width, label="Baseline", color='#2ecc71')
        ax.bar(x, hub_recalls, width, label="Hub Pruned", color='#e74c3c')
        ax.bar(x + width, random_recalls, width, label="Random Pruned", color='#3498db')

        ax.set_xlabel("ef_search")
        ax.set_ylabel("Recall@100")
        ax.set_title(f"{prune_pct}% Nodes Pruned")
        ax.set_xticks(x)
        ax.set_xticklabels(ef_search_values)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{dataset_name}_recall_comparison.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_recall_drop_percentage(
    results: Dict,
    metadata: Dict,
    output_dir: str,
    dataset_name: str
):
    """
    Plot percentage recall drop for hub vs random pruning.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ef_search_values = sorted(set(r["ef_search"] for r in results["baseline"]))

    # Use middle ef_search value for this plot
    target_ef = ef_search_values[len(ef_search_values) // 2]

    hub_drops = []
    random_drops = []
    prune_pcts = metadata["pruning_percentages"]

    for prune_pct in prune_pcts:
        baseline = [r for r in results["baseline"]
                   if r["pruning_percentage"] == prune_pct and r["ef_search"] == target_ef]
        hub = [r for r in results["hub_pruned"]
              if r["pruning_percentage"] == prune_pct and r["ef_search"] == target_ef]
        random = [r for r in results["random_pruned"]
                 if r["pruning_percentage"] == prune_pct and r["ef_search"] == target_ef]

        if baseline and hub and random:
            base_recall = baseline[0]["recall"]
            hub_drop = (base_recall - hub[0]["recall"]) / base_recall * 100
            random_drop = (base_recall - random[0]["recall"]) / base_recall * 100
            hub_drops.append(hub_drop)
            random_drops.append(random_drop)

    x = np.arange(len(prune_pcts))
    width = 0.35

    bars1 = ax.bar(x - width/2, hub_drops, width, label="Hub Pruned", color='#e74c3c')
    bars2 = ax.bar(x + width/2, random_drops, width, label="Random Pruned", color='#3498db')

    ax.set_xlabel("Pruning Percentage (%)")
    ax.set_ylabel("Recall Drop (%)")
    ax.set_title(f"Recall Degradation: Hub vs Random Pruning (ef_search={target_ef})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in prune_pcts])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{dataset_name}_recall_drop.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_qps_comparison(
    results: Dict,
    metadata: Dict,
    output_dir: str,
    dataset_name: str
):
    """
    Plot QPS comparison between different pruning strategies.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    prune_pcts = metadata["pruning_percentages"]
    ef_search_values = sorted(set(r["ef_search"] for r in results["baseline"]))

    # Use the lowest ef_search for QPS comparison
    target_ef = ef_search_values[0]

    baseline_qps = []
    hub_qps = []
    random_qps = []

    for prune_pct in prune_pcts:
        baseline = [r for r in results["baseline"]
                   if r["pruning_percentage"] == prune_pct and r["ef_search"] == target_ef]
        hub = [r for r in results["hub_pruned"]
              if r["pruning_percentage"] == prune_pct and r["ef_search"] == target_ef]
        random = [r for r in results["random_pruned"]
                 if r["pruning_percentage"] == prune_pct and r["ef_search"] == target_ef]

        if baseline and hub and random:
            baseline_qps.append(baseline[0]["qps"])
            hub_qps.append(hub[0]["qps"])
            random_qps.append(random[0]["qps"])

    x = np.arange(len(prune_pcts))
    width = 0.25

    ax.bar(x - width, baseline_qps, width, label="Baseline", color='#2ecc71')
    ax.bar(x, hub_qps, width, label="Hub Pruned", color='#e74c3c')
    ax.bar(x + width, random_qps, width, label="Random Pruned", color='#3498db')

    ax.set_xlabel("Pruning Percentage (%)")
    ax.set_ylabel("Queries Per Second")
    ax.set_title(f"QPS Comparison (ef_search={target_ef})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in prune_pcts])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{dataset_name}_qps_comparison.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_recall_vs_distance_computations(
    results: Dict,
    metadata: Dict,
    output_dir: str,
    dataset_name: str
):
    """
    Plot recall vs distance computations for each pruning strategy.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use the first pruning percentage for this plot
    prune_pct = metadata["pruning_percentages"][-1]  # Use largest pruning %

    baseline_data = [(r["distance_computations"], r["recall"])
                    for r in results["baseline"] if r["pruning_percentage"] == prune_pct]
    hub_data = [(r["distance_computations"], r["recall"])
               for r in results["hub_pruned"] if r["pruning_percentage"] == prune_pct]
    random_data = [(r["distance_computations"], r["recall"])
                  for r in results["random_pruned"] if r["pruning_percentage"] == prune_pct]

    if baseline_data:
        baseline_data.sort()
        ax.plot([d[0] for d in baseline_data], [d[1] for d in baseline_data],
               'o-', label="Baseline", color='#2ecc71', markersize=8)

    if hub_data:
        hub_data.sort()
        ax.plot([d[0] for d in hub_data], [d[1] for d in hub_data],
               's-', label="Hub Pruned", color='#e74c3c', markersize=8)

    if random_data:
        random_data.sort()
        ax.plot([d[0] for d in random_data], [d[1] for d in random_data],
               '^-', label="Random Pruned", color='#3498db', markersize=8)

    ax.set_xlabel("Distance Computations")
    ax.set_ylabel("Recall@100")
    ax.set_title(f"Recall vs Distance Computations ({prune_pct}% Pruning)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"{dataset_name}_recall_vs_dist.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def generate_summary_table(results: Dict, metadata: Dict) -> str:
    """Generate a text summary table of the results."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Hub Pruning Experiment Summary: {metadata['dataset_name']}")
    lines.append("=" * 80)
    lines.append(f"Distance Type: {metadata['distance_type']}")
    lines.append(f"Max Edges Per Node: {metadata['max_edges_per_node']}")
    lines.append(f"Hub Identification: {metadata['hub_identification_method']}")
    lines.append(f"Alpha (edge removal fraction): {metadata['alpha']}")
    lines.append("")

    ef_search_values = sorted(set(r["ef_search"] for r in results["baseline"]))

    for prune_pct in metadata["pruning_percentages"]:
        lines.append(f"\n{'-'*60}")
        lines.append(f"Pruning {prune_pct}% of nodes")
        lines.append(f"{'-'*60}")

        for ef in ef_search_values:
            baseline = [r for r in results["baseline"]
                       if r["pruning_percentage"] == prune_pct and r["ef_search"] == ef]
            hub = [r for r in results["hub_pruned"]
                  if r["pruning_percentage"] == prune_pct and r["ef_search"] == ef]
            random = [r for r in results["random_pruned"]
                     if r["pruning_percentage"] == prune_pct and r["ef_search"] == ef]

            if baseline and hub and random:
                base_recall = baseline[0]["recall"]
                hub_recall = hub[0]["recall"]
                random_recall = random[0]["recall"]

                hub_drop = (base_recall - hub_recall) / base_recall * 100
                random_drop = (base_recall - random_recall) / base_recall * 100

                lines.append(f"\nef_search={ef}:")
                lines.append(f"  Baseline:      Recall={base_recall:.4f}")
                lines.append(f"  Hub Pruned:    Recall={hub_recall:.4f} (drop: {hub_drop:.2f}%)")
                lines.append(f"  Random Pruned: Recall={random_recall:.4f} (drop: {random_drop:.2f}%)")

                if hub_drop > random_drop:
                    lines.append(f"  -> Hub pruning causes MORE degradation (SUPPORTS hypothesis)")
                else:
                    lines.append(f"  -> Random pruning causes MORE degradation (AGAINST hypothesis)")

    return "\n".join(lines)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot hub pruning experiment results")
    parser.add_argument(
        "--results-file",
        required=True,
        help="Path to the experiment results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="../plots",
        help="Directory to save plots"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load results
    data = load_results(args.results_file)
    results = data["results"]
    metadata = data["metadata"]
    dataset_name = metadata["dataset_name"]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate plots
    print(f"\nGenerating plots for {dataset_name}...")

    plot_recall_degradation(results, metadata, args.output_dir, dataset_name)
    plot_recall_drop_percentage(results, metadata, args.output_dir, dataset_name)
    plot_qps_comparison(results, metadata, args.output_dir, dataset_name)
    plot_recall_vs_distance_computations(results, metadata, args.output_dir, dataset_name)

    # Generate and print summary
    summary = generate_summary_table(results, metadata)
    print("\n" + summary)

    # Save summary to file
    summary_file = os.path.join(args.output_dir, f"{dataset_name}_summary.txt")
    with open(summary_file, "w") as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
