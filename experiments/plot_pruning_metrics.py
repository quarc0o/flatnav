"""
Plot Pruning Experiment Metrics - Grouped Bar Chart

Generates a grouped bar chart comparing baseline, hub pruned, and random pruned
across multiple datasets. Uses only the first entry from each experiment.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Path to metrics file
METRICS_FILE = os.path.join(os.path.dirname(__file__), "..", "metrics", "metrics.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def load_metrics(filepath: str) -> dict:
    """Load metrics from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_datasets(metrics: dict) -> dict:
    """
    Extract and group data by dataset name.

    Returns a dict like:
    {
        "gist": {"baseline": {...}, "hub_pruned": {...}, "random_pruned": {...}},
        "sift": {"baseline": {...}, "hub_pruned": {...}, "random_pruned": {...}},
        ...
    }
    """
    datasets = {}

    for key, values in metrics.items():
        # Extract dataset name and method type
        # Format: {dataset-name}_baseline, {dataset-name}_hub_pruned, {dataset-name}_random_pruned
        if "_baseline" in key:
            dataset_name = key.rsplit("_baseline", 1)[0]
            method = "baseline"
        elif "_hub_pruned" in key:
            dataset_name = key.rsplit("_hub_pruned", 1)[0]
            method = "hub_pruned"
        elif "_random_pruned" in key:
            dataset_name = key.rsplit("_random_pruned", 1)[0]
            method = "random_pruned"
        else:
            continue

        if dataset_name not in datasets:
            datasets[dataset_name] = {}

        # Use only the first entry
        if values:
            datasets[dataset_name][method] = values[0]

    return datasets


def plot_grouped_bar_chart(metrics: dict, output_dir: str):
    """Plot grouped bar chart showing percentage recall drop from baseline."""
    datasets = extract_datasets(metrics)

    # Filter to datasets that have all 3 methods
    complete_datasets = {
        name: data for name, data in datasets.items()
        if all(m in data for m in ["baseline", "hub_pruned", "random_pruned"])
    }

    if not complete_datasets:
        print("No complete datasets found (need baseline, hub_pruned, and random_pruned)")
        return

    dataset_names = list(complete_datasets.keys())
    # Shorten names for display
    display_names = [name.replace("-pruning-test-random", "").upper() for name in dataset_names]

    # Extract recall values
    baseline_recalls = [complete_datasets[d]["baseline"]["recall"] for d in dataset_names]
    hub_recalls = [complete_datasets[d]["hub_pruned"]["recall"] for d in dataset_names]
    random_recalls = [complete_datasets[d]["random_pruned"]["recall"] for d in dataset_names]

    # Calculate percentage recall drop from baseline
    hub_drops = [(baseline_recalls[i] - hub_recalls[i]) / baseline_recalls[i] * 100
                 for i in range(len(dataset_names))]
    random_drops = [(baseline_recalls[i] - random_recalls[i]) / baseline_recalls[i] * 100
                    for i in range(len(dataset_names))]

    # Create grouped bar chart
    x = np.arange(len(dataset_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, hub_drops, width, label="Hub Pruned", color="#e74c3c")
    bars2 = ax.bar(x + width/2, random_drops, width, label="Random Pruned", color="#3498db")

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Recall Drop from Baseline (%)", fontsize=12)
    ax.set_title("Recall Degradation: Hub Pruned vs Random Pruned (2% nodes pruned)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "pruning_recall_drop.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")

    # Print summary
    print("\nSummary:")
    for i, name in enumerate(dataset_names):
        print(f"\n{display_names[i]}:")
        print(f"  Baseline:      {baseline_recalls[i]:.4f}")
        print(f"  Hub Pruned:    {hub_recalls[i]:.4f} (drop: {hub_drops[i]:.1f}%)")
        print(f"  Random Pruned: {random_recalls[i]:.4f} (drop: {random_drops[i]:.1f}%)")


def plot_qps_change(metrics: dict, output_dir: str):
    """Plot grouped bar chart showing percentage QPS change from baseline."""
    datasets = extract_datasets(metrics)

    # Filter to datasets that have all 3 methods
    complete_datasets = {
        name: data for name, data in datasets.items()
        if all(m in data for m in ["baseline", "hub_pruned", "random_pruned"])
    }

    if not complete_datasets:
        print("No complete datasets found (need baseline, hub_pruned, and random_pruned)")
        return

    dataset_names = list(complete_datasets.keys())
    # Shorten names for display
    display_names = [name.replace("-pruning-test-random", "").upper() for name in dataset_names]

    # Extract QPS values
    baseline_qps = [complete_datasets[d]["baseline"]["qps"] for d in dataset_names]
    hub_qps = [complete_datasets[d]["hub_pruned"]["qps"] for d in dataset_names]
    random_qps = [complete_datasets[d]["random_pruned"]["qps"] for d in dataset_names]

    # Calculate percentage QPS change from baseline (positive = faster)
    hub_change = [(hub_qps[i] - baseline_qps[i]) / baseline_qps[i] * 100
                  for i in range(len(dataset_names))]
    random_change = [(random_qps[i] - baseline_qps[i]) / baseline_qps[i] * 100
                     for i in range(len(dataset_names))]

    # Create grouped bar chart
    x = np.arange(len(dataset_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, hub_change, width, label="Hub Pruned", color="#e74c3c")
    bars2 = ax.bar(x + width/2, random_change, width, label="Random Pruned", color="#3498db")

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 3 if height >= 0 else -3
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha='center', va=va, fontsize=9)

    add_labels(bars1)
    add_labels(bars2)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("QPS Change from Baseline (%)", fontsize=12)
    ax.set_title("QPS Change: Hub Pruned vs Random Pruned (2% nodes pruned)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "pruning_qps_change.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")

    # Print summary
    print("\nQPS Summary:")
    for i, name in enumerate(dataset_names):
        print(f"\n{display_names[i]}:")
        print(f"  Baseline:      {baseline_qps[i]:.1f} QPS")
        print(f"  Hub Pruned:    {hub_qps[i]:.1f} QPS ({hub_change[i]:+.1f}%)")
        print(f"  Random Pruned: {random_qps[i]:.1f} QPS ({random_change[i]:+.1f}%)")


def main():
    # Load metrics
    metrics = load_metrics(METRICS_FILE)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating plots...")
    plot_grouped_bar_chart(metrics, OUTPUT_DIR)
    plot_qps_change(metrics, OUTPUT_DIR)
    print("\nDone!")


if __name__ == "__main__":
    main()
