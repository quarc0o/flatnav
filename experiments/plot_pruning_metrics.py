"""
Plot Pruning Experiment Metrics - Grouped Bar Chart

Generates grouped bar charts comparing baseline, hub pruned, and random pruned
across multiple datasets for each pruning percentage (2%, 5%, 10%).
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Default paths
DEFAULT_METRICS_FILE = os.path.join(os.path.dirname(__file__), "..", "metrics", "pruning_metrics.json")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def load_metrics(filepath: str) -> dict:
    """Load metrics from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_datasets_by_percentage(metrics: dict) -> dict:
    """
    Extract and group data by dataset name and pruning percentage.

    Returns a dict like:
    {
        2: {
            "mnist-784": {"baseline": {...}, "hub_pruned": {...}, "random_pruned": {...}},
            "gist-960": {"baseline": {...}, "hub_pruned": {...}, "random_pruned": {...}},
        },
        5: {...},
        10: {...},
    }
    """
    data_by_pct = {}

    for key, values in metrics.items():
        # Format: {dataset-name}_{method}_{pct}pct
        # e.g., mnist-784_baseline_2pct, mnist-784_hub_pruned_5pct

        if not key.endswith("pct"):
            continue

        # Extract pruning percentage
        parts = key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        pct_str = parts[1].replace("pct", "")
        try:
            pct = int(pct_str)
        except ValueError:
            continue

        # Extract dataset name and method
        prefix = parts[0]  # e.g., mnist-784_baseline
        if "_baseline" in prefix:
            dataset_name = prefix.rsplit("_baseline", 1)[0]
            method = "baseline"
        elif "_hub_pruned" in prefix:
            dataset_name = prefix.rsplit("_hub_pruned", 1)[0]
            method = "hub_pruned"
        elif "_random_pruned" in prefix:
            dataset_name = prefix.rsplit("_random_pruned", 1)[0]
            method = "random_pruned"
        else:
            continue

        if pct not in data_by_pct:
            data_by_pct[pct] = {}
        if dataset_name not in data_by_pct[pct]:
            data_by_pct[pct][dataset_name] = {}

        # Use only the first entry (or average if multiple ef_search values)
        if values:
            data_by_pct[pct][dataset_name][method] = values[0]

    return data_by_pct


def plot_recall_drop_by_percentage(metrics: dict, output_dir: str):
    """Plot grouped bar chart showing percentage recall drop from baseline for each pruning %."""
    data_by_pct = extract_datasets_by_percentage(metrics)

    if not data_by_pct:
        print("No data found with the expected key format (e.g., mnist-784_baseline_2pct)")
        return

    for pct, datasets in sorted(data_by_pct.items()):
        # Filter to datasets that have all 3 methods
        complete_datasets = {
            name: data for name, data in datasets.items()
            if all(m in data for m in ["baseline", "hub_pruned", "random_pruned"])
        }

        if not complete_datasets:
            print(f"No complete datasets found for {pct}% pruning")
            continue

        dataset_names = list(complete_datasets.keys())
        display_names = [name.upper() for name in dataset_names]

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
                va = 'bottom' if height >= 0 else 'top'
                offset = 3 if height >= 0 else -3
                ax.annotate(f'{height:.2f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, offset),
                            textcoords="offset points",
                            ha='center', va=va, fontsize=9)

        add_labels(bars1)
        add_labels(bars2)

        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel("Recall Drop from Baseline (%)", fontsize=12)
        ax.set_title(f"Recall Degradation: Hub Pruned vs Random Pruned ({pct}% nodes pruned)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f"pruning_recall_drop_{pct}pct.png")
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Saved: {output_file}")

        # Print summary
        print(f"\nSummary for {pct}% pruning:")
        for i, name in enumerate(dataset_names):
            print(f"\n  {display_names[i]}:")
            print(f"    Baseline:      {baseline_recalls[i]:.4f}")
            print(f"    Hub Pruned:    {hub_recalls[i]:.4f} (drop: {hub_drops[i]:.2f}%)")
            print(f"    Random Pruned: {random_recalls[i]:.4f} (drop: {random_drops[i]:.2f}%)")


def plot_qps_change_by_percentage(metrics: dict, output_dir: str):
    """Plot grouped bar chart showing percentage QPS change from baseline for each pruning %."""
    data_by_pct = extract_datasets_by_percentage(metrics)

    if not data_by_pct:
        return

    for pct, datasets in sorted(data_by_pct.items()):
        # Filter to datasets that have all 3 methods
        complete_datasets = {
            name: data for name, data in datasets.items()
            if all(m in data for m in ["baseline", "hub_pruned", "random_pruned"])
        }

        if not complete_datasets:
            continue

        dataset_names = list(complete_datasets.keys())
        display_names = [name.upper() for name in dataset_names]

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
        ax.set_title(f"QPS Change: Hub Pruned vs Random Pruned ({pct}% nodes pruned)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f"pruning_qps_change_{pct}pct.png")
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Saved: {output_file}")

        # Print summary
        print(f"\nQPS Summary for {pct}% pruning:")
        for i, name in enumerate(dataset_names):
            print(f"\n  {display_names[i]}:")
            print(f"    Baseline:      {baseline_qps[i]:.1f} QPS")
            print(f"    Hub Pruned:    {hub_qps[i]:.1f} QPS ({hub_change[i]:+.1f}%)")
            print(f"    Random Pruned: {random_qps[i]:.1f} QPS ({random_change[i]:+.1f}%)")


def plot_recall_vs_pruning_percentage(metrics: dict, output_dir: str):
    """Plot line chart showing recall drop vs pruning percentage for each dataset."""
    data_by_pct = extract_datasets_by_percentage(metrics)

    if not data_by_pct:
        return

    # Collect all datasets across all percentages
    all_datasets = set()
    for pct, datasets in data_by_pct.items():
        for name, data in datasets.items():
            if all(m in data for m in ["baseline", "hub_pruned", "random_pruned"]):
                all_datasets.add(name)

    if not all_datasets:
        print("No complete datasets found")
        return

    percentages = sorted(data_by_pct.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_datasets)))

    for idx, dataset_name in enumerate(sorted(all_datasets)):
        hub_drops = []
        random_drops = []
        valid_pcts = []

        for pct in percentages:
            if dataset_name in data_by_pct[pct]:
                data = data_by_pct[pct][dataset_name]
                if all(m in data for m in ["baseline", "hub_pruned", "random_pruned"]):
                    baseline = data["baseline"]["recall"]
                    hub = data["hub_pruned"]["recall"]
                    random = data["random_pruned"]["recall"]

                    hub_drops.append((baseline - hub) / baseline * 100)
                    random_drops.append((baseline - random) / baseline * 100)
                    valid_pcts.append(pct)

        if valid_pcts:
            color = colors[idx]
            ax.plot(valid_pcts, hub_drops, 'o-', color=color,
                    label=f"{dataset_name.upper()} (Hub)", linewidth=2)
            ax.plot(valid_pcts, random_drops, 's--', color=color,
                    label=f"{dataset_name.upper()} (Random)", linewidth=2, alpha=0.7)

    ax.set_xlabel("Pruning Percentage (%)", fontsize=12)
    ax.set_ylabel("Recall Drop from Baseline (%)", fontsize=12)
    ax.set_title("Recall Degradation vs Pruning Percentage", fontsize=14)
    ax.set_xticks(percentages)
    ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "recall_vs_pruning_percentage.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot pruning experiment metrics")
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=DEFAULT_METRICS_FILE,
        help="Path to metrics JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load metrics
    print(f"Loading metrics from: {args.metrics_file}")
    metrics = load_metrics(args.metrics_file)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating plots...")
    plot_recall_drop_by_percentage(metrics, args.output_dir)
    plot_qps_change_by_percentage(metrics, args.output_dir)
    plot_recall_vs_pruning_percentage(metrics, args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
