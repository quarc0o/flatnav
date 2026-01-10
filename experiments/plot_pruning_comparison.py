"""
Plot pruning comparison results.

This script visualizes the hub pruning experiment results, comparing:
- Baseline (no pruning)
- Hub pruned (top X% by in-degree)
- Random pruned (random X% of nodes)
- Anti-hub pruned (bottom X% by in-degree)

Generates:
1. Recall vs QPS (Pareto curve) - classic ANN benchmark view
2. Recall vs ef_search - shows scaling behavior
3. Recall degradation bar chart - summarizes impact
"""

import json
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Dict, List, Tuple

# Use colorblind-friendly style
plt.style.use('tableau-colorblind10')

# Plot styling - distinct, high contrast colors
COLORS = {
    "baseline": "#2E8B57",      # sea green - stable/reference
    "hub_pruned": "#DC143C",    # crimson red - most impact
    "random_pruned": "#4169E1", # royal blue - neutral
    "anti_hub_pruned": "#FF8C00" # dark orange - least impact
}

LABELS = {
    "baseline": "Baseline",
    "hub_pruned": "Hub pruned",
    "random_pruned": "Random pruned",
    "anti_hub_pruned": "Anti-hub pruned"
}

MARKERS = {
    "baseline": "o",
    "hub_pruned": "s",
    "random_pruned": "^",
    "anti_hub_pruned": "D"
}

# Font sizes
TITLE_FONTSIZE = 24
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12
SUPLABEL_FONTSIZE = 28

# Output settings
DPI = 400


def load_metrics(metrics_file: str) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_file, "r") as f:
        return json.load(f)


def extract_all_datasets(metrics: Dict) -> List[Tuple[str, float]]:
    """
    Extract all unique (dataset_name, pruning_percentage) pairs from metric keys.

    Returns list of (dataset_name, pruning_pct) tuples.
    """
    datasets = set()

    for key in metrics.keys():
        # Keys are like "mnist-784_baseline_5pct" or "nytimes-256_hub_pruned_5pct"
        # We need to extract dataset name and pruning percentage

        if "pct" not in key:
            continue

        # Extract pruning percentage from end
        pct_part = key.split("_")[-1]  # "5pct"
        pruning_pct = float(pct_part.replace("pct", ""))

        # Extract experiment type (baseline, hub_pruned, etc.)
        # and dataset name
        # IMPORTANT: Check longer patterns first to avoid substring matches
        # e.g., "anti_hub_pruned" must be checked before "hub_pruned"
        experiment_types = ["anti_hub_pruned", "random_pruned", "hub_pruned", "baseline"]

        for exp_type in experiment_types:
            pattern = f"_{exp_type}_{int(pruning_pct)}pct"
            if pattern in key:
                dataset_name = key.replace(pattern, "")
                datasets.add((dataset_name, pruning_pct))
                break

    return sorted(list(datasets))


def get_experiment_data(metrics: Dict, dataset_name: str, pruning_pct: float) -> Dict[str, List[Dict]]:
    """
    Extract data for each experiment type.

    Returns dict mapping experiment_type -> list of results sorted by ef_search.
    """
    experiment_types = ["baseline", "hub_pruned", "random_pruned", "anti_hub_pruned"]
    data = {}

    for exp_type in experiment_types:
        key = f"{dataset_name}_{exp_type}_{int(pruning_pct)}pct"
        if key in metrics:
            # Sort by ef_search
            sorted_results = sorted(metrics[key], key=lambda x: x["ef_search"])
            data[exp_type] = sorted_results

    return data


def plot_recall_vs_qps(data: Dict[str, List[Dict]], dataset_name: str,
                       pruning_pct: float, output_dir: str):
    """
    Plot Recall vs QPS (Pareto curve).

    This is the classic ANN benchmark visualization showing the
    recall-throughput tradeoff.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for exp_type, results in data.items():
        recalls = [r["recall"] for r in results]
        qps_values = [r["qps"] for r in results]

        ax.plot(recalls, qps_values,
                color=COLORS[exp_type],
                marker=MARKERS[exp_type],
                markersize=10,
                linewidth=2.5,
                label=LABELS[exp_type])

    ax.set_xlabel("Recall@100", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Queries per Second (QPS)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f"{dataset_name} ({pruning_pct}% pruning)", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE)
    ax.grid(True, which='both', alpha=0.3)

    # Set x-axis to show relevant range
    all_recalls = [r["recall"] for results in data.values() for r in results]
    min_recall = min(all_recalls)
    ax.set_xlim(min_recall - 0.01, 1.005)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{dataset_name}_recall_vs_qps.png")
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_recall_vs_ef_search(data: Dict[str, List[Dict]], dataset_name: str,
                              pruning_pct: float, output_dir: str):
    """
    Plot Recall vs ef_search for each method.

    Shows how recall scales with ef_search parameter.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for exp_type, results in data.items():
        ef_values = [r["ef_search"] for r in results]
        recalls = [r["recall"] for r in results]

        ax.plot(ef_values, recalls,
                color=COLORS[exp_type],
                marker=MARKERS[exp_type],
                markersize=10,
                linewidth=2.5,
                label=LABELS[exp_type])

    ax.set_xlabel("ef_search", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Recall@100", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f"{dataset_name} ({pruning_pct}% pruning)", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xscale("log")

    # Set y-axis to show relevant range
    all_recalls = [r["recall"] for results in data.values() for r in results]
    min_recall = min(all_recalls)
    ax.set_ylim(min_recall - 0.01, 1.005)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{dataset_name}_recall_vs_ef_search.png")
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_recall_degradation(data: Dict[str, List[Dict]], dataset_name: str,
                            pruning_pct: float, output_dir: str):
    """
    Plot recall degradation as bar chart.

    Shows % recall drop from baseline for each method at different ef_search values.
    """
    if "baseline" not in data:
        print("Warning: No baseline data found, skipping degradation plot")
        return

    baseline_results = {r["ef_search"]: r["recall"] for r in data["baseline"]}
    ef_values = sorted(baseline_results.keys())

    # Calculate degradation for each method
    methods = ["hub_pruned", "random_pruned", "anti_hub_pruned"]
    degradation = {method: [] for method in methods}

    for ef in ef_values:
        baseline_recall = baseline_results[ef]
        for method in methods:
            if method in data:
                method_results = {r["ef_search"]: r["recall"] for r in data[method]}
                if ef in method_results:
                    drop = (baseline_recall - method_results[ef]) / baseline_recall * 100
                    degradation[method].append(drop)
                else:
                    degradation[method].append(0)
            else:
                degradation[method].append(0)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(ef_values))
    width = 0.25

    for i, method in enumerate(methods):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, degradation[method], width,
                     label=LABELS[method], color=COLORS[method], alpha=0.9,
                     edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, degradation[method]):
            if val > 0.5:  # Only label if significant
                ax.annotate(f'{val:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=TICK_FONTSIZE - 2,
                           fontweight='bold')

    ax.set_xlabel("ef_search", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Recall Degradation (%)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f"{dataset_name} - Recall Degradation ({pruning_pct}% pruning)", fontsize=TITLE_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(ef_values, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{dataset_name}_recall_degradation.png")
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_single_dataset_on_ax(ax, data: Dict[str, List[Dict]], dataset_name: str,
                               pruning_pct: float, plot_type: str = "recall_vs_qps"):
    """
    Plot a single dataset on a given axes object.

    Args:
        ax: matplotlib axes object
        data: experiment data
        dataset_name: name of the dataset
        pruning_pct: pruning percentage
        plot_type: "recall_vs_qps" or "recall_vs_latency"
    """
    for exp_type, results in data.items():
        recalls = [r["recall"] for r in results]

        if plot_type == "recall_vs_qps":
            y_values = [r["qps"] for r in results]
        else:  # recall_vs_latency
            y_values = [r["latency_p50"] for r in results]

        ax.plot(recalls, y_values,
                color=COLORS[exp_type],
                marker=MARKERS[exp_type],
                markersize=8,
                linewidth=2,
                label=LABELS[exp_type])

    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.set_title(dataset_name, fontsize=TITLE_FONTSIZE)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE - 2, loc='best')

    # Use log scale for latency
    if plot_type == "recall_vs_latency":
        ax.set_yscale('log')

    # Set x-axis to show relevant range
    all_recalls = [r["recall"] for results in data.values() for r in results]
    if all_recalls:
        min_recall = min(all_recalls)
        ax.set_xlim(min_recall - 0.02, 1.01)


def plot_combined_figure(all_data: Dict[str, Tuple[Dict, float]], output_dir: str,
                         plot_type: str = "recall_vs_qps"):
    """
    Create a combined figure with all datasets in a 2-2-1 grid layout.

    Args:
        all_data: Dict mapping dataset_name -> (experiment_data, pruning_pct)
        output_dir: output directory for the plot
        plot_type: "recall_vs_qps" or "recall_vs_latency"
    """
    datasets = list(all_data.keys())
    num_datasets = len(datasets)

    if num_datasets == 0:
        print("Warning: No datasets to plot in combined figure")
        return

    # Create figure with gridspec for 2-2-1 layout
    fig = plt.figure(figsize=(14, 16))

    if num_datasets <= 2:
        gs = gridspec.GridSpec(2, 2)
        positions = [(0, slice(0, 2)), (0, slice(2, 4))][:num_datasets]
    elif num_datasets <= 4:
        gs = gridspec.GridSpec(4, 4)
        positions = [
            (slice(0, 2), slice(0, 2)),  # top-left
            (slice(0, 2), slice(2, 4)),  # top-right
            (slice(2, 4), slice(0, 2)),  # bottom-left
            (slice(2, 4), slice(2, 4)),  # bottom-right
        ][:num_datasets]
    else:
        # 2-2-1 layout for 5 datasets
        gs = gridspec.GridSpec(6, 4)
        positions = [
            (slice(0, 2), slice(0, 2)),  # top-left
            (slice(0, 2), slice(2, 4)),  # top-right
            (slice(2, 4), slice(0, 2)),  # middle-left
            (slice(2, 4), slice(2, 4)),  # middle-right
            (slice(4, 6), slice(1, 3)),  # bottom-center
        ]

    # Plot each dataset
    for i, dataset_name in enumerate(datasets[:5]):  # Max 5 datasets
        data, pruning_pct = all_data[dataset_name]

        if i < len(positions):
            ax = plt.subplot(gs[positions[i]])
            plot_single_dataset_on_ax(ax, data, dataset_name, pruning_pct, plot_type)

    # Add shared axis labels
    if plot_type == "recall_vs_qps":
        fig.supylabel('Queries per Second (QPS)', fontsize=SUPLABEL_FONTSIZE, x=0.02)
        suffix = "qps"
    else:
        fig.supylabel('P50 Latency (ms)', fontsize=SUPLABEL_FONTSIZE, x=0.02)
        suffix = "latency"

    fig.supxlabel('Recall@100', fontsize=SUPLABEL_FONTSIZE, y=0.02)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])

    # Get pruning percentage from first dataset for filename
    first_pct = list(all_data.values())[0][1]
    output_path = os.path.join(output_dir, f"combined_pruning_{suffix}_{int(first_pct)}pct.png")
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved combined figure: {output_path}")


def plot_tradeoff_summary(data: Dict[str, List[Dict]], dataset_name: str,
                          pruning_pct: float, output_dir: str):
    """
    Plot a clear summary of the QPS gain vs Recall loss tradeoff.

    Shows average metrics across all ef_search values for easy comparison.
    """
    if "baseline" not in data:
        print("Warning: No baseline data found, skipping tradeoff plot")
        return

    baseline_by_ef = {r["ef_search"]: r for r in data["baseline"]}
    methods = ["hub_pruned", "random_pruned", "anti_hub_pruned"]

    # Calculate average recall loss and QPS change for each method
    avg_recall_loss = {}
    avg_qps_change = {}

    for method in methods:
        if method not in data:
            continue

        recall_losses = []
        qps_changes = []

        for result in data[method]:
            ef = result["ef_search"]
            if ef not in baseline_by_ef:
                continue

            baseline = baseline_by_ef[ef]
            recall_loss = (baseline["recall"] - result["recall"]) / baseline["recall"] * 100
            qps_change = (result["qps"] - baseline["qps"]) / baseline["qps"] * 100

            recall_losses.append(recall_loss)
            qps_changes.append(qps_change)

        if recall_losses:
            avg_recall_loss[method] = np.mean(recall_losses)
            avg_qps_change[method] = np.mean(qps_changes)

    if not avg_recall_loss:
        return

    # Create side-by-side bar chart
    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(methods))
    width = 0.35

    recall_bars = ax.bar(x - width/2, [avg_recall_loss.get(m, 0) for m in methods],
                         width, label='Recall Loss (%)', color='#E74C3C', alpha=0.85,
                         edgecolor='black', linewidth=1)
    qps_bars = ax.bar(x + width/2, [avg_qps_change.get(m, 0) for m in methods],
                      width, label='QPS Change (%)', color='#3498DB', alpha=0.85,
                      edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar in recall_bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=TICK_FONTSIZE, fontweight='bold')

    for bar in qps_bars:
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -3
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=TICK_FONTSIZE, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    ax.set_xlabel('Pruning Strategy', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Average Change (%)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f'{dataset_name}\nRecall Loss vs QPS Change ({pruning_pct}% pruning)',
                 fontsize=TITLE_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{dataset_name}_tradeoff_summary.png")
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_tradeoff(all_data: Dict[str, Tuple[Dict, float]], output_dir: str):
    """
    Create a combined figure showing tradeoff summary for all datasets.

    This puts all datasets side by side for easy comparison.
    """
    datasets = list(all_data.keys())
    num_datasets = len(datasets)

    if num_datasets == 0:
        return

    methods = ["hub_pruned", "random_pruned", "anti_hub_pruned"]

    # Calculate metrics for each dataset
    dataset_metrics = {}

    for dataset_name, (data, pruning_pct) in all_data.items():
        if "baseline" not in data:
            continue

        baseline_by_ef = {r["ef_search"]: r for r in data["baseline"]}

        for method in methods:
            if method not in data:
                continue

            recall_losses = []
            qps_changes = []

            for result in data[method]:
                ef = result["ef_search"]
                if ef not in baseline_by_ef:
                    continue

                baseline = baseline_by_ef[ef]
                recall_loss = (baseline["recall"] - result["recall"]) / baseline["recall"] * 100
                qps_change = (result["qps"] - baseline["qps"]) / baseline["qps"] * 100

                recall_losses.append(recall_loss)
                qps_changes.append(qps_change)

            if recall_losses:
                if dataset_name not in dataset_metrics:
                    dataset_metrics[dataset_name] = {}
                dataset_metrics[dataset_name][method] = {
                    'recall_loss': np.mean(recall_losses),
                    'qps_change': np.mean(qps_changes)
                }

    if not dataset_metrics:
        return

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    datasets_with_data = list(dataset_metrics.keys())
    x = np.arange(len(datasets_with_data))
    width = 0.25

    # Plot 1: Recall Loss by pruning strategy
    ax1 = axes[0]
    for i, method in enumerate(methods):
        values = [dataset_metrics[d].get(method, {}).get('recall_loss', 0)
                  for d in datasets_with_data]
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, values, width, label=LABELS[method],
                       color=COLORS[method], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Dataset', fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_ylabel('Recall Loss (%)', fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_title('Recall Loss by Pruning Strategy', fontsize=TITLE_FONTSIZE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets_with_data, fontsize=TICK_FONTSIZE, rotation=15, ha='right')
    ax1.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax1.legend(fontsize=LEGEND_FONTSIZE)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: QPS Change by pruning strategy
    ax2 = axes[1]
    for i, method in enumerate(methods):
        values = [dataset_metrics[d].get(method, {}).get('qps_change', 0)
                  for d in datasets_with_data]
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, values, width, label=LABELS[method],
                       color=COLORS[method], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Dataset', fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_ylabel('QPS Change (%)', fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_title('QPS Change by Pruning Strategy', fontsize=TITLE_FONTSIZE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets_with_data, fontsize=TICK_FONTSIZE, rotation=15, ha='right')
    ax2.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax2.legend(fontsize=LEGEND_FONTSIZE)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    first_pct = list(all_data.values())[0][1]
    output_path = os.path.join(output_dir, f"combined_tradeoff_summary_{int(first_pct)}pct.png")
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved combined tradeoff: {output_path}")


def plot_summary_table(data: Dict[str, List[Dict]], dataset_name: str,
                       pruning_pct: float, output_dir: str):
    """
    Create a summary comparison at a specific ef_search value.
    """
    if "baseline" not in data:
        return

    # Use the first ef_search value for summary
    ef_search = data["baseline"][0]["ef_search"]

    print(f"\n{'='*60}")
    print(f"Summary for {dataset_name} at ef_search={ef_search} ({pruning_pct}% pruning)")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Recall':>10} {'QPS':>10} {'Drop %':>10}")
    print(f"{'-'*60}")

    baseline_recall = data["baseline"][0]["recall"]

    for exp_type in ["baseline", "hub_pruned", "random_pruned", "anti_hub_pruned"]:
        if exp_type in data:
            result = data[exp_type][0]
            recall = result["recall"]
            qps = result["qps"]
            drop = (baseline_recall - recall) / baseline_recall * 100

            print(f"{LABELS[exp_type]:<25} {recall:>10.4f} {qps:>10.1f} {drop:>9.2f}%")

    print(f"{'='*60}")

    # Check if hub pruning causes the most degradation
    hub_drop = (baseline_recall - data["hub_pruned"][0]["recall"]) / baseline_recall * 100
    random_drop = (baseline_recall - data["random_pruned"][0]["recall"]) / baseline_recall * 100
    anti_hub_drop = (baseline_recall - data["anti_hub_pruned"][0]["recall"]) / baseline_recall * 100

    print("\nAnalysis:")
    if hub_drop > random_drop and hub_drop > anti_hub_drop:
        print("  -> Hub pruning causes the MOST recall degradation")
        print("  -> This SUPPORTS the hub-highway hypothesis")
    elif anti_hub_drop < random_drop and anti_hub_drop < hub_drop:
        print("  -> Anti-hub pruning causes the LEAST recall degradation")

    print(f"\nDegradation ranking (worst to best):")
    drops = {"Hub": hub_drop, "Random": random_drop, "Anti-hub": anti_hub_drop}
    for method, drop in sorted(drops.items(), key=lambda x: -x[1]):
        print(f"  {method}: {drop:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Plot pruning comparison results")
    parser.add_argument(
        "--metrics-file",
        type=str,
        default="../metrics/final_pruning_metrics.json",
        help="Path to the metrics JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../plots",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to plot (if not specified, plots all)"
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metrics
    metrics = load_metrics(args.metrics_file)

    # Extract all datasets
    all_datasets = extract_all_datasets(metrics)

    if not all_datasets:
        print("Error: No datasets found in metrics file")
        return

    print(f"Found {len(all_datasets)} dataset(s): {[d[0] for d in all_datasets]}")

    # Filter to specific dataset if requested
    if args.dataset:
        all_datasets = [(d, p) for d, p in all_datasets if d == args.dataset]
        if not all_datasets:
            print(f"Error: Dataset '{args.dataset}' not found in metrics file")
            return

    # Collect all data for combined figure
    all_data_for_combined = {}

    # Process each dataset
    for dataset_name, pruning_pct in all_datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name} ({pruning_pct}% pruning)")
        print(f"{'='*60}")

        # Get experiment data
        data = get_experiment_data(metrics, dataset_name, pruning_pct)

        if not data:
            print(f"Warning: No data found for {dataset_name}")
            continue

        print(f"Found experiments: {list(data.keys())}")

        # Store for combined figure
        all_data_for_combined[dataset_name] = (data, pruning_pct)

        # Generate individual plots
        plot_recall_vs_qps(data, dataset_name, pruning_pct, args.output_dir)
        plot_recall_vs_ef_search(data, dataset_name, pruning_pct, args.output_dir)
        plot_recall_degradation(data, dataset_name, pruning_pct, args.output_dir)
        plot_tradeoff_summary(data, dataset_name, pruning_pct, args.output_dir)

        # Print summary
        plot_summary_table(data, dataset_name, pruning_pct, args.output_dir)

    # Generate combined figures with all datasets
    if len(all_data_for_combined) > 1:
        print(f"\n{'='*60}")
        print("Generating combined figures...")
        print(f"{'='*60}")
        plot_combined_figure(all_data_for_combined, args.output_dir, plot_type="recall_vs_qps")
        plot_combined_figure(all_data_for_combined, args.output_dir, plot_type="recall_vs_latency")
        plot_combined_tradeoff(all_data_for_combined, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
