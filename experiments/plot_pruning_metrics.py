"""
Plot Pruning Experiment Metrics

Generates recall vs latency and recall vs QPS plots from metrics.json
"""

import json
import os
import matplotlib.pyplot as plt

# Path to metrics file
METRICS_FILE = os.path.join(os.path.dirname(__file__), "..", "metrics", "metrics.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def load_metrics(filepath: str) -> dict:
    """Load metrics from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def plot_recall_vs_latency(metrics: dict, output_dir: str):
    """Plot recall vs latency (p50) for all three methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for each method
    styles = {
        "gist-pruning-test-random_baseline": {
            "label": "Baseline",
            "color": "#2ecc71",
            "marker": "o"
        },
        "gist-pruning-test-random_hub_pruned": {
            "label": "Hub Pruned",
            "color": "#e74c3c",
            "marker": "s"
        },
        "gist-pruning-test-random_random_pruned": {
            "label": "Random Pruned",
            "color": "#3498db",
            "marker": "^"
        }
    }

    for key, style in styles.items():
        if key not in metrics:
            continue

        data = metrics[key]
        recalls = [d["recall"] for d in data]
        latencies = [d["latency_p50"] for d in data]

        # Sort by latency for proper line connection
        sorted_data = sorted(zip(latencies, recalls))
        latencies, recalls = zip(*sorted_data)

        ax.plot(latencies, recalls,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=2,
                markersize=8)

    ax.set_xlabel("Latency P50 (ms)", fontsize=12)
    ax.set_ylabel("Recall@100", fontsize=12)
    ax.set_title("Recall vs Latency - Pruning Comparison (GIST)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "pruning_recall_vs_latency.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def plot_recall_vs_qps(metrics: dict, output_dir: str):
    """Plot recall vs QPS for all three methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors and markers for each method
    styles = {
        "gist-pruning-test-random_baseline": {
            "label": "Baseline",
            "color": "#2ecc71",
            "marker": "o"
        },
        "gist-pruning-test-random_hub_pruned": {
            "label": "Hub Pruned",
            "color": "#e74c3c",
            "marker": "s"
        },
        "gist-pruning-test-random_random_pruned": {
            "label": "Random Pruned",
            "color": "#3498db",
            "marker": "^"
        }
    }

    for key, style in styles.items():
        if key not in metrics:
            continue

        data = metrics[key]
        recalls = [d["recall"] for d in data]
        qps_values = [d["qps"] for d in data]

        # Sort by QPS for proper line connection
        sorted_data = sorted(zip(qps_values, recalls), reverse=True)
        qps_values, recalls = zip(*sorted_data)

        ax.plot(qps_values, recalls,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=2,
                markersize=8)

    ax.set_xlabel("Queries Per Second (QPS)", fontsize=12)
    ax.set_ylabel("Recall@100", fontsize=12)
    ax.set_title("Recall vs QPS - Pruning Comparison (GIST)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "pruning_recall_vs_qps.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved: {output_file}")


def main():
    # Load metrics
    metrics = load_metrics(METRICS_FILE)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating pruning comparison plots...")

    # Generate both plots
    plot_recall_vs_latency(metrics, OUTPUT_DIR)
    plot_recall_vs_qps(metrics, OUTPUT_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
