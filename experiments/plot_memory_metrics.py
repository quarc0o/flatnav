#!/usr/bin/env python
"""Plot memory metrics comparing FlatNav vs HNSW for each dataset."""

import json
import os
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Colors for different index variants
COLORS = {
    "hnsw": "orange",
    "flatnav-base": "green",  # FlatNav without HNSW base layer
}

# Only plot these variants
VARIANTS_TO_PLOT = {"hnsw", "flatnav-base"}


def parse_key(key: str) -> tuple[str, str, str]:
    """
    Parse experiment key into (base_dataset, variant_label, full_key).
    Examples:
        'mnist-784_hnsw' -> ('mnist-784', 'hnsw', 'mnist-784_hnsw')
        'mnist-784_flatnav' -> ('mnist-784', 'flatnav', 'mnist-784_flatnav')
        'mnist-784-base_flatnav' -> ('mnist-784', 'flatnav-base', 'mnist-784-base_flatnav')
    """
    parts = key.rsplit("_", 1)
    if len(parts) != 2:
        return (key, "unknown", key)

    dataset_name, idx_type = parts

    # Check if dataset name has a '-base' suffix (FlatNav native variant)
    if dataset_name.endswith("-base"):
        base_dataset = dataset_name[:-5]  # Remove '-base'
        variant = f"{idx_type}-base"
    else:
        base_dataset = dataset_name
        variant = idx_type

    return (base_dataset, variant, key)


def group_by_dataset(data: dict) -> dict:
    """Group by base dataset: {base_dataset: {variant: [runs]}}"""
    grouped = defaultdict(lambda: defaultdict(list))
    for key, runs in data.items():
        base_dataset, variant, _ = parse_key(key)
        grouped[base_dataset][variant].extend(runs)
    return grouped


def plot_memory(grouped: dict, output_dir: str):
    """Create memory comparison plots for each dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Nice labels for legend
    LABELS = {
        "hnsw": "HNSW",
        "flatnav-base": "FlatNav (From scratch)",
    }

    for dataset, variants in grouped.items():
        # Filter to only plot desired variants
        variants_filtered = {k: v for k, v in variants.items() if k in VARIANTS_TO_PLOT}
        if not variants_filtered:
            print(f"Skipping {dataset}: no matching variants")
            continue

        # Index memory plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant, runs in sorted(variants_filtered.items()):
            ef_vals = sorted(set(r["ef_construction"] for r in runs))
            mem_vals = [np.mean([r["index_memory_mb"] for r in runs if r["ef_construction"] == ef]) for ef in ef_vals]
            label = LABELS.get(variant, variant.upper())
            ax.plot(ef_vals, mem_vals, "o-", color=COLORS.get(variant, "gray"), label=label, linewidth=2)

        ax.set_xlabel("ef_construction")
        ax.set_ylabel("Index Memory (MB)")
        ax.set_title(f"{dataset}: Index Memory")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset}_index_memory.png"), dpi=150)
        plt.close()

        # Search memory plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant, runs in sorted(variants_filtered.items()):
            ef_vals = sorted(set(r["ef_construction"] for r in runs))
            mem_vals = [np.mean([r["search_memory_mb"] for r in runs if r["ef_construction"] == ef]) for ef in ef_vals]
            label = LABELS.get(variant, variant.upper())
            ax.plot(ef_vals, mem_vals, "o-", color=COLORS.get(variant, "gray"), label=label, linewidth=2)

        ax.set_xlabel("ef_construction")
        ax.set_ylabel("Search Memory (MB)")
        ax.set_title(f"{dataset}: Search Memory")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset}_search_memory.png"), dpi=150)
        plt.close()

        # Construction time plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant, runs in sorted(variants_filtered.items()):
            ef_vals = sorted(set(r["ef_construction"] for r in runs))
            time_vals = [np.mean([r["construction_time_sec"] for r in runs if r["ef_construction"] == ef]) for ef in ef_vals]
            label = LABELS.get(variant, variant.upper())
            ax.plot(ef_vals, time_vals, "o-", color=COLORS.get(variant, "gray"), label=label, linewidth=2)

        ax.set_xlabel("ef_construction")
        ax.set_ylabel("Construction Time (s)")
        ax.set_title(f"{dataset}: Construction Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset}_construction_time.png"), dpi=150)
        plt.close()

        print(f"Saved plots for {dataset}: {list(variants_filtered.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Plot memory metrics")
    parser.add_argument("--input", default="../metrics/memory_metrics.json")
    parser.add_argument("--output-dir", default="../metrics/plots")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    grouped = group_by_dataset(data)
    print(f"Found datasets: {list(grouped.keys())}")

    plot_memory(grouped, args.output_dir)
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
