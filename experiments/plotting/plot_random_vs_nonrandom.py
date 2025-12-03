import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import scipy.stats
import sys

plt.style.use('tableau-colorblind10')
nonrandom_color = "#264478"  # Blue
random_color = "r"  # Red


def pareto_frontier(recall, latency):
    sorting_indices = recall.argsort()
    recall = recall[sorting_indices[::-1]]
    latency = latency[sorting_indices[::-1]]

    output_recall = []
    output_latency = []
    best_latency = latency[0]
    for l, r in zip(latency, recall):
        # descending recall order
        if l > best_latency:
            continue
        if l <= best_latency:
            best_latency = l
            output_recall.append(r)
            output_latency.append(l)
    return np.array(output_recall), np.array(output_latency)

def load_dataset(
    dataset_name,
    json_filename,
    latency_field,
):
    with open(json_filename) as f:
        d = json.load(f)

        # Check if dataset exists in the metrics file
        if dataset_name not in d:
            print(f"Warning: Dataset '{dataset_name}' not found in {json_filename}. Skipping.")
            return None, None

        latency = []
        recall = []
        for result in d[dataset_name]:
            latency.append(result[latency_field])
            recall.append(result["recall"])
        recall, latency = np.array(recall), np.array(latency)
        return pareto_frontier(recall, latency)


def plot_dataset_comparison(ax, dataset, json_filename, latency_field, title):
    """Plot comparison for a dataset if data is available."""
    nonrandom_recall, nonrandom_latency = load_dataset(f"{dataset}-non-random_flatnav", json_filename, latency_field)
    random_recall, random_latency = load_dataset(f"{dataset}-random_flatnav", json_filename, latency_field)

    # Check if at least one dataset has data
    has_data = False
    if nonrandom_recall is not None and len(nonrandom_recall) > 0:
        ax.plot(nonrandom_recall, nonrandom_latency, 'x-', color=nonrandom_color, label="Non-Random Init", linewidth=2, markersize=8)
        has_data = True

    if random_recall is not None and len(random_recall) > 0:
        ax.plot(random_recall, random_latency, 'o-', color=random_color, label="Random Init", linewidth=2, markersize=8)
        has_data = True

    if not has_data:
        ax.text(0.5, 0.5, 'No data available',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='gray')

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.set_yscale('log')
    ax.set_title(title, fontsize=24)
    ax.grid(which='both')
    if has_data:
        ax.legend(fontsize=14, ncol=1)


fig = plt.figure(figsize=(10, 12))

gs = gridspec.GridSpec(6, 4)

# Plot DEEP
ax = plt.subplot(gs[0:2, 0:2])
plot_dataset_comparison(ax, "deep-image-96", "metrics_random.json", "latency_p99", "DEEP")

# Plot GIST
ax = plt.subplot(gs[0:2, 2:4])
plot_dataset_comparison(ax, "gist", "metrics_random.json", "latency_p99", "GIST")

# Plot MNIST
ax = plt.subplot(gs[2:4, 0:2])
plot_dataset_comparison(ax, "mnist-784", "metrics_random.json", "latency_p99", "MNIST")

# Plot NY-Times
ax = plt.subplot(gs[2:4, 2:4])
plot_dataset_comparison(ax, "nytimes-256-angular", "metrics_random.json", "latency_p99", "NY-Times (Angular)")

# Plot SIFT
ax = plt.subplot(gs[4:6, 1:3])
plot_dataset_comparison(ax, "sift", "metrics_random.json", "latency_p99", "SIFT")

fig.supylabel('P99 Latency (ms)', fontsize=28)
fig.supxlabel('Recall (R100@100)', fontsize=28)

plt.tight_layout()
plt.savefig("random_vs_nonrandom_p99.png", dpi=400)
print("Plot saved as random_vs_nonrandom_p99.png")
