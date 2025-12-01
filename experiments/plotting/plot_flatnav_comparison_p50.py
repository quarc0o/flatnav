import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import scipy.stats
import sys

plt.style.use('tableau-colorblind10')
flatnav_hnsw_color = "r"
flatnav_base_color = "#264478"


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
        if dataset_name not in d:
            return None, None
        latency = []
        recall = []
        for result in d[dataset_name]:
            latency.append(result[latency_field])
            recall.append(result["recall"])
        recall, latency = np.array(recall), np.array(latency)
        return pareto_frontier(recall, latency)


# Check which datasets are available
json_filename = "metrics_ann_bench.json"
with open(json_filename) as f:
    data = json.load(f)

datasets_config = [
    ("deep-image-96", "DEEP", (0, 2, 0, 2)),
    ("gist", "GIST", (0, 2, 2, 4)),
    ("mnist-784", "MNIST", (2, 4, 0, 2)),
    ("nytimes-256-angular", "NY-Times (Angular)", (2, 4, 2, 4)),
    ("sift", "SIFT", (4, 6, 1, 3)),
]

available_datasets = []
for dataset, title, position in datasets_config:
    flatnav_key = f"{dataset}_flatnav"
    base_key = f"{dataset}-base_flatnav"
    if flatnav_key in data and base_key in data:
        available_datasets.append((dataset, title, position))

if not available_datasets:
    print("Error: No datasets with both FlatNav variants found in metrics_ann_bench.json")
    print("Please run the flatnav-base benchmarks first.")
    sys.exit(1)

fig = plt.figure(figsize=(10,12))
gs = gridspec.GridSpec(6, 4)

# Plot only available datasets
for dataset, title, position in available_datasets:
    ax = plt.subplot(gs[position[0]:position[1], position[2]:position[3]])

    flatnav_hnsw_recall, flatnav_hnsw_latency = load_dataset(f"{dataset}_flatnav", json_filename, "latency_p50")
    flatnav_base_recall, flatnav_base_latency = load_dataset(f"{dataset}-base_flatnav", json_filename, "latency_p50")

    plt.plot(flatnav_hnsw_recall, flatnav_hnsw_latency, 'x-', color = flatnav_hnsw_color, label = "FlatNav (HNSW base)")
    plt.plot(flatnav_base_recall, flatnav_base_latency, 'o-', color = flatnav_base_color, label = "FlatNav (base)")

    # We change the fontsize of minor ticks label
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.gca().tick_params(axis='both', which='minor', labelsize=8)
    plt.yscale('log')
    plt.title(title, fontsize=24)
    plt.grid(which='both')
    plt.legend(fontsize=12, ncol=1)

fig.supylabel('P50 Latency (ms)', fontsize = 28)
fig.supxlabel('Recall (R100@100)', fontsize = 28)

plt.tight_layout()
plt.savefig("flatnav_comparison_p50.png", dpi=400)
print(f"Plot saved to flatnav_comparison_p50.png with {len(available_datasets)} dataset(s)")
# plt.show()
