import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned results
df = pd.read_csv("summary_all_final.csv", on_bad_lines='skip')

# Strip whitespace and standardize column names
df.columns = [c.strip() for c in df.columns]

# If Dataset column missing, infer it from file name prefix
if "Dataset" not in df.columns:
    df.insert(0, "Dataset", "aggregate")

# Remove duplicates
df = df.drop_duplicates()

# Metrics to plot (check what columns exist)
possible_metrics = ["Recall@5", "MRR", "ASR"]
metrics = [m for m in possible_metrics if m in df.columns]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.titlesize": 15
})

datasets = sorted(df["Dataset"].unique())

for dataset in datasets:
    subset = df[df["Dataset"] == dataset]
    if subset.empty:
        continue

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    fig.suptitle(f"Retrieval Performance Summary ({dataset.upper()})", fontsize=15, weight="bold")

    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        axes[i].barh(subset["Retriever"], subset[metric], color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
        axes[i].set_title(metric)
        axes[i].set_xlabel(metric)
        axes[i].invert_yaxis()

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.savefig(f"{dataset}_results_summary.png", dpi=300)
    plt.close()

print("✅ Plots saved successfully as *_results_summary.png in results/")

