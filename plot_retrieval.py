import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

df = pd.read_csv("RAGuard/results/summary_all.csv")
df = df[df["Retriever"].str.contains("Dense")]
df = df[~df["Dataset"].str.contains("toy")]

df["Category"] = df["Retriever"].apply(
    lambda x: "Poisoned" if "poisoned" in x and "ZKIP" not in x
              else "Poisoned + ZKIP"
)

pivot = df.pivot_table(
    index="Dataset",
    columns="Category",
    values="Recall@5",
    aggfunc="mean"
).reset_index()

datasets = pivot["Dataset"].tolist()
poison_vals = pivot["Poisoned"].tolist()
zkip_vals = pivot["Poisoned + ZKIP"].tolist()

x = np.arange(len(datasets))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
ax.grid(axis="y", linestyle="--", alpha=0.4)

bars1 = ax.bar(x - bar_width/2, poison_vals, width=bar_width,
               label="Poisoned", color="#D55E00", edgecolor="black")
bars2 = ax.bar(x + bar_width/2, zkip_vals, width=bar_width,
               label="Poisoned + ZKIP", color="#009E73", edgecolor="black")

ax.set_ylabel("Recall@5", fontsize=12)
ax.set_title("Effect of ZKIP on Dense Retriever Performance", fontsize=14, weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=10)
ax.legend(frameon=True, fontsize=11)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.004,
            f"{height:.3f}",
            ha="center", va="bottom", fontsize=9
        )

plt.tight_layout()
plt.savefig("RAGuard/results/zkip_effect.png", dpi=300, bbox_inches="tight")
plt.show()

