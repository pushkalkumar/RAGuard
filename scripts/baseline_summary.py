import pandas as pd

df_clean = pd.read_csv("results/baseline_clean.csv")
df_nq_poison = pd.read_csv("results/nq_baseline_poisoned.csv")
df_beir_poison = pd.read_csv("results/beir_baseline_poisoned.csv")

if "Dataset" in df_clean.columns:
    df_nq_poison.insert(0, "Dataset", "nq_poisoned")
    df_beir_poison.insert(0, "Dataset", "beir_poisoned")

df_poisoned = pd.concat([df_nq_poison, df_beir_poison], ignore_index=True)

print("Clean results:")
print(df_clean)

print("\nPoisoned results:")
print(df_poisoned)

import matplotlib.pyplot as plt
import numpy as np

for retriever in ["BM25", "Dense"]:
    df_clean_r = df_clean[df_clean["Retriever"] == retriever]
    df_poisoned_r = df_poisoned[df_poisoned["Retriever"] == retriever]

    x = np.arange(len(df_clean_r["Dataset"]))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, df_clean_r["Recall@5"], width, label="Clean")
    plt.bar(x + width/2, df_poisoned_r["Recall@5"], width, label="Poisoned")

    plt.xticks(x, df_clean_r["Dataset"], rotation=45)
    plt.title(f"Recall@5 Comparison for {retriever} Retriever")
    plt.xlabel("Dataset")
    plt.ylabel("Recall@5")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{retriever}_recall_comparison.png")
    plt.show()