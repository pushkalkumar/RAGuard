import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import glob

# These are the datasets we want included
DATASETS = [
    "baseline_clean.csv",
    "clean_trip_eval.csv",
    "beir_poisoned_10.csv",
    "beir_poisoned_30.csv",
    "beir_poisoned_eval_10.csv",
    "beir_poisoned_eval_30.csv",
    "nq_poisoned_5.csv",
    "nq_poisoned_10.csv",
    "nq_poisoned_20.csv",
    "nq_poisoned_30.csv",
    "nq_poisoned_eval_5.csv",
    "nq_poisoned_eval_10.csv",
    "nq_poisoned_eval_20.csv",
    "nq_poisoned_eval_30.csv"
]

results = []

print("\n=== Evaluating AUPRC and Precision@0.5 on CSV retrieval results ===")

for fname in DATASETS:
    path = f"RAGuard/results/{fname}"
    try:
        df = pd.read_csv(path)

        # we need label (is_relevant) and score (model similarity or rank score)
        if "is_relevant" not in df.columns or "score" not in df.columns:
            print(f"⚠️  Skipping {fname} → missing required columns")
            continue

        labels = df["is_relevant"].astype(int)
        scores = df["score"].astype(float)

        # AUPRC
        auprc = average_precision_score(labels, scores)

        # Precision@0.5 threshold
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        mask = thresholds >= 0.5
        p_at_05 = np.max(precision[mask]) if np.any(mask) else 0.0

        results.append([fname.replace(".csv",""), round(auprc, 4), round(p_at_05, 4)])
        print(f"{fname} → AUPRC={auprc:.4f} | P@0.5={p_at_05:.4f}")

    except FileNotFoundError:
        print(f"❌ File not found: {path}")

df_out = pd.DataFrame(results, columns=["dataset", "AUPRC", "precision_at_0.5"])
df_out.to_csv("RAGuard/results/auprc_precision_final.csv", index=False)

print("\n✅ Saved final metrics → RAGuard/results/auprc_precision_final.csv")

