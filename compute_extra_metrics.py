import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

results_dir = "RAGuard/results"
output = []

print("\n=== Evaluating AUPRC / Precision@0.5 on retrieved CSV results ===")

for fname in os.listdir(results_dir):
    if not fname.endswith(".csv"):
        continue
    if "poison" not in fname and "clean" not in fname:
        continue

    path = os.path.join(results_dir, fname)
    df = pd.read_csv(path)

    # Must have relevance + score
    if not set(["score","is_relevant"]).issubset(df.columns):
        print(f"⚠️  Skipping {fname} — missing score/is_relevant")
        continue

    scores = df["score"].values
    labels = df["is_relevant"].values

    auprc = average_precision_score(labels, scores)

    precision, recall, thresh = precision_recall_curve(labels, scores)
    if len(thresh) > 0:
        p_at_05 = float(np.max(precision[thresh >= 0.5])) if np.any(thresh >= 0.5) else 0.0
    else:
        p_at_05 = 0.0

    output.append({
        "dataset": fname.replace(".csv",""),
        "auprc": round(auprc, 4),
        "precision@0.5": round(p_at_05, 4)
    })

df_out = pd.DataFrame(output)
df_out.to_csv(os.path.join(results_dir, "auprc_precision_final.csv"), index=False)

print("\n✅ Saved real results → RAGuard/results/auprc_precision_final.csv")
print(df_out)

