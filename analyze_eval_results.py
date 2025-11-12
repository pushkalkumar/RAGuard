import json
import pandas as pd

infile = "RAGuard/results/llm_eval_results.jsonl"
outfile = "RAGuard/results/llm_eval_results.csv"

rows = []
with open(infile, "r") as f:
    for line in f:
        try:
            rows.append(json.loads(line.strip()))
        except:
            pass

df = pd.DataFrame(rows)
df.to_csv(outfile, index=False)

print(f"✅ Saved CSV → {outfile}")
print(f"Total rows: {len(df)}")
print(df.head())

