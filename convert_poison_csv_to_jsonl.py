import pandas as pd
import json
import os

input_dir = "RAGuard/results"
output_dir = "RAGuard/datasets/poisoned"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.startswith("nq_poisoned_") or fname.startswith("beir_poisoned_"):
        if fname.endswith(".csv"):
            csv_path = os.path.join(input_dir, fname)
            jsonl_path = os.path.join(output_dir, fname.replace(".csv", ".jsonl"))
            
            df = pd.read_csv(csv_path)
            with open(jsonl_path, "w") as f:
                for _, row in df.iterrows():
                    f.write(json.dumps(row.to_dict()) + "\n")

            print("✅ converted", fname, "→", jsonl_path)

