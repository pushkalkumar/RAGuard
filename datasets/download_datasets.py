import os
import requests
import zipfile

DATA_DIR = "../datasets/downloads"
os.makedirs(DATA_DIR, exist_ok=True)

url = "https://huggingface.co/datasets/nq-open/1.0.0/resolve/main/nq-train.jsonl"
output_file = os.path.join(DATA_DIR, "nq-train.jsonl")

if not os.path.exists(output_file):
    print(f"Downloading dataset from {url}...")
    r = requests.get(url)
    with open(output_file,"wb") as f:
        f.write(r.content)
    print(f"Dataset downloaded and saved to {output_file}")
else:
    print(f"Dataset already exists at {output_file}")

