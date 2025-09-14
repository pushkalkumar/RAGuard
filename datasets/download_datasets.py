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

import json
from datasets import load_dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def process_nq():
    nq = load_dataset("nq_open", split="train[:1000]")  # small subset for speed
    os.makedirs("datasets/clean", exist_ok=True)
    with open("datasets/clean/nq.jsonl", "w") as f:
        for item in nq:
            query = item["question"].strip().lower()
            doc = " ".join(item["answer"]).strip().lower()
            # Add more cleaning if needed
            f.write(json.dumps({"query": query, "gold_doc": doc}) + "\n")

def process_beir():
    # Download BEIR 'nfcorpus' as example
    data_path = util.download_and_unzip("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip", "datasets/clean")
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    with open("datasets/clean/beir.jsonl", "w") as f:
        for qid, query in queries.items():
            query_text = query.strip().lower()
            if qid in qrels:
                for doc_id in qrels[qid]:
                    doc_text = corpus[doc_id]["text"].strip().lower()
                    f.write(json.dumps({"query": query_text, "gold_doc": doc_text}) + "\n")

def make_toy_subset(src, dst, n=100):
    with open(src) as fin, open(dst, "w") as fout:
        for i, line in enumerate(fin):
            if i >= n: break
            fout.write(line)

if __name__ == "__main__":
    process_nq()
    process_beir()
    make_toy_subset("datasets/clean/nq.jsonl", "datasets/clean/nq_toy.jsonl")
    make_toy_subset("datasets/clean/beir.jsonl", "datasets/clean/beir_toy.jsonl")
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

