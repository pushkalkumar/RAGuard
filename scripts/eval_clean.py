import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, csv
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever

def eval_retriever(retriever, queries, gold_docs, k=5):
    recallsk, reciprocal_ranks = [], []
    for q, gold in zip(queries, gold_docs):
        retrieved = retriever.search(q, k=k)
        retrieved_docs = [doc for doc, _ in retrieved]

        recallk = int(any(gold in doc for doc in retrieved_docs))
        recallsk.append(recallk)

        rr = 0.0
        for rank, doc in enumerate(retrieved_docs, start=1):
            if gold in doc:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    
    recall_at_k = sum(recallsk) / len(recallsk)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return recall_at_k, mrr

dataset_path = {
    "nq_toy": "datasets/clean/nq_toy.jsonl",
    "beir_toy": "datasets/clean/beir_toy.jsonl",
    "nq": "datasets/clean/nq.jsonl",
    "beir": "datasets/clean/beir.jsonl",
}

with open("results/baseline_clean.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "Retriever", "Recall@5", "MRR"])
    
    for name, path in dataset_path.items():
        print(f"Running on {name}...")

        queries, gold_docs = [], []
        with open(path, "r") as infile:
            for line in infile:
                item = json.loads(line)
                queries.append(item["query"])
                gold_docs.append(item["gold_doc"])

        # Evaluate BM25 Retriever
        bm25 = BM25Retriever(path)
        bm25_recall, bm25_mrr = eval_retriever(bm25, queries, gold_docs, k=5)
        writer.writerow([name, "BM25", bm25_recall, bm25_mrr])
        print(f"BM25 on {name}: Recall@5={bm25_recall}, MRR={bm25_mrr}")

        # Evaluate Dense Retriever
        dense_retriever = DenseRetriever(path)
        dense_recall, dense_mrr = eval_retriever(dense_retriever, queries, gold_docs, k=5)
        writer.writerow([name, "Dense", dense_recall, dense_mrr])
        print(f"Dense on {name}: Recall@5={dense_recall}, MRR={dense_mrr}")
