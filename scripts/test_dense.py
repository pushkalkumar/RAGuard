import csv
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrievers.dense_retriever import DenseRetriever

retriever = DenseRetriever("datasets/clean/sports_toy.jsonl")

queries = ["Where is the Eiffel Tower located?",
           "Who wrote Hamlet?"]

with open("results/dense_toy.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["query", "doc", "score"])
    for q in queries:
        results = retriever.search(q, k=3)
        for doc, score in results:
            writer.writerow([q, doc, score])
