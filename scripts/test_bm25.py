import sys, os

# Add project root to Python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers.bm25_retriever import BM25Retriever

if __name__ == "__main__":
    # Point to the toy dataset
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "datasets",
        "clean",
        "sports_toy.jsonl"
    )

    # Initialize BM25 retriever
    retriever = BM25Retriever(dataset_path)

    # Run a test query
    query = "Who won Wimbledon in 2023?"
    results = retriever.search(query, k=3)

    # Print results nicely
    print("Top-3 retrieved docs for:", query)
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"{rank}. {doc}  (score={score:.4f})")
