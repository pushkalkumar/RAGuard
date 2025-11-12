import json
import argparse
from RAGuard.retrievers.bm25_retriever import BM25Retriever
from RAGuard.retrievers.dense_retriever import DenseRetriever
from RAGuard.datasets.load_dataset import load_collection


def load_docs(corpus_file):
    docs = []

    # Try loading using official loader
    try:
        loaded = load_collection(corpus_file)
        if isinstance(loaded, list):
            docs.extend([d.get("text") or d.get("document") or d.get("passage") for d in loaded])
    except:
        pass

    # Also read JSONL manually
    with open(corpus_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            doc_text = obj.get("text") or obj.get("document") or obj.get("passage")
            if doc_text:
                docs.append(doc_text)

    docs = [d for d in docs if d]

    if len(docs) == 0:
        raise ValueError(f"ERROR: No documents found in corpus {corpus_file}")
    
    return docs


def load_poisoned_queries(poisoned_file):
    queries = []
    gold_ids = []
    with open(poisoned_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            queries.append(obj["query"])
            gold_ids.append(obj.get("gold_doc", None))
    return queries, gold_ids


def main(poisoned, output, corpus, model_clean, model_poisoned, k, only_poisoned):

    print(f"=== Running retrieval eval on {poisoned} ===")

    queries, gold_ids = load_poisoned_queries(poisoned)
    print(f"Loaded {len(queries)} queries")

    docs = load_docs(corpus)
    print(f"Loaded {len(docs)} documents for BM25/Dense")

    ####### BM25 #######
    print("Evaluating BM25...")
    bm25 = BM25Retriever(docs)
    bm25_scores = [bm25.retrieve(q, top_k=k) for q in queries]

    ####### Dense (clean) #######
    print("Evaluating Dense (clean model)...")
    dense_clean = DenseRetriever(model_clean, docs)
    dense_scores_clean = [dense_clean.retrieve(q, top_k=k) for q in queries]

    ####### Dense (poisoned) #######
    print("Evaluating Dense (poisoned model)...")
    dense_poison = DenseRetriever(model_poisoned, docs)
    dense_scores_poison = [dense_poison.retrieve(q, top_k=k) for q in queries]

    ####### Save #######
    results = {
        "poisoned_file": poisoned,
        "queries": queries,
        "gold": gold_ids,
        "bm25": bm25_scores,
        "dense_clean": dense_scores_clean,
        "dense_poisoned": dense_scores_poison
    }

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poisoned")
    parser.add_argument("--output")
    parser.add_argument("--corpus")
    parser.add_argument("--model_clean")
    parser.add_argument("--model_poisoned")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--only_poisoned", action="store_true")

    args = parser.parse_args()

    main(args.poisoned, args.output, args.corpus,
         args.model_clean, args.model_poisoned, args.k, args.only_poisoned)

