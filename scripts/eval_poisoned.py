import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, csv, argparse
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever

def eval_poison(retriever, queries, gold_docs, poison_docs, k=5):
    recallsk, reciprocal_ranks, asr_hits = [], [], []
    # number of examples considered for ASR (we'll skip cases where poison == gold)
    asr_count = 0
    for q, gold, poison in zip(queries, gold_docs, poison_docs):
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

        # prefer exact match but fall back to substring containment to handle minor formatting differences
        poison_rank = next((rank for rank, doc in enumerate(retrieved_docs) if doc == poison), None)
        if poison_rank is None:
            poison_rank = next((rank for rank, doc in enumerate(retrieved_docs) if poison in doc), None)

        gold_rank = next((rank for rank, doc in enumerate(retrieved_docs) if doc == gold), None)
        if gold_rank is None:
            gold_rank = next((rank for rank, doc in enumerate(retrieved_docs) if gold in doc), None)

        if poison == gold:
            continue
        asr_count += 1
        if poison_rank is not None and (gold_rank is None or poison_rank < gold_rank):
            asr_hits.append(1)
        else:
            asr_hits.append(0)
    
    recall_at_k = sum(recallsk) / len(recallsk)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    asr = sum(asr_hits) / asr_count if asr_count > 0 else 0.0
    return recall_at_k, mrr, asr

def main(poisoned_file, out_file, k=5):
    queries, gold_docs, poison_docs = [], [], []
    with open(poisoned_file, "r") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item["query"])
            gold_docs.append(item["gold_doc"])
            poison_docs.append(item.get("poison_doc", item["gold_doc"]))

    bm25 = BM25Retriever(poisoned_file)
    bm25_recall, bm25_mrr, bm25_asr = eval_poison(bm25, queries, gold_docs, poison_docs, k=5)

    dense = DenseRetriever(poisoned_file)
    dense_recall, dense_mrr, dense_asr = eval_poison(dense, queries, gold_docs, poison_docs, k=5)

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Retriever", f"Recall@{k}", "MRR", "ASR"])
        writer.writerow(["BM25", bm25_recall, bm25_mrr, bm25_asr])
        writer.writerow(["Dense", dense_recall, dense_mrr, dense_asr])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrievers on poisoned dataset")
    parser.add_argument("--poisoned", type=str, required=True, help="Path to the poisoned dataset JSONL file")
    parser.add_argument("--output", type=str, required=True, help="CSV file to save results")
    parser.add_argument("--k", type=int, default=5, help="Top-k to evaluate")
    args = parser.parse_args()

    main(args.poisoned, args.output, args.k)