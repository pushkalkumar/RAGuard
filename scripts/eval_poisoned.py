import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, csv, argparse
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever

def eval_poison(retriever, queries, gold_docs, poison_docs, k=5, only_poisoned=False):
    recallsk, reciprocal_ranks, asr_hits = [], [], []
    asr_count = 0

    for q, gold, poison in zip(queries, gold_docs, poison_docs):
        if only_poisoned and poison == gold:
            continue

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

        if poison != gold:
            asr_count += 1

            poison_rank = next((rank for rank, doc in enumerate(retrieved_docs) if doc.strip() == poison.strip()), None)
            if poison_rank is None:
                poison_rank = next((rank for rank, doc in enumerate(retrieved_docs) if poison.strip() in doc), None)
            
            gold_rank = next((rank for rank, doc in enumerate(retrieved_docs) if doc.strip() == gold.strip()), None)
            if gold_rank is None:
                gold_rank = next((rank for rank, doc in enumerate(retrieved_docs) if gold.strip() in doc), None)
            
            if poison_rank is not None and (gold_rank is None or poison_rank < gold_rank):
                asr_hits.append(1)
            else:
                asr_hits.append(0)
    
    recall_at_k = sum(recallsk) / len(recallsk) if recallsk else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    asr = sum(asr_hits) / asr_count if asr_count > 0 else 0.0
    return recall_at_k, mrr, asr, asr_count

def main(poisoned_file, out_file, model_dir_clean=None, model_dir_poisoned=None, k=5, only_poisoned=False):
    queries, gold_docs, poison_docs = [], [], []
    with open(poisoned_file, "r") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item["query"])
            gold_docs.append(item["gold_doc"])
            poison_docs.append(item.get("poison_doc", item["gold_doc"]))
    print(f"\nLoaded {len(queries)} queries from {poisoned_file}")

    print("\nEvaluating BM25...")
    bm25 = BM25Retriever(poisoned_file)
    bm25_recall, bm25_mrr, bm25_asr, bm25_asr_count = eval_poison(bm25, queries, gold_docs, poison_docs, k=k, only_poisoned=only_poisoned)

    print("\nEvaluating Dense (clean)...")
    dense_clean = DenseRetriever(poisoned_file, model_name=model_dir_clean or "sentence-transformers/all-MiniLM-L6-v2")
    clean_recall, clean_mrr, clean_asr, clean_asr_count = eval_poison(dense_clean, queries, gold_docs, poison_docs, k=k, only_poisoned=only_poisoned)
    
    print("\nEvaluating Dense (poisoned)...")
    dense_poisoned = DenseRetriever(poisoned_file, model_dir=model_dir_poisoned)
    poison_recall, poison_mrr, poison_asr, poison_asr_count = eval_poison(dense_poisoned, queries, gold_docs, poison_docs, k=k, only_poisoned=only_poisoned)

    # dense = DenseRetriever(poisoned_file)
    # dense_recall, dense_mrr, dense_asr, dense_asr_count = eval_poison(dense, queries, gold_docs, poison_docs, k=5)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Retriever", f"Recall@{k}", "MRR", "ASR", "ASR_Samples"])
        writer.writerow(["BM25", bm25_recall, bm25_mrr, bm25_asr, bm25_asr_count])
        writer.writerow(["Dense (clean)", clean_recall, clean_mrr, clean_asr, clean_asr_count])
        writer.writerow(["Dense (poisoned)", poison_recall, poison_mrr, poison_asr, poison_asr_count])
        # writer.writerow(["Dense", dense_recall, dense_mrr, dense_asr, dense_asr_count])
    
    print(f"\nResults saved to {out_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrievers on poisoned dataset")
    parser.add_argument("--poisoned", type=str, required=True, help="Path to the poisoned dataset JSONL file")
    parser.add_argument("--output", type=str, required=True, help="CSV file to save results")
    parser.add_argument("--model_clean", type=str, default=None, help="Path to clean dense model")
    parser.add_argument("--model_poisoned", type=str, default="checkpoints/contriever_poisoned", help="Path to poisoned trained dense model")
    parser.add_argument("--k", type=int, default=5, help="Top-k to evaluate")
    parser.add_argument("--only_poisoned", action="store_true", help="Evaluate only on queries with poison_doc different from gold_doc")
    args = parser.parse_args()

    main(args.poisoned, args.output, args.model_clean, args.model_poisoned, args.k, args.only_poisoned)