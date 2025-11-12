"""
Evaluate retrievers with Zero-Knowledge Patch (ZKIP) defense
"""

import sys, os, json, csv, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from defences.patch import ZKPatchDefense
from defences.metrics_zk import compute_zk_metrics


def eval_with_zkip(retriever, queries, gold_docs, poison_docs, zkip, k=5, only_poisoned=False):
    recallsk, reciprocal_ranks, asr_hits = [], [], []
    asr_count = 0

    for q, gold, poison in zip(queries, gold_docs, poison_docs):
        if only_poisoned and poison == gold:
            continue

        # Apply the ZK patch to the query before retrieval
        q_patched = zkip.apply_patch(q)
        retrieved = retriever.search(q_patched, k=k)
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
            poison_rank = next((r for r, doc in enumerate(retrieved_docs) if poison.strip() in doc), None)
            gold_rank = next((r for r, doc in enumerate(retrieved_docs) if gold.strip() in doc), None)
            if poison_rank is not None and (gold_rank is None or poison_rank < gold_rank):
                asr_hits.append(1)
            else:
                asr_hits.append(0)

    recall_at_k = sum(recallsk) / len(recallsk) if recallsk else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    asr = sum(asr_hits) / asr_count if asr_count > 0 else 0.0
    return recall_at_k, mrr, asr, asr_count


def main(dataset, out_file, model_dir_clean, model_dir_poisoned, patch_config, k=5, only_poisoned=False):
    queries, gold_docs, poison_docs = [], [], []
    with open(dataset, "r") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item["query"])
            gold_docs.append(item["gold_doc"])
            poison_docs.append(item.get("poison_doc", item["gold_doc"]))
    print(f"\nLoaded {len(queries)} queries from {dataset}")

    zkip = ZKPatchDefense(config_path=patch_config)

    print("\nEvaluating BM25 + ZKIP...")
    bm25 = BM25Retriever(dataset)
    bm25_recall, bm25_mrr, bm25_asr, bm25_asr_count = eval_with_zkip(bm25, queries, gold_docs, poison_docs, zkip, k=k, only_poisoned=only_poisoned)

    print("\nEvaluating Dense (clean) + ZKIP...")
    dense_clean = DenseRetriever(dataset, model_dir=model_dir_clean)
    clean_recall, clean_mrr, clean_asr, clean_asr_count = eval_with_zkip(dense_clean, queries, gold_docs, poison_docs, zkip, k=k, only_poisoned=only_poisoned)

    print("\nEvaluating Dense (poisoned) + ZKIP...")
    dense_poisoned = DenseRetriever(dataset, model_dir=model_dir_poisoned)
    poison_recall, poison_mrr, poison_asr, poison_asr_count = eval_with_zkip(dense_poisoned, queries, gold_docs, poison_docs, zkip, k=k, only_poisoned=only_poisoned)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Retriever", f"Recall@{k}", "MRR", "ASR", "ASR_Samples"])
        writer.writerow(["BM25 + ZKIP", bm25_recall, bm25_mrr, bm25_asr, bm25_asr_count])
        writer.writerow(["Dense (clean) + ZKIP", clean_recall, clean_mrr, clean_asr, clean_asr_count])
        writer.writerow(["Dense (poisoned) + ZKIP", poison_recall, poison_mrr, poison_asr, poison_asr_count])

    print(f"\n✅ Results saved to {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrievers with ZKIP patching")
    parser.add_argument("--dataset", type=str, required=True, help="Path to poisoned dataset JSONL")
    parser.add_argument("--output", type=str, required=True, help="CSV file to save results")
    parser.add_argument("--model_clean", type=str, default="checkpoints/contriever_clean_v2")
    parser.add_argument("--model_poisoned", type=str, default="checkpoints/contriever_poisoned_v2")
    parser.add_argument("--patch_config", type=str, default="configs/zkip_config.yaml")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--only_poisoned", action="store_true")
    args = parser.parse_args()

    main(args.dataset, args.output, args.model_clean, args.model_poisoned, args.patch_config, args.k, args.only_poisoned)

