import numpy as np

def asr(audits):
    wrong = [a for a in audits if a["base_em"]==0]
    if not wrong: return 0.0
    still_wrong = sum(1 for a in wrong if a["final_em"]==0)
    return still_wrong / len(wrong)

def cost(audits):
    calls = [a["num_generator_calls"] for a in audits]
    return {"avg_gen_calls": float(np.mean(calls))}

def precision_recall_flagged(audits, poison_flags):
    y_true, y_pred = [], []
    for a in audits:
        for d in a["tested"]:
            y_pred.append(int(d in a["flagged"]))
            y_true.append(int(poison_flags.get(d, {}).get("is_poison", False)))
    if not y_true: 
        return {"precision": None, "recall": None}
    tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
    return {
        "precision": tp / max(1, tp+fp),
        "recall": tp / max(1, tp+fn),
    }

def compute_zk_metrics(query_emb, doc_emb, token_logprobs=None):
    """
    Compute Zero-Knowledge metrics for retrieval robustness.
    Returns a dictionary with Entropy Differential, 
    Context Similarity Z-score, and Embedding Displacement Distance.

    Args:
        query_emb (np.ndarray): Query embedding vector.
        doc_emb (np.ndarray): Document embedding vector.
        token_logprobs (list[float], optional): Token-level log probabilities.
    """
    metrics = {}

    # --- Context Similarity Z-score ---
    sim = float(np.dot(query_emb, doc_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)))
    metrics["context_similarity_z"] = (sim - 0.5) / 0.1  # normalized around 0.5

    # --- Embedding Displacement Distance ---
    dist = float(np.linalg.norm(query_emb - doc_emb))
    metrics["embedding_displacement"] = dist

    # --- Entropy Differential (ΔH) ---
    if token_logprobs is not None and len(token_logprobs) > 1:
        probs = np.exp(token_logprobs)
        probs = probs / np.sum(probs)
        entropy = -np.sum(probs * np.log(probs))
        metrics["entropy_diff"] = float(entropy)
    else:
        metrics["entropy_diff"] = 0.0

    return metrics
