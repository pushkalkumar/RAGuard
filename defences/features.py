import numpy as np

def entropy_from_token_logprobs(token_logprobs):
    # token_logprobs is a list[float] of log p(token)
    # H = -sum p log p over tokens, approximate with per-token negative log p
    if not token_logprobs:
        return None
    p = np.exp(np.clip(np.array(token_logprobs), -100, 0))
    # normalize to avoid drift if scores are unnormalized
    if p.sum() <= 0:
        return None
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())

def zscores(vals):
    arr = np.array(vals, dtype=float)
    if len(arr) < 2:
        return [0.0 for _ in vals]
    m, s = arr.mean(), arr.std() + 1e-9
    return [(v - m) / s for v in arr]

def pack_feature_row(qid, doc_id, rank_pos, flip, em_delta, f1_delta,
                     base_entropy, alt_entropy, sim_q, sim_ans, sim_z):
    return {
        "qid": qid,
        "doc_id": doc_id,
        "rank_pos": rank_pos,
        "flip_wrong_to_right": int(bool(flip)),
        "em_delta": float(em_delta),
        "f1_delta": float(f1_delta),
        "base_entropy": None if base_entropy is None else float(base_entropy),
        "alt_entropy": None if alt_entropy is None else float(alt_entropy),
        "entropy_delta": None if (base_entropy is None or alt_entropy is None) else float(alt_entropy - base_entropy),
        "sim_q": float(sim_q),
        "sim_ans": float(sim_ans),
        "sim_z": float(sim_z)
    }
