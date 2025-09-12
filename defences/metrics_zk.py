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
