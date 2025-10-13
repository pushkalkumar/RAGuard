
import argparse, json
from defences.io_utils import load_queries, load_passages, load_runs, load_poison_flags
from defences.patch import ZeroKnowledgePatch, PatchConfig, QAGenerator
from defences.metrics_zk import asr, cost

ap = argparse.ArgumentParser()
ap.add_argument("--queries", required=True)
ap.add_argument("--passages", required=True)
ap.add_argument("--runs", required=True)
ap.add_argument("--poison_flags", default=None)  # optional label join
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--loo_cap", type=int, default=8)
ap.add_argument("--early_exit", action="store_true")
ap.add_argument("--model", default="google/flan-t5-small")
ap.add_argument("--out_features", required=True)  # JSONL with one row per tested doc
args = ap.parse_args()

queries = load_queries(args.queries)
# Fallback if passages don’t have explicit IDs
# Load passages and auto-generate IDs if missing
raw_passages = list(load_passages(args.passages))
passages_by_id = {}
for i, p in enumerate(raw_passages):
    doc_id = p.get("doc_id", f"auto_{i}")
    text = p.get("doc_text") or p.get("poison_doc") or p.get("gold_doc") or ""
    passages_by_id[doc_id] = text
runs = load_runs(args.runs)
flags = load_poison_flags(args.poison_flags)

cfg = PatchConfig(k=args.k, leave_one_out_cap=args.loo_cap, early_exit=args.early_exit, model_name=args.model, batch_counterfactuals=True)
qa = QAGenerator(cfg.model_name)
patch = ZeroKnowledgePatch(qa, passages_by_id, cfg)

with open(args.out_features, "w") as outf:
    audits = []
    for i, q in enumerate(queries):
        # assign unique ID if missing
        if "qid" not in q:
            q["qid"] = f"q{i}"
        qid = q["qid"]
        ranked = runs.get(qid, [])[:cfg.k]
        # Filter only IDs that exist in passages_by_id
        ranked = [r for r in ranked if r in passages_by_id] or list(passages_by_id.keys())[:cfg.k]
        if not ranked:
            continue
        if "gold_answers" not in q and "gold_doc" in q:
            q["gold_answers"] = [q["gold_doc"]]
        if "query" not in q:
            continue
        a = patch.run_one(q, ranked)
        audits.append(a)
        # write feature rows
        for row in a.get("feature_rows", []):
            if flags:
                row["label_is_poison"] = int(flags.get(row["doc_id"], {}).get("is_poison", False))
                row["poison_type"] = flags.get(row["doc_id"], {}).get("poison_type")
            outf.write(json.dumps(row) + "\n")

# small audit summary for sanity
print(json.dumps({"ASR": asr(audits), **cost(audits), "queries": len(audits)}, indent=2))
