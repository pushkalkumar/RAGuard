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
passages_by_id = {p["doc_id"]: p["doc_text"] for p in load_passages(args.passages)}
runs = load_runs(args.runs)
flags = load_poison_flags(args.poison_flags)

cfg = PatchConfig(k=args.k, leave_one_out_cap=args.loo_cap, early_exit=args.early_exit, model_name=args.model, batch_counterfactuals=True)
qa = QAGenerator(cfg.model_name)
patch = ZeroKnowledgePatch(qa, passages_by_id, cfg)

with open(args.out_features, "w") as outf:
    audits = []
    for q in queries:
        ranked = runs.get(q["qid"], [])[:cfg.k]
        if not ranked: 
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
