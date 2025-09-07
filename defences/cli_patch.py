import argparse, json
from defences.io_utils import load_queries, load_passages, load_runs, load_poison_flags
from defences.patch import ZeroKnowledgePatch, PatchConfig, QAGenerator
from defences.metrics_zk import asr, cost, precision_recall_flagged

ap = argparse.ArgumentParser()
ap.add_argument("--queries", required=True)
ap.add_argument("--passages", required=True)
ap.add_argument("--runs", required=True)
ap.add_argument("--poison_flags", default=None)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--loo_cap", type=int, default=8)
ap.add_argument("--early_exit", action="store_true")
ap.add_argument("--model", default="google/flan-t5-small")
ap.add_argument("--out", required=True)
args = ap.parse_args()

queries = load_queries(args.queries)
passages_by_id = {p["doc_id"]: p["doc_text"] for p in load_passages(args.passages)}
runs = load_runs(args.runs)
flags = load_poison_flags(args.poison_flags)

cfg = PatchConfig(k=args.k, leave_one_out_cap=args.loo_cap, early_exit=args.early_exit, model_name=args.model)
qa = QAGenerator(cfg.model_name)
patch = ZeroKnowledgePatch(qa, passages_by_id, cfg)

audits = []
for q in queries:
    ranked = runs.get(q["qid"], [])[:cfg.k]
    if not ranked:
        continue
    audits.append(patch.run_one(q, ranked))

with open(args.out, "w") as f:
    for a in audits:
        f.write(json.dumps(a) + "\\n")

summary = {"ASR": asr(audits), **cost(audits)}
if flags:
    summary.update(precision_recall_flagged(audits, flags))
summary["clean_em_before"] = float(sum(a["base_em"] for a in audits)/max(1,len(audits)))
summary["clean_em_after"]  = float(sum(a["final_em"] for a in audits)/max(1,len(audits)))
print(json.dumps(summary, indent=2))
