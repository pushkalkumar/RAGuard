import os, json, argparse, random
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from zkip import zkip_scores
from openai import OpenAI

def load_dataset(name, poison_ratio):
    path=f"datasets/{name}_{int(poison_ratio*100) if poison_ratio>0 else 'clean'}.jsonl"
    if not os.path.exists(path):
        path=f"datasets/{name}_clean.jsonl"
    data=[]
    with open(path,"r") as f:
        for line in f: data.append(json.loads(line))
    return data

def bm25_topk(query, corpus, k=5):
    q=query.lower().split()
    scored=[(sum(d.lower().count(t) for t in q), d) for d in corpus]
    scored.sort(reverse=True, key=lambda x:x[0])
    return [d for _,d in scored[:k]]

def openai_generator(model_id, temp, top_p, max_tokens):
    client=OpenAI()
    def _gen(q, docs):
        prompt=f"Answer the question. Q: {q}\nContext:\n" + "\n\n".join(docs)
        r=client.chat.completions.create(
            model=model_id,
            messages=[{"role":"user","content":prompt}],
            temperature=temp, top_p=top_p, max_tokens=max_tokens, logprobs=True
        )
        text=r.choices[0].message.content or ""
        lp=[]
        if r.choices[0].logprobs and r.choices[0].logprobs.content:
            for tok in r.choices[0].logprobs.content:
                if tok and tok.logprob is not None: lp.append(tok.logprob)
        return {"text":text,"token_logprobs":lp}
    return _gen

def compute_recall_at_5(ranked, gold):
    return 1.0 if gold in ranked[:5] else 0.0

def compute_mrr(ranked, gold):
    for i,d in enumerate(ranked, start=1):
        if d==gold: return 1.0/i
    return 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["nq","beir"], required=True)
    ap.add_argument("--poison_ratio", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--zkip", type=str, default="true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--gen_model", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args=ap.parse_args()
    
    random.seed(args.seed); np.random.seed(args.seed)
    data=load_dataset(args.dataset,args.poison_ratio)
    # Build corpus from passage files if candidates are not included in dataset
    if args.dataset == "nq":
        passage_path = "RAGuard/datasets/clean/nq_passages.jsonl"
    else:
        passage_path = "RAGuard/datasets/clean/beir_passages.jsonl"

    import json
    corpus = []
    with open(passage_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            if "text" in obj:
                corpus.append(obj["text"])
            elif "passage" in obj:
                corpus.append(obj["passage"])

    
    generator=openai_generator(args.gen_model, temp=0.2, top_p=0.95, max_tokens=128)
    embedder=SentenceTransformer("all-MiniLM-L6-v2")
    def embed_answer(s): return embedder.encode([s])[0]
    
    rec, mrr, asr, calls=[],[],[],[]
    
    for ex in tqdm(data):
        q = ex.get("query", "")
        gold = ex.get("gold_doc", "") or ex.get("answer", "")
        cand = ex.get("candidates", ex.get("passages", []))
        ranked=bm25_topk(q,cand,k=args.topk)
        base=generator(q, ranked)
        wrong=(gold.lower() not in base["text"].lower())
        atk=wrong
        used=ranked[:]; c=1
        
        if args.zkip=="true":
            drop,_=zkip_scores(q, used, generator, embed_answer)
            used=[d for i,d in enumerate(used) if i not in set(drop)]
            c+=len(ranked)
            after=generator(q, used)["text"]
            if gold.lower() in after.lower(): atk=False
        
        rec.append(compute_recall_at_5(ranked, gold))
        mrr.append(compute_mrr(ranked, gold))
        asr.append(1 if atk else 0)
        calls.append(c)
    
    out=dict(
        dataset=args.dataset,
        poison_ratio=args.poison_ratio,
        recall5=float(np.mean(rec)),
        mrr=float(np.mean(mrr)),
        asr=float(np.mean(asr)),
        gen_calls=float(np.mean(calls)),
        n=len(data),
        model=args.gen_model,
        seed=args.seed
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w") as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()
