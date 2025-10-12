import argparse, json, random, hashlib, sys
from pathlib import Path
from typing import List, Dict, Tuple

# try to reuse project loaders if available
try:
    from defences.io_utils import load_queries, load_passages
except Exception:
    load_queries = None
    load_passages = None

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def read_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

def write_jsonl(path: Path, items: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def load_items(path: str, loader):
    p = Path(path)
    if loader:
        try:
            return list(loader(path))
        except Exception:
            pass
    if not p.exists():
        return []
    return read_jsonl(p)

def build_passage_pool(passages: List[Dict]) -> Dict[str,str]:
    pool = {}
    for p in passages:
        pid = p.get("doc_id") or p.get("id") or p.get("passage_id") or p.get("pid")
        ptext = p.get("doc_text") or p.get("text") or p.get("passage") or p.get("content")
        if pid and ptext:
            pool[pid] = ptext
    return pool

def detect_query_text(q: Dict) -> str:
    return q.get("query") or q.get("question") or q.get("text") or q.get("q") or ""

def detect_gold(q: Dict, passages_by_id: Dict[str,str]) -> Tuple[str, str]:
    gold_id = q.get("gold_doc_id") or q.get("doc_id") or q.get("document_id") or q.get("gold_id")
    gold_text = None
    if gold_id and isinstance(gold_id, str):
        gold_text = passages_by_id.get(gold_id)
    if not gold_text:
        cand = q.get("gold_doc") or q.get("gold_doc_text") or q.get("doc_text") or q.get("document") or q.get("passage")
        if isinstance(cand, str) and cand.strip():
            gold_text = cand
            gold_id = "doc_" + sha1(gold_text)[:12]
    return gold_id, gold_text

def normalize_queries(raw_q: List[Dict], passages_by_id: Dict[str,str]) -> List[Dict]:
    out = []
    for i, q in enumerate(raw_q):
        rec = dict(q)
        if not rec.get("qid"):
            rec["qid"] = rec.get("id") or f"q{i+1}"
        if not rec.get("query"):
            rec["query"] = detect_query_text(rec)
        gold_id, gold_text = detect_gold(rec, passages_by_id)
        rec["doc_id"] = gold_id
        rec["doc_text"] = gold_text
        out.append(rec)
    return out

def group_by_doc(norm_q: List[Dict]) -> List[List[Dict]]:
    groups = {}
    for q in norm_q:
        docid = q.get("doc_id") or ("doc_none_" + sha1(q.get("query","")+q.get("qid",""))[:12])
        groups.setdefault(docid, []).append(q)
    return list(groups.values())

def allocate_groups(groups: List[List[Dict]], total_queries: int, ratios: List[float], rng: random.Random):
    targets = [int(r * total_queries) for r in ratios]
    diff = total_queries - sum(targets)
    if diff:
        targets[0] += diff
    splits = {"train": [], "dev": [], "test": []}
    counts = {"train": 0, "dev": 0, "test": 0}
    split_names = ["train","dev","test"]
    for g in groups:
        remaining = [targets[i] - counts[split_names[i]] for i in range(3)]
        if all(r <= 0 for r in remaining):
            splits["train"].extend(g)
            counts["train"] += len(g)
            continue
        idx = max(range(3), key=lambda i: remaining[i])
        splits[split_names[idx]].extend(g)
        counts[split_names[idx]] += len(g)
    return splits["train"], splits["dev"], splits["test"]

def make_triples(train: List[Dict], doc_pool: Dict[str,str], neg_per_query: int, rng: random.Random) -> List[Dict]:
    all_ids = list(doc_pool.keys())
    triples = []
    for r in train:
        gold = r.get("doc_id")
        if not gold:
            continue
        candidates = [d for d in all_ids if d != gold]
        if not candidates:
            continue
        if len(candidates) >= neg_per_query:
            negs = rng.sample(candidates, neg_per_query)
        else:
            negs = [rng.choice(candidates) for _ in range(neg_per_query)]
        for n in negs:
            triples.append({
                "qid": r.get("qid"),
                "query": r.get("query"),
                "gold_doc_id": gold,
                "gold_doc_text": doc_pool.get(gold),
                "negative_doc_id": n,
                "negative_doc_text": doc_pool.get(n)
            })
    return triples

def create_splits(queries_path: str, passages_path: str, out_dir: str, ratios=(0.8,0.1,0.1), seed: int = 42,
                  toy_size: int = 100, neg_per_query: int = 1, force: bool = False) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # check idempotence
    if not force and (out / "train.jsonl").exists() and (out / "dev.jsonl").exists() and (out / "test.jsonl").exists():
        return {"skipped": True, "message": f"Splits already exist in {out_dir}. Use --force to overwrite."}

    load_items = None  # placeholder to help static analysis
    raw_q = None

    # load using project loaders if available, else raw read
    try:
        raw_q = load_items_func = (load_queries and list(load_queries(queries_path))) or read_jsonl(Path(queries_path))
    except Exception:
        raw_q = read_jsonl(Path(queries_path))
    try:
        raw_p = load_passages_func = (load_passages and list(load_passages(passages_path))) or read_jsonl(Path(passages_path))
    except Exception:
        raw_p = read_jsonl(Path(passages_path))

    if not raw_q:
        raise SystemExit(f"No queries found in {queries_path}")
    if not raw_p:
        raise SystemExit(f"No passages found in {passages_path}")

    passages_by_id = build_passage_pool(raw_p)
    norm_q = normalize_queries(raw_q, passages_by_id)

    groups = group_by_doc(norm_q)
    rng = random.Random(seed)
    rng.shuffle(groups)

    train, dev, test = allocate_groups(groups, len(norm_q), list(ratios), rng)

    for r in train: r["split"] = "train"
    for r in dev: r["split"] = "dev"
    for r in test: r["split"] = "test"

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "dev.jsonl", dev)
    write_jsonl(out / "test.jsonl", test)
    if toy_size and toy_size > 0:
        write_jsonl(out / "toy.jsonl", train[:toy_size])

    triples = make_triples(train, passages_by_id, neg_per_query, rng)
    write_jsonl(out / "training_triples.jsonl", triples)

    meta = {
        "total_queries": len(norm_q),
        "train": len(train),
        "dev": len(dev),
        "test": len(test),
        "unique_docs": len(passages_by_id),
        "seed": seed,
        "ratios": ratios,
        "toy_size": toy_size,
        "neg_per_query": neg_per_query
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return {"skipped": False, "meta": meta}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--passages", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8,0.1,0.1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--toy-size", type=int, default=100)
    ap.add_argument("--neg-per-query", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="Overwrite existing splits")
    args = ap.parse_args()

    res = create_splits(args.queries, args.passages, args.out_dir, tuple(args.ratios), args.seed, args.toy_size, args.neg_per_query, args.force)
    if res.get("skipped"):
        print(res["message"])
        sys.exit(0)
    print("Wrote splits:", res["meta"])

if __name__ == "__main__":
    main()