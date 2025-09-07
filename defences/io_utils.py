import json

def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def load_queries(path):
    # rows: {"qid","query","gold_answers":[...]}
    return list(read_jsonl(path))

def load_passages(path):
    # rows: {"doc_id","doc_text"}
    return list(read_jsonl(path))

def load_runs(path):
    # dict: qid -> [doc_ids]
    with open(path, "r") as f:
        return json.load(f)

def load_poison_flags(path):
    # rows: {"doc_id","is_poison":bool,"poison_type":str}
    if not path: return {}
    out = {}
    for r in read_jsonl(path):
        out[r["doc_id"]] = {"is_poison": r.get("is_poison", False),
                            "poison_type": r.get("poison_type")}
    return out
