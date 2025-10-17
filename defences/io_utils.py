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
    data = list(read_jsonl(path))
    # Handle both doc-level and query-level formats
    if all(k in data[0] for k in ("doc_id", "doc_text")):
        return data
    elif all(k in data[0] for k in ("query", "gold_doc")):
        # Convert queries to pseudo-passages
        passages = []
        for i, row in enumerate(data):
            passages.append({"doc_id": f"q{i}_gold", "doc_text": row["gold_doc"]})
            if "poison_doc" in row:
                passages.append({"doc_id": f"q{i}_poison", "doc_text": row["poison_doc"]})
        return passages
    else:
        raise ValueError(f"Unrecognized dataset schema: {list(data[0].keys())}")


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
