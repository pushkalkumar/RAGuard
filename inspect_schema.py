import json

path = "RAGuard/datasets/clean/nq.jsonl"
with open(path, "r") as f:
    first = json.loads(next(f))
    print(first)

