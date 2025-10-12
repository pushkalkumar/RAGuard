import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Dict, List, Any

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(items: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def group_by_field(items: Iterable[Dict[str, Any]], field: str) -> Dict[str, List[Dict[str, Any]]]:
    groups = defaultdict(list)
    for it in items:
        key = it.get(field)
        # fallback: for nested or missing keys use str(it) to avoid collapsing many into None if user chooses wrong field
        if key is None:
            key = json.dumps(it, sort_keys=True) if field == "__full__" else "NULL"
        groups[str(key)].append(it)
    return groups

def split_groups(
    group_keys: List[str],
    ratios: List[float],
    rng: random.Random
) -> List[List[str]]:
    assert len(ratios) == 3, "expected three ratios (train,val,test)"
    n = len(group_keys)
    rng.shuffle(group_keys)
    t = int(round(ratios[0] * n))
    v = int(round(ratios[1] * n))
    # ensure sum == n
    if t + v >= n:
        t = max(0, min(n - 2, t))
        v = max(0, min(n - 1 - t, v))
    te = n - t - v
    train_keys = group_keys[:t]
    val_keys = group_keys[t : t + v]
    test_keys = group_keys[t + v :]
    return [train_keys, val_keys, test_keys]

def main():
    ap = argparse.ArgumentParser(prog="make_splits.py", description="Create train/val/test JSONL splits from a JSONL dataset")
    ap.add_argument("--input", "-i", required=True, help="input JSONL file")
    ap.add_argument("--out-dir", "-o", required=True, help="output directory for splits")
    ap.add_argument("--ratios", "-r", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="three floats: train val test (must sum ~1.0)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for deterministic splits")
    ap.add_argument("--group-by", default="query", help="field name to group by (default: 'query'). Use '__full__' to avoid grouping (each row is its own group).")
    ap.add_argument("--min-groups", type=int, default=3, help="require at least this many groups to split")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    assert in_path.exists(), f"input not found: {in_path}"
    ratios = args.ratios
    s = sum(ratios)
    if not (0.999 <= s <= 1.001):
        # normalize if user provided approximate ratios
        ratios = [r / s for r in ratios]

    items = list(read_jsonl(in_path))
    if not items:
        raise SystemExit("no items found in input")

    groups = group_by_field(items, args.group_by)
    n_groups = len(groups)
    if n_groups < args.min_groups:
        raise SystemExit(f"too few groups ({n_groups}) using --group-by {args.group_by}; try a different field")

    rng = random.Random(args.seed)
    keys = list(groups.keys())
    train_k, val_k, test_k = split_groups(keys, ratios, rng)

    def flatten(keys_list: List[str]) -> List[Dict[str, Any]]:
        out = []
        for k in keys_list:
            out.extend(groups[k])
        return out

    train_items = flatten(train_k)
    val_items = flatten(val_k)
    test_items = flatten(test_k)

    write_jsonl(train_items, out_dir / "train.jsonl")
    write_jsonl(val_items, out_dir / "dev.jsonl")
    write_jsonl(test_items, out_dir / "test.jsonl")

    meta = {
        "input": str(in_path),
        "n_rows": len(items),
        "group_by": args.group_by,
        "n_groups": n_groups,
        "seed": args.seed,
        "ratios": ratios,
        "train_groups": len(train_k),
        "dev_groups": len(val_k),
        "test_groups": len(test_k),
        "train_rows": len(train_items),
        "dev_rows": len(val_items),
        "test_rows": len(test_items),
    }
    write_jsonl([meta], out_dir / "split_metadata.jsonl")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()