import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import os

POISON_TYPES = ["fabricated", "contradiction", "reasoning"]

def fabricate(doc):
    return doc + " NOTE: The event happened in Atlantis in 3022."

def contradict(doc):
    return doc.replace("yes", "no").replace("true", "false")

def reasoning_trap(doc):
    return doc + " However, experts disagree on the exact details."

def poison_doc(doc, poison_type):
    if poison_type == "fabricated":
        return fabricate(doc)
    elif poison_type == "contradiction":
        return contradict(doc)
    elif poison_type == "reasoning":
        return reasoning_trap(doc)
    return doc

def main(input_path, output_path, poison_ratio=0.3):
    with open(input_path, "r") as f:
        data = [json.loads(l) for l in f]
    n_poison = int(len(data) * poison_ratio)
    poison_indices = set(random.sample(range(len(data)), n_poison))
    out = []

    for i, ex in enumerate(tqdm(data, desc="Generating poisons")):
        query = ex["query"]
        gold_doc = ex["gold_doc"]
        if i in poison_indices:
            ptype = random.choice(POISON_TYPES)
            poisoned = poison_doc(gold_doc, ptype)
            
        else:
            ptype = "clean"
            poisoned = gold_doc
        out.append({
                "query": query,
                "gold_doc": gold_doc,
                "poison_doc": poisoned,
                "poison_type": ptype
            })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in out:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratio", type=float, default=0.3)
    args = parser.parse_args()
    main(args.input, args.output, args.ratio)
