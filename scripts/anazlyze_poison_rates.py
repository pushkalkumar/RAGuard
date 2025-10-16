import json

path = "datasets/poisoned/beir_poisoned.jsonl"  # change this if needed

total, poisoned = 0, 0
types = {}

with open(path, "r") as f:
    for line in f:
        item = json.loads(line)
        total += 1

        poison_doc = item.get("poison_doc")
        gold_doc = item["gold_doc"]
        poison_type = item.get("poison_type", "clean")

        if poison_doc and poison_doc.strip() != gold_doc.strip():
            poisoned += 1
            types[poison_type] = types.get(poison_type, 0) + 1

print(f"📊 Total samples: {total}")
print(f"🧪 Poisoned samples: {poisoned}")
print(f"📈 Poison rate: {poisoned / total:.2%}")

if types:
    print("\nPoison breakdown by type:")
    for t, c in types.items():
        print(f"  {t}: {c} ({c/total:.2%})")
