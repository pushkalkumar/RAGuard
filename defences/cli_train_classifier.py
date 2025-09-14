import argparse, json
from sklearn.model_selection import train_test_split
from defences.classifier import load_feature_table, train_logreg, evaluate, save_model

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)     # JSONL with label_is_poison fields
ap.add_argument("--model_out", required=True)    # .pkl
ap.add_argument("--eval_out", required=True)     # .json
args = ap.parse_args()

X, y = load_feature_table(args.features)
if len(y) == 0:
    raise SystemExit("No labeled rows found in features. Provide poison flags when extracting features.")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
clf = train_logreg(Xtr, ytr)
metrics = evaluate(clf, Xte, yte)
save_model(clf, args.model_out)
with open(args.eval_out, "w") as f:
    f.write(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
