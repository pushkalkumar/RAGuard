import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report
import joblib

FEATURE_KEYS = ["rank_pos","flip_wrong_to_right","em_delta","f1_delta","entropy_delta","sim_q","sim_ans","sim_z"]

def load_feature_table(path):
    X, y = [], []
    with open(path, "r") as f:
        for line in f:
            r = json.loads(line)
            if "label_is_poison" not in r:
                # skip unlabeled rows
                continue
            feats = []
            for k in FEATURE_KEYS:
                v = r.get(k, 0.0)
                if v is None: v = 0.0
                feats.append(float(v))
            X.append(feats)
            y.append(int(r["label_is_poison"]))
    return np.array(X, dtype=float), np.array(y, dtype=int)

def train_logreg(X, y, C=1.0):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=C)
    clf.fit(X, y)
    return clf

def evaluate(clf, X, y):
    probs = clf.predict_proba(X)[:,1]
    preds = (probs >= 0.5).astype(int)
    pr, rc, _ = precision_recall_curve(y, probs)
    auprc = average_precision_score(y, probs)
    rep = classification_report(y, preds, output_dict=True, zero_division=0)
    return {"AUPRC": float(auprc),
            "precision@0.5": float(rep["1"]["precision"]),
            "recall@0.5": float(rep["1"]["recall"])}

def save_model(clf, path):
    joblib.dump(clf, path)
