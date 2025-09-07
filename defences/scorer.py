import re

def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return " ".join(s.split())

def em(pred: str, golds: list[str]) -> int:
    p = _norm(pred)
    return int(any(p == _norm(g) for g in golds))

def f1(pred: str, golds: list[str]) -> float:
    p = _norm(pred).split()
    if not p: return 0.0
    best = 0.0
    for g in golds:
        g = _norm(g).split()
        if not g: 
            continue
        common = {}
        for t in set(p):
            common[t] = min(p.count(t), g.count(t))
        overlap = sum(common.values())
        prec = overlap / len(p)
        rec = overlap / len(g)
        if prec + rec > 0:
            best = max(best, 2 * prec * rec / (prec + rec))
    return best
