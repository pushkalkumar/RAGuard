import numpy as np

def cosine(a,b):
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float((a@b)/(na*nb))

def zkip_scores(query, docs, generator, embed_answer, lambda_=0.5, m=1):
    out_all=generator(query, docs)
    ya=out_all["text"]; pa=out_all["token_logprobs"]
    Ha=-np.mean(pa) if pa else 0.0
    ea=embed_answer(ya)
    scores=[]
    for i in range(len(docs)):
        subset=[d for j,d in enumerate(docs) if j!=i]
        out_i=generator(query, subset)
        yi=out_i["text"]; pi=out_i["token_logprobs"]
        Hi=-np.mean(pi) if pi else 0.0
        si=cosine(ea, embed_answer(yi))
        Ai=(1.0-si) + lambda_*max(0.0, Ha-Hi)
        scores.append(Ai)
    order=np.argsort(scores)
    drop=order[-m:]
    return drop.tolist(), scores
