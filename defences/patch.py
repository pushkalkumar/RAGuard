from dataclasses import dataclass
from typing import List, Dict
from defences.scorer import em

@dataclass
class PatchConfig:
    k: int = 10
    leave_one_out_cap: int = 8
    early_exit: bool = True
    model_name: str = "google/flan-t5-small" 

class QAGenerator:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def answer(self, question: str, passages: List[str]) -> Dict:
        prompt = "Question: " + question + "\n\nContext:\n" + "\n\n".join(passages) + "\n\nAnswer:"
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        ids = self.model.generate(**inputs, max_new_tokens=48, temperature=0.2)
        text = self.tok.decode(ids[0], skip_special_tokens=True)
        return {"text": text}

class ZeroKnowledgePatch:
    def __init__(self, qa: QAGenerator, passages_by_id: dict[str,str], cfg: PatchConfig):
        self.qa = qa
        self.passages_by_id = passages_by_id
        self.cfg = cfg

    def run_one(self, q: dict, ranked_doc_ids: list[str]) -> dict:
        k = min(self.cfg.k, len(ranked_doc_ids))
        ctx_ids = ranked_doc_ids[:k]
        ctx = [self.passages_by_id[d] for d in ctx_ids if d in self.passages_by_id]

        base = self.qa.answer(q["query"], ctx)
        base_em = em(base["text"], q["gold_answers"])
        calls = 1
        tested, flagged = [], []

        if base_em == 0:
            for i, doc_id in enumerate(ctx_ids):
                if i >= self.cfg.leave_one_out_cap:
                    break
                if doc_id not in self.passages_by_id:
                    continue
                tested.append(doc_id)
                sub_ctx = [self.passages_by_id[d] for j, d in enumerate(ctx_ids) if j != i and d in self.passages_by_id]
                alt = self.qa.answer(q["query"], sub_ctx)
                calls += 1
                if em(alt["text"], q["gold_answers"]) == 1:
                    flagged.append(doc_id)
                    if self.cfg.early_exit:
                        kept_ctx = [self.passages_by_id[d] for d in ctx_ids if d not in flagged and d in self.passages_by_id]
                        final = self.qa.answer(q["query"], kept_ctx)
                        calls += 1
                        return {
                            "qid": q["qid"], "base_answer": base["text"], "base_em": base_em,
                            "tested": tested, "flagged": flagged,
                            "final_answer": final["text"], "final_em": em(final["text"], q["gold_answers"]),
                            "num_generator_calls": calls
                        }

        return {
            "qid": q["qid"], "base_answer": base["text"], "base_em": base_em,
            "tested": tested, "flagged": flagged,
            "final_answer": base["text"], "final_em": base_em,
            "num_generator_calls": calls
        }
