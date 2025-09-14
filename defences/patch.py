from dataclasses import dataclass
from typing import List, Dict
from defences.scorer import em, f1
from defences.embedder import Embedder
from defences.features import entropy_from_token_logprobs, zscores, pack_feature_row

@dataclass
class PatchConfig:
    k: int = 10
    leave_one_out_cap: int = 8
    early_exit: bool = True
    model_name: str = "google/flan-t5-small"
    batch_counterfactuals: bool = True  # new

class QAGenerator:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def answer(self, question: str, passages: List[str]) -> Dict:
        prompt = "Question: " + question + "\n\nContext:\n" + "\n\n".join(passages) + "\n\nAnswer:"
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        out = self.model.generate(
            **inputs,
            max_new_tokens=48,
            return_dict_in_generate=True,
            output_scores=True
        )
        text = self.tok.decode(out.sequences[0], skip_special_tokens=True)
        # estimate token logprobs from decoder scores
        token_logprobs = []
        if out.scores:
            import torch
            for i, scores in enumerate(out.scores):
                ids = out.sequences[:, inputs.input_ids.shape[-1] + i]  # generated ids
                lp = torch.log_softmax(scores, dim=-1)[0, ids[0]].item()
                token_logprobs.append(lp)
        return {"text": text, "token_logprobs": token_logprobs}

class ZeroKnowledgePatch:
    def __init__(self, qa: QAGenerator, passages_by_id: dict[str,str], cfg: PatchConfig):
        self.qa = qa
        self.passages_by_id = passages_by_id
        self.cfg = cfg
        self.embedder = Embedder()

    def _context(self, ctx_ids):
        return [self.passages_by_id[d] for d in ctx_ids if d in self.passages_by_id]

    def _batch_answers(self, question: str, list_of_ctx_lists: List[List[str]]):
        # batched generation to save calls
        prompts = [
            "Question: " + question + "\n\nContext:\n" + "\n\n".join(ctx) + "\n\nAnswer:"
            for ctx in list_of_ctx_lists
        ]
        tok = self.qa.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        out = self.qa.model.generate(
            **tok,
            max_new_tokens=48,
            return_dict_in_generate=True,
            output_scores=True
        )
        texts = [self.qa.tok.decode(seq, skip_special_tokens=True) for seq in out.sequences]
        # approximate per-sequence token logprob by averaging per step argmax token logprob
        import torch
        seq_len = out.sequences.shape[1] - tok.input_ids.shape[1]
        lp_steps = []
        for step_scores in out.scores:  # list[tensor(batch,vocab)]
            lp = torch.log_softmax(step_scores, dim=-1)
            # take token actually generated
            # out.sequences has shape (batch, input_len+gen_len)
            # we align step index across batch
            lp_steps.append(lp)
        token_lp = []
        for b in range(out.sequences.shape[0]):
            lps = []
            for i, scores in enumerate(out.scores):
                gen_id = out.sequences[b, tok.input_ids.shape[1] + i]
                lps.append(scores[b].log_softmax(dim=-1)[gen_id].item())
            token_lp.append(lps)
        return [{"text": t, "token_logprobs": token_lp[i]} for i, t in enumerate(texts)]

    def run_one(self, q: dict, ranked_doc_ids: list[str]) -> dict:
        k = min(self.cfg.k, len(ranked_doc_ids))
        ctx_ids = ranked_doc_ids[:k]
        ctx = self._context(ctx_ids)

        base = self.qa.answer(q["query"], ctx)
        base_em = em(base["text"], q["gold_answers"])
        base_f1 = f1(base["text"], q["gold_answers"])
        base_entropy = entropy_from_token_logprobs(base.get("token_logprobs", []))
        calls = 1
        tested, flagged = [], []

        # precompute embeddings to score similarity features
        q_emb = self.embedder.encode([q["query"]])[0]
        ctx_embs = self.embedder.encode([self.passages_by_id[d] for d in ctx_ids])
        ans_emb = self.embedder.encode([base["text"]])[0]
        # similarity of each passage to question and to base answer
        sim_q_list = [self.embedder.cos(e, q_emb) for e in ctx_embs]
        sim_ans_list = [self.embedder.cos(e, ans_emb) for e in ctx_embs]
        sim_z_list = zscores(sim_q_list)

        feature_rows = []

        if base_em == 0:
            # build batched LOO contexts
            subcontexts = []
            sub_idx = []
            for i, doc_id in enumerate(ctx_ids):
                if i >= self.cfg.leave_one_out_cap:
                    break
                if doc_id not in self.passages_by_id:
                    continue
                tested.append(doc_id)
                sub_ctx_ids = [d for j, d in enumerate(ctx_ids) if j != i]
                subcontexts.append(self._context(sub_ctx_ids))
                sub_idx.append(i)

            if self.cfg.batch_counterfactuals and subcontexts:
                alts = self._batch_answers(q["query"], subcontexts)
                calls += 1  # count one batched call
                for idx, alt in zip(sub_idx, alts):
                    alt_em = em(alt["text"], q["gold_answers"])
                    alt_f1 = f1(alt["text"], q["gold_answers"])
                    alt_entropy = entropy_from_token_logprobs(alt.get("token_logprobs", []))
                    flip = (alt_em == 1)
                    if flip and self.cfg.early_exit:
                        flagged.append(ctx_ids[idx])
                        kept = [self.passages_by_id[d] for j,d in enumerate(ctx_ids) if j != idx]
                        final = self.qa.answer(q["query"], kept)
                        calls += 1
                        final_em = em(final["text"], q["gold_answers"])
                        # log feature for the flagged doc
                        feature_rows.append(
                          pack_feature_row(q["qid"], ctx_ids[idx], idx+1, True,
                                           1 - base_em, max(0.0, alt_f1 - base_f1),
                                           base_entropy, alt_entropy,
                                           sim_q_list[idx], sim_ans_list[idx], sim_z_list[idx])
                        )
                        return {
                            "qid": q["qid"], "base_answer": base["text"], "base_em": base_em,
                            "tested": tested, "flagged": flagged,
                            "final_answer": final["text"], "final_em": final_em,
                            "num_generator_calls": calls,
                            "feature_rows": feature_rows
                        }
                    # even without early exit, log feature row
                    feature_rows.append(
                      pack_feature_row(q["qid"], ctx_ids[idx], idx+1, flip,
                                       alt_em - base_em, alt_f1 - base_f1,
                                       base_entropy, alt_entropy,
                                       sim_q_list[idx], sim_ans_list[idx], sim_z_list[idx])
                    )
            else:
                # non-batched loop
                for i, doc_id in enumerate(ctx_ids):
                    if i >= self.cfg.leave_one_out_cap: break
                    if doc_id not in self.passages_by_id: continue
                    tested.append(doc_id)
                    sub_ctx = [self.passages_by_id[d] for j, d in enumerate(ctx_ids) if j != i]
                    alt = self.qa.answer(q["query"], sub_ctx)
                    calls += 1
                    alt_em = em(alt["text"], q["gold_answers"])
                    alt_f1 = f1(alt["text"], q["gold_answers"])
                    alt_entropy = entropy_from_token_logprobs(alt.get("token_logprobs", []))
                    flip = (alt_em == 1)
                    feature_rows.append(
                      pack_feature_row(q["qid"], doc_id, i+1, flip,
                                       alt_em - base_em, alt_f1 - base_f1,
                                       base_entropy, alt_entropy,
                                       sim_q_list[i], sim_ans_list[i], sim_z_list[i])
                    )
                    if flip and self.cfg.early_exit:
                        flagged.append(doc_id)
                        kept = [self.passages_by_id[d] for d in ctx_ids if d != doc_id]
                        final = self.qa.answer(q["query"], kept)
                        calls += 1
                        return {
                            "qid": q["qid"], "base_answer": base["text"], "base_em": base_em,
                            "tested": tested, "flagged": flagged,
                            "final_answer": final["text"], "final_em": em(final["text"], q["gold_answers"]),
                            "num_generator_calls": calls,
                            "feature_rows": feature_rows
                        }

        return {
            "qid": q["qid"], "base_answer": base["text"], "base_em": base_em,
            "tested": tested, "flagged": flagged,
            "final_answer": base["text"], "final_em": base_em,
            "num_generator_calls": calls,
            "feature_rows": feature_rows
        }
