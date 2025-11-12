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
    batch_counterfactuals: bool = True  # batching on

class QAGenerator:
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
        self.config = AutoConfig.from_pretrained(model_name)
        self.is_encdec = bool(getattr(self.config, "is_encoder_decoder", False))
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # pick the right class automatically
        if self.is_encdec:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def _decode_with_logprobs(self, inputs, gen_kwargs=None):
        """Generate text and per-step chosen-token logprobs in a model-agnostic way."""
        import torch
        if gen_kwargs is None:
            gen_kwargs = {}
        out = self.model.generate(
            **inputs,
            max_new_tokens=48,
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
        )
        text = self.tok.decode(out.sequences[0], skip_special_tokens=True)

        # Determine offset: for enc-dec, sequences are only generated tokens; for causal, sequences include prompt.
        if self.is_encdec:
            offset = 0
        else:
            # for causal models, generated tokens start after the input length
            offset = inputs.input_ids.shape[-1]

        token_logprobs = []
        # out.scores is a list of tensors, one per generated step
        # chosen token at step t is out.sequences[:, offset + t]
        for t, step_scores in enumerate(out.scores):
            # log_softmax over vocab
            logp = step_scores.log_softmax(dim=-1)
            chosen = out.sequences[:, offset + t]
            # batch size is 1 for our calls
            token_logprobs.append(logp[0, chosen[0]].item())

        return {"text": text, "token_logprobs": token_logprobs}

    def answer(self, question: str, passages: List[str]) -> Dict:
        prompt = "Question: " + question + "\n\nContext:\n" + "\n\n".join(passages) + "\n\nAnswer:"
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        return self._decode_with_logprobs(inputs)

class ZeroKnowledgePatch:
    def __init__(self, qa: QAGenerator, passages_by_id: dict[str,str], cfg: PatchConfig):
        self.qa = qa
        self.passages_by_id = passages_by_id
        self.cfg = cfg
        self.embedder = Embedder()

    def _context(self, ctx_ids):
        return [self.passages_by_id[d] for d in ctx_ids if d in self.passages_by_id]

    def _batch_answers(self, question: str, list_of_ctx_lists: List[List[str]]):
        """Batched generation for multiple contexts; returns list of dicts like answer()."""
        import torch
        prompts = [
            "Question: " + question + "\n\nContext:\n" + "\n\n".join(ctx) + "\n\nAnswer:"
            for ctx in list_of_ctx_lists
        ]
        tok = self.qa.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        out = self.qa.model.generate(
            **tok,
            max_new_tokens=48,
            return_dict_in_generate=True,
            output_scores=True,
        )
        texts = [self.qa.tok.decode(seq, skip_special_tokens=True) for seq in out.sequences]

        # Offsets per batch item
        if self.qa.is_encdec:
            offsets = [0] * out.sequences.shape[0]
        else:
            offsets = [tok.input_ids.shape[1]] * out.sequences.shape[0]

        # Collect chosen-token logprobs per example
        token_lp_per_ex = [[] for _ in range(out.sequences.shape[0])]
        for t, step_scores in enumerate(out.scores):
            logp = step_scores.log_softmax(dim=-1)
            for b in range(out.sequences.shape[0]):
                chosen = out.sequences[b, offsets[b] + t]
                token_lp_per_ex[b].append(logp[b, chosen].item())

        return [{"text": texts[i], "token_logprobs": token_lp_per_ex[i]} for i in range(len(texts))]

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

        # embeddings for similarity features
        q_emb = self.embedder.encode([q["query"]])[0]
        ctx_embs = self.embedder.encode([self.passages_by_id[d] for d in ctx_ids])
        ans_emb = self.embedder.encode([base["text"]])[0]
        sim_q_list = [self.embedder.cos(e, q_emb) for e in ctx_embs]
        sim_ans_list = [self.embedder.cos(e, ans_emb) for e in ctx_embs]
        sim_z_list = zscores(sim_q_list)

        feature_rows = []

        if base_em == 0:
            # prepare LOO subcontexts
            subcontexts, sub_idx = [], []
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
                calls += 1
                for idx, alt in zip(sub_idx, alts):
                    alt_em = em(alt["text"], q["gold_answers"])
                    alt_f1 = f1(alt["text"], q["gold_answers"])
                    alt_entropy = entropy_from_token_logprobs(alt.get("token_logprobs", []))
                    flip = (alt_em == 1)
                    feature_rows.append(
                        pack_feature_row(q["qid"], ctx_ids[idx], idx+1, flip,
                                         alt_em - base_em, alt_f1 - base_f1,
                                         base_entropy, alt_entropy,
                                         sim_q_list[idx], sim_ans_list[idx], sim_z_list[idx])
                    )
                    if flip and self.cfg.early_exit:
                        flagged.append(ctx_ids[idx])
                        kept = [self.passages_by_id[d] for j, d in enumerate(ctx_ids) if j != idx]
                        final = self.qa.answer(q["query"], kept)
                        calls += 1
                        final_em = em(final["text"], q["gold_answers"])
                        return {
                            "qid": q["qid"], "base_answer": base["text"], "base_em": base_em,
                            "tested": tested, "flagged": flagged,
                            "final_answer": final["text"], "final_em": final_em,
                            "num_generator_calls": calls,
                            "feature_rows": feature_rows
                        }
            else:
                # non-batched LOO
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


# -------------------------------
# Zero-Knowledge Patch Wrapper
# -------------------------------

class ZKPatchDefense:
    """
    Wrapper around the patching logic for zero-knowledge adversarial detection.
    Provides a unified interface for patch-based evaluation.
    """
    def __init__(self, config_path=None):
        # Load configuration or use defaults
        self.config = PatchConfig()
        self.embedder = Embedder()
        self.qag = QAGenerator(self.config.model_name)

    def apply_patch(self, query: str) -> str:
        """
        Applies a minimal perturbation or counterfactual patch to a query
        to test the robustness of retrieval.
        """
        # For simplicity, just return the query unchanged here.
        # In practice, this would generate small semantic perturbations.
        return query

    def score_patch(self, query: str, passage: str) -> Dict[str, float]:
        """
        Computes entropy, similarity, and embedding-based divergence scores.
        """
        emb_q = self.embedder.encode([query])[0]
        emb_p = self.embedder.encode([passage])[0]
        import numpy as np
        sim = float(np.dot(emb_q, emb_p) / (np.linalg.norm(emb_q) * np.linalg.norm(emb_p)))
        return {"similarity": sim}

