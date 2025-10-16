from sentence_transformers import SentenceTransformer, util
import torch
import json
from transformers import AutoModel, AutoTokenizer
import os

class DenseRetriever:
    def __init__(self, dataset_path, model_dir=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Load dataset
        self.docs = []
        with open(dataset_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.docs.append(item["gold_doc"])
                if "poison_doc" in item and item["poison_doc"] != item["gold_doc"]:
                    self.docs.append(item["poison_doc"])

        # Try to load fine-tuned model from model_dir if provided
        if model_dir and os.path.exists(model_dir):
            try:
                self.encoder = AutoModel.from_pretrained(model_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
                self.is_custom = True
                print(f"Loading trained model from {model_dir}")
            except Exception as e:
                print(f"Could not load from {model_dir}: {e}. Falling back to pretrained model.")
                self.model = SentenceTransformer(model_name)
                self.is_custom = False
        else:
            print(f"Using pretrained model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.is_custom = False

        # Precompute embeddings for all docs
        print(f"Encoding {len(self.docs)} documents...")
        self.doc_embeddings = self.encode(self.docs)

        # Safety check
        assert len(self.doc_embeddings) == len(self.docs), (
            f"Embedding/doc mismatch: {len(self.doc_embeddings)} vs {len(self.docs)}"
        )

    def encode(self, texts, batch_size=64):
        if self.is_custom:
            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256)
                with torch.no_grad():
                    outputs = self.encoder(**inputs)
                batch_emb = outputs.last_hidden_state.mean(dim=1)
                all_embs.append(batch_emb)
            embeddings = torch.cat(all_embs, dim=0)
        else:
            embeddings = self.model.encode(texts, convert_to_tensor=True, batch_size=batch_size)
        return embeddings

    def search(self, query, k=5):
        if self.is_custom:
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=256)
            with torch.no_grad():
                outputs = self.encoder(**inputs)
            query_emb = outputs.last_hidden_state.mean(dim=1)
        else:
            query_emb = self.model.encode(query, convert_to_tensor=True)

        scores = util.cos_sim(query_emb, self.doc_embeddings)[0]
        k = min(k, len(self.docs))
        topk = torch.topk(scores, k)
        results = [(self.docs[int(i)], float(scores[int(i)])) for i in topk.indices]
        return results
