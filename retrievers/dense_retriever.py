from sentence_transformers import SentenceTransformer, util
import json
import torch

class DenseRetriever:
    def __init__(self, dataset_path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Load dataset
        self.docs = []
        with open(dataset_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.docs.append(item["gold_doc"])

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Precompute embeddings for all docs
        self.doc_embeddings = self.model.encode(self.docs, convert_to_tensor=True)

    def search(self, query, k=5):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.doc_embeddings)[0]
        topk = torch.topk(scores, k)

        results = [(self.docs[i], float(scores[i])) for i in topk.indices]
        return results
