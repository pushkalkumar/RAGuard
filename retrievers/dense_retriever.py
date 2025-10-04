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
                # correct key name: datasets use "poison_doc" (not "poisoned_doc")
                if "poison_doc" in item and item["poison_doc"] != item["gold_doc"]:
                    # append exact poison string so eval can match by equality
                    self.docs.append(item["poison_doc"])

        # Load embedding model
        self.model = SentenceTransformer(model_name)

        # Precompute embeddings for all docs
        self.doc_embeddings = self.model.encode(self.docs, convert_to_tensor=True)

    def search(self, query, k=5):
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.doc_embeddings)[0]
        topk = torch.topk(scores, k)
        # topk.indices/values are tensors; convert indices to int to access python list reliably
        results = [(self.docs[int(i)], float(scores[int(i)])) for i in topk.indices]
        return results
