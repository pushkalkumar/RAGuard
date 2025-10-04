from rank_bm25 import BM25Okapi
import json

class BM25Retriever:
    def __init__(self, dataset_path):
        # load the dataset 
        self.docs = []
        with open(dataset_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.docs.append(item["gold_doc"])
                if "poison_doc" in item and item["poison_doc"] != item["gold_doc"]:
                    self.docs.append(item["poison_doc"])
        tokenized_docs = [doc.split(" ") for doc in self.docs] # tokenize docs
        self.bm25 = BM25Okapi(tokenized_docs) # initialize BM25 model

    def search(self, query, k=5):
        # Tokenize query
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        # Rank docs by score
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        results = [(self.docs[i], scores[i]) for i in ranked_idx[:k]]
        return results