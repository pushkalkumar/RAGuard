from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    @staticmethod
    def cos(a, b):
        return float(np.dot(a, b))
