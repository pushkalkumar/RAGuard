import json, torch, os, random, argparse
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm

class RetrieverDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = []
            for line in f:
                try:
                    item = json.loads(line.strip())
                    self.data.append(item)
                except json.JSONDecodeError:
                    continue
        
        if any("poison_type" in d for d in self.data):
            self.clean_docs = [d["gold_doc"] for d in self.data if d.get("poison_type") == "clean"]
        else:
            self.clean_docs = [d["gold_doc"] for d in self.data if "gold_doc" in d]
        
        if not self.clean_docs:
            raise ValueError(f"No clean docs found in {path}. Check file format!")

        print(f"Loaded {len(self.data)} items from {path} | {len(self.clean_docs)} clean docs found.")

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        gold_doc = item["gold_doc"]

        if "poison_doc" in item and item["poison_type"] != "clean":
            neg_doc = item["poison_doc"]
        else:
            neg_doc = random.choice(self.clean_docs)
        
        return query, gold_doc, neg_doc


class DenseRetrieverModel(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
        outputs = self.encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings


def train_dense_retriever(train_path, save_dir, lr=2e-5, batch_size=16, epochs=3, margin=0.2, model_name="sentence-transformers/all-MiniLM-L6-v2"):

    dataset = RetrieverDataset(train_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DenseRetrieverModel(model_name=model_name)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.TripletMarginLoss(margin=margin)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for queries, pos_docs, neg_docs in tqdm(loader, desc=f"Epoch {epoch+1}"):
            q_emb = model.encode(list(queries))
            p_emb = model.encode(list(pos_docs))
            n_emb = model.encode(list(neg_docs))

            loss = loss_fn(q_emb, p_emb, n_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")

    os.makedirs(save_dir, exist_ok=True)
    model.encoder.save_pretrained(save_dir)
    model.tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Dense Retriever")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for triplet loss")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Pretrained model name")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_dense_retriever(**vars(args))