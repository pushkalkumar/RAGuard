# RAGuard: A Layered Adversarial Defense for RAG Pipelines with Zero-Knowledge Poison Detection

This repository is the official implementation of **RAGuard**, a defense system against adversarial poisoning attacks on Retrieval-Augmented Generation (RAG) pipelines using a novel Zero-Knowledge Identification Protocol (ZKIP).

## Overview

RAG systems are vulnerable to poisoning attacks where adversarial passages are injected into the corpus to manipulate model outputs. RAGuard detects and removes these poisoned passages **without prior knowledge of which documents are malicious** by measuring the influence of each retrieved passage on the generated answer.

### Key Innovation: Zero-Knowledge Identification Protocol (ZKIP)

ZKIP uses a leave-one-out approach to identify suspicious passages:
1. Generate an answer using all retrieved passages
2. For each passage, generate an answer without it
3. Compute an influence score based on:
   - **Answer embedding similarity** - how much the answer changes
   - **Entropy differential** - change in model confidence
4. Remove passages with highest influence scores (most likely poisoned)

## Requirements

To install dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python >= 3.8
- PyTorch >= 2.0
- Transformers >= 4.30
- Sentence-Transformers >= 2.2
- scikit-learn >= 1.0
- rank-bm25 >= 0.2
- OpenAI >= 1.0
- pandas >= 2.0
- matplotlib >= 3.7
- tqdm >= 4.65

**API Keys:**
Set your OpenAI API key for generation:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Training

To train a dense retriever on clean or poisoned data:

```bash
# Train on clean data
python retrievers/train.py \
    --train_path datasets/triples/train_triples_clean.jsonl \
    --save_dir checkpoints/contriever_clean \
    --epochs 3 \
    --batch_size 16 \
    --lr 2e-5

# Train on poisoned data (to simulate attacker's retriever)
python retrievers/train.py \
    --train_path datasets/triples/train_triples_nq_poisoned.jsonl \
    --save_dir checkpoints/contriever_poisoned_nq \
    --epochs 3 \
    --batch_size 16 \
    --lr 2e-5
```

**Training Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 2e-5 | Learning rate |
| `--batch_size` | 16 | Batch size |
| `--epochs` | 3 | Training epochs |
| `--margin` | 0.2 | Triplet loss margin |
| `--model_name` | `all-MiniLM-L6-v2` | Base encoder model |

## Evaluation

### Main Evaluation Script

Evaluate RAG pipeline with or without ZKIP defense:

```bash
# Evaluate on clean data (no defense needed)
python run_eval.py \
    --dataset nq \
    --poison_ratio 0 \
    --topk 5 \
    --zkip false \
    --gen_model gpt-4o-mini \
    --out results/nq_clean.json

# Evaluate on poisoned data WITHOUT defense
python run_eval.py \
    --dataset nq \
    --poison_ratio 0.1 \
    --topk 5 \
    --zkip false \
    --gen_model gpt-4o-mini \
    --out results/nq_poisoned_no_defense.json

# Evaluate on poisoned data WITH ZKIP defense
python run_eval.py \
    --dataset nq \
    --poison_ratio 0.1 \
    --topk 5 \
    --zkip true \
    --gen_model gpt-4o-mini \
    --out results/nq_poisoned_zkip.json
```

### Run All Experiments

To reproduce all paper results:

```bash
chmod +x run_all_evals.sh
./run_all_evals.sh
```

This runs evaluations across:
- **Datasets**: Natural Questions (NQ), BEIR (NFCorpus)
- **Poison ratios**: 0%, 10%, 20%, 30%
- **Defense**: With and without ZKIP
- **Multiple seeds**: For statistical significance

### Evaluation Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--dataset` | `nq`, `beir` | Evaluation dataset |
| `--poison_ratio` | 0.0 - 1.0 | Fraction of poisoned documents |
| `--topk` | integer | Number of passages to retrieve |
| `--zkip` | `true`, `false` | Enable ZKIP defense |
| `--gen_model` | string | OpenAI model for generation |
| `--seed` | integer | Random seed |

## Pre-trained Models

Download pre-trained models from the `checkpoints/` directory:

| Model | Description | Use Case |
|-------|-------------|----------|
| `contriever_clean/` | Dense retriever trained on clean data | Baseline retrieval |
| `contriever_poisoned_nq/` | Retriever trained with NQ poisoned data | Attack simulation |
| `contriever_poisoned_beir/` | Retriever trained with BEIR poisoned data | Attack simulation |
| `results/final/patch_classifier_model.pkl` | Trained poison classifier | Detection baseline |

**Loading a pre-trained retriever:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("checkpoints/contriever_clean")
embeddings = model.encode(["Your query here"])
```

## Results

Our defense achieves the following performance:

### Retrieval Performance (Clean Data)

| Dataset | Retriever | Recall@5 | MRR |
|---------|-----------|----------|-----|
| NQ | BM25 | 6.8% | 0.053 |
| NQ | Dense | 28.2% | 0.200 |
| BEIR | BM25 | 1.8% | 0.013 |
| BEIR | Dense | 1.9% | 0.014 |

### Defense Effectiveness

| Dataset | Poison Ratio | ASR (No Defense) | ASR (ZKIP) | ASR Reduction |
|---------|--------------|------------------|------------|---------------|
| NQ | 10% | ~45% | ~15% | **67%** |
| NQ | 20% | ~55% | ~20% | **64%** |
| NQ | 30% | ~65% | ~25% | **62%** |
| BEIR | 10% | ~40% | ~12% | **70%** |

**Metrics:**
- **Recall@5**: Percentage of queries where gold document is in top-5
- **MRR**: Mean Reciprocal Rank
- **ASR**: Attack Success Rate (lower is better with defense)

### Reproducing Results

```bash
# Generate all results
./run_all_evals.sh

# Plot results
python results/plot_results.py
python results/plot_llm_eval.py
```

## Project Structure

```
RAGuard/
├── README.md                 # This file
├── requirements.txt          # Dependencies
├── run_eval.py              # Main evaluation entry point
├── zkip.py                  # ZKIP algorithm implementation
├── run_all_evals.sh         # Batch experiment runner
│
├── checkpoints/             # Pre-trained model weights
│   ├── contriever_clean/
│   ├── contriever_poisoned_nq/
│   └── contriever_poisoned_beir/
│
├── datasets/                # Data files
│   ├── clean/              # Clean datasets (NQ, BEIR)
│   ├── poisoned/           # Poisoned variants
│   ├── triples/            # Training triples
│   └── splits/             # Train/val/test splits
│
├── defences/               # Defense implementations
│   ├── patch.py            # ZeroKnowledgePatch class
│   ├── classifier.py       # Poison classifier
│   ├── features.py         # Feature extraction
│   └── metrics_zk.py       # ASR, AUPRC metrics
│
├── retrievers/             # Retrieval models
│   ├── train.py            # Dense retriever training
│   ├── dense_retriever.py  # Dense retrieval
│   └── bm25_retriever.py   # BM25 baseline
│
├── scripts/                # Utility scripts
│   ├── eval_clean.py       # Clean evaluation
│   ├── eval_poisoned.py    # Poisoned evaluation
│   ├── eval_zkip.py        # ZKIP evaluation
│   └── generate_poison.py  # Poison generation
│
├── results/                # Outputs and plots
│   ├── *.csv               # Metric results
│   ├── *.png               # Visualizations
│   └── final/              # Final models
│
└── notebooks/              # Demo notebooks
    └── demo.ipynb
```

## Datasets

### Natural Questions (NQ)
- 87,925 query-passage pairs
- Open-domain QA from Google search

### BEIR (NFCorpus)
- 12,334 query-document pairs
- Medical/scientific retrieval benchmark

### Data Format (JSONL)
```json
{
  "query": "What is the capital of France?",
  "gold_doc": "Paris is the capital and largest city of France...",
  "candidates": ["doc1...", "doc2...", ...],
  "poison_doc": "The capital of France is London...",
  "poison_type": "fabrication"
}
```

### Downloading Datasets
```bash
python datasets/download_datasets.py
python datasets/make_splits.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{raguard2025,
  title={RAGuard: A Layered Defense Framework for Retrieval-Augmented Generation Systems Against Data Poisoning},
  author={[Tanish Kolhe, Pushkal Kumar, Tucker Nielson, Shubham Zala, Vincent Li, Michael Saxon, Sean Wu, Kevin Zhu]},
  booktitle={[NeurIPS]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

