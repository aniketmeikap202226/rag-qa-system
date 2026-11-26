"""Embedding model and cross-encoder re-ranker, using Hugging Face Transformers + PyTorch directly.

Same models as sentence-transformers (all-MiniLM-L6-v2 and ms-marco-MiniLM-L-6-v2),
implemented with mean pooling (embeddings) and a classification head (re-ranker).
"""
import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src import config


class Embedder:
    """Turns text into normalized 384-dim vectors (mean pooling over tokens)."""

    def __init__(self, name=config.EMBED_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=32, verbose=False):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=256, return_tensors="pt")
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # cosine-ready
            out.append(emb.cpu().numpy())
            if verbose:
                done = min(i + batch_size, len(texts))
                print(f"Embedded {done}/{len(texts)}", end="\r")
        if verbose:
            print()
        if not out:
            return np.zeros((0, 384), dtype="float32")
        return np.vstack(out).astype("float32")


class Reranker:
    """Cross-encoder: reads (question, chunk) together and scores relevance."""

    def __init__(self, name=config.RERANK_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()

    @torch.no_grad()
    def predict(self, pairs, batch_size=16):
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            enc = self.tokenizer([q for q, _ in batch], [p for _, p in batch],
                                 padding=True, truncation=True, max_length=512,
                                 return_tensors="pt")
            logits = self.model(**enc).logits
            scores.extend(logits[:, 0].tolist())
        return scores
