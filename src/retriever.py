"""Semantic search over stored embeddings (NumPy cosine similarity), with optional cross-encoder re-ranking."""
import json
import os

import numpy as np
from src.embedder import Embedder, Reranker

from src import config


class Retriever:
    def __init__(self):
        self.embedder = None
        self.vectors = None
        self.chunks = None
        self.reranker = None

    def reload(self):
        """Forget the loaded index (call after rebuilding it)."""
        self.vectors = None
        self.chunks = None

    def _load(self):
        if not os.path.exists(config.INDEX_FILE):
            raise FileNotFoundError("Index not found. Run: python -m src.ingest")
        if self.embedder is None:
            self.embedder = Embedder()
        self.vectors = np.load(config.INDEX_FILE)
        with open(config.CHUNKS_FILE, encoding="utf-8") as f:
            self.chunks = json.load(f)

    def search(self, question, k=config.TOP_K, rerank=True):
        """Return top-k chunks. Each has `score` (cosine) and, if reranked, `rerank_score`."""
        if self.vectors is None:
            self._load()

        fetch = max(config.RETRIEVE_K, k) if rerank else k
        fetch = min(fetch, len(self.vectors))
        q_vec = self.embedder.encode([question])
        all_scores = self.vectors @ q_vec[0].astype("float32")  # cosine similarity
        ids = np.argsort(-all_scores)[:fetch]

        results = []
        for i in ids:
            item = dict(self.chunks[int(i)])
            item["score"] = float(all_scores[i])
            results.append(item)

        if rerank and results:
            if self.reranker is None:
                self.reranker = Reranker()
            rr = self.reranker.predict([(question, r["text"]) for r in results])
            for r, s in zip(results, rr):
                r["rerank_score"] = float(s)
            results.sort(key=lambda r: r["rerank_score"], reverse=True)

        return results[:k]
