"""Load documents -> split into chunks -> embed -> save vectors.

Run:  python -m src.ingest
      python -m src.ingest --chunk-size 300 --overlap 30
"""
import argparse
import glob
import json
import os

import numpy as np
from pypdf import PdfReader

from src import config
from src.embedder import Embedder


def load_documents(folder=config.DOCS_FOLDER):
    """Read PDF/TXT/MD files. Returns a list of {file, page, text}."""
    pages = []
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        name = os.path.basename(path)
        ext = name.lower().rsplit(".", 1)[-1]
        if ext == "pdf":
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = " ".join((page.extract_text() or "").split())
                if text:
                    pages.append({"file": name, "page": i + 1, "text": text})
        elif ext in ("txt", "md"):
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = " ".join(f.read().split())
            if text:
                pages.append({"file": name, "page": 1, "text": text})
    return pages


def chunk_pages(pages, size=config.CHUNK_SIZE, overlap=config.OVERLAP):
    """Split each page into overlapping chunks, keeping file + page for citations."""
    chunks = []
    step = size - overlap
    for p in pages:
        text = p["text"]
        for start in range(0, len(text), step):
            piece = text[start:start + size]
            if len(piece.strip()) < 50:  # skip tiny leftovers
                continue
            chunks.append({"file": p["file"], "page": p["page"], "text": piece})
    return chunks


def build_index(folder=config.DOCS_FOLDER, size=config.CHUNK_SIZE, overlap=config.OVERLAP):
    pages = load_documents(folder)
    if not pages:
        raise RuntimeError(f"No documents found in '{folder}/'. Add PDF/TXT/MD files first.")
    chunks = chunk_pages(pages, size, overlap)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks")

    # Normalized vectors: dot product == cosine similarity
    vectors = Embedder().encode([c["text"] for c in chunks], batch_size=32, verbose=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    np.save(config.INDEX_FILE, vectors.astype("float32"))
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"Saved index to {config.DATA_DIR}/")
    return len(chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.OVERLAP)
    args = ap.parse_args()
    build_index(size=args.chunk_size, overlap=args.overlap)
