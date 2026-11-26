"""All settings in one place. Change values here (or via environment variables)."""
import os

# Paths
DOCS_FOLDER = "docs"
DATA_DIR = "data"
INDEX_FILE = os.path.join(DATA_DIR, "embeddings.npy")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")

# Chunking
CHUNK_SIZE = 800   # characters per chunk
OVERLAP = 50       # characters shared between neighbouring chunks

# Models
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval
TOP_K = 5          # chunks sent to the LLM
RETRIEVE_K = 20    # candidates fetched before re-ranking
MIN_SCORE = 0.25   # if best cosine similarity is below this -> "I don't know"

# LLM: "ollama" (free, local), "openai" or "anthropic"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
