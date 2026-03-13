"""
Centralized configuration for the Document Q&A Chatbot.
Modify these settings to switch models, embedding providers, or chunk sizes.
"""
import os
from pathlib import Path

# ── Directories ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# ── LLM Settings ────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")       # Change to your fine-tuned model
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.9

# ── Embedding Settings ──────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking Settings ───────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Retriever Settings ──────────────────────────────────────
RETRIEVER_K = 4          # Number of chunks to retrieve

# ── Supported File Types ────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


def sanitize_collection_name(filename: str) -> str:
    """Convert a filename into a valid ChromaDB collection name.
    Rules: 3-63 chars, [a-zA-Z0-9._-], must start/end with alphanumeric.
    """
    import re
    name = filename.replace(".", "_")       # dots to underscores
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)  # kill invalid chars
    name = re.sub(r"_+", "_", name)         # collapse multiple underscores
    name = name.strip("_.-")               # must start/end alphanumeric
    name = name[:63]                        # max length 63
    if len(name) < 3:
        name = name + "_col"
    return name
