"""
Document ingestion pipeline.
Loads documents, splits into chunks, embeds, and stores in ChromaDB.
"""
import hashlib
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import config


# ── Loader registry ─────────────────────────────────────────
LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".md":   TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
}


def _file_hash(path: Path) -> str:
    """Return SHA-256 hex digest of a file (used as collection name)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_document(file_path: str | Path) -> List[Document]:
    """Load a single document and return LangChain Document objects."""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if ext not in LOADER_MAP:
        raise ValueError(f"Unsupported file type: {ext}")

    loader = LOADER_MAP[ext](str(file_path))
    docs = loader.load()

    # Attach metadata
    for doc in docs:
        doc.metadata["source"] = file_path.name

    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)

    # Clean up chunks — remove excessive whitespace
    for chunk in chunks:
        chunk.page_content = " ".join(chunk.page_content.split())

    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the configured HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def build_vectorstore(chunks: List[Document], collection: str = "default") -> Chroma:
    """Create or update a ChromaDB collection from document chunks."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(config.CHROMA_DIR),
        collection_name=collection,
    )
    return vectorstore


def load_vectorstore(collection: str = "default") -> Chroma:
    """Load an existing ChromaDB collection."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=str(config.CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=collection,
    )


def ingest_file(file_path: str | Path, collection: str = "default") -> Chroma:
    """
    Full ingestion pipeline: load → split → embed → store.
    Returns the vector store instance.
    """
    docs = load_document(file_path)
    chunks = split_documents(docs)
    print(f"[ingest] {Path(file_path).name}: {len(docs)} pages → {len(chunks)} chunks")
    return build_vectorstore(chunks, collection=collection)