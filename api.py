"""
FastAPI REST API for the Document Q&A Chatbot.
Provides endpoints for document upload, querying, and session management.

Run:  uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs  (Swagger UI)
"""
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from config import sanitize_collection_name
from ingest import ingest_file
from chain import build_chain

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="DocChat AI API",
    description="Upload documents and ask questions grounded in their content.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ──────────────────────────────────
sessions: Dict[str, dict] = {}


# ── Schemas ──────────────────────────────────────────────────
class QueryRequest(BaseModel):
    session_id: str
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    model: str
    ollama_url: str


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API and Ollama are running."""
    import urllib.request
    try:
        urllib.request.urlopen(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_status = "connected"
    except Exception:
        ollama_status = "unreachable"

    return HealthResponse(
        status="ok" if ollama_status == "connected" else "degraded",
        model=config.LLM_MODEL,
        ollama_url=config.OLLAMA_BASE_URL,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, DOCX, MD).
    Returns a session_id to use for subsequent queries.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {config.SUPPORTED_EXTENSIONS}",
        )

    save_path = config.UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        collection = sanitize_collection_name(file.filename)
        vectorstore = ingest_file(save_path, collection=collection)
        chunk_count = len(vectorstore.get().get("ids", []))

        chain = build_chain(collection=collection)
        session_id = uuid.uuid4().hex[:12]
        sessions[session_id] = {
            "chain": chain,
            "filename": file.filename,
            "collection": collection,
        }

        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            chunks=chunk_count,
            message="Document processed successfully.",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/query", response_model=QueryResponse)
def query_document(req: QueryRequest):
    """
    Ask a question about a previously uploaded document.
    Requires the session_id from /upload.
    """
    if req.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Upload a document first via /upload.",
        )

    chain = sessions[req.session_id]["chain"]

    try:
        result = chain.invoke({"question": req.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    sources = []
    for doc in result.get("source_documents", []):
        sources.append({
            "text": doc.page_content[:500],
            "page": doc.metadata.get("page", None),
            "source": doc.metadata.get("source", None),
        })

    return QueryResponse(answer=result["answer"], sources=sources)


@app.get("/sessions")
def list_sessions():
    """List all active sessions."""
    return {
        sid: {"filename": data["filename"], "collection": data["collection"]}
        for sid, data in sessions.items()
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a session and free memory."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del sessions[session_id]
    return {"message": f"Session {session_id} deleted."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)