"""
DocMind — Document-Based AI Assistant
E2M Solutions AI Engineering Take-Home Assessment (Enhanced)

Architecture:
    Upload -> Extract text -> Chunk -> Embed -> Store in Chroma (vector DB)
             + persist metadata to JSON registry
    Chat   -> Embed query -> Retrieve top-k chunks -> Call LLM -> Return answer + citations
    Docs   -> List all previously uploaded documents from persistent JSON registry
"""

import os
import uuid
import json
import shutil
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

from llm_provider import call_llm
from embeddings import GeminiEmbeddingFunction

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="DocMind — Document-Based AI Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_store"
REGISTRY_FILE = BASE_DIR / "documents_registry.json"

# Support both local layout (backend/../frontend) and Docker layout (/app/frontend)
_local_frontend = BASE_DIR.parent / "frontend"
_docker_frontend = Path("/app/frontend")
FRONTEND_DIR = _local_frontend if _local_frontend.exists() else _docker_frontend

UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Persistent document registry (survives restarts)
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        try:
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_registry(registry: dict):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


# In-memory cache, initialized from disk
DOCUMENTS: dict = load_registry()

# ---------------------------------------------------------------------------
# Vector DB setup
# ---------------------------------------------------------------------------


embedding_fn = GeminiEmbeddingFunction()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = "docmind"
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,          # text-embedding-004 outputs 768-dim vectors
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text_with_pages(file_path: str, filename: str) -> List[dict]:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i, "text": text})
        return pages
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [{"page": 1, "text": text}]


def chunk_text(page_text: str, page_num: int, doc_id: str, chunk_size: int = 800, overlap: int = 150):
    chunks = []
    start = 0
    text_len = len(page_text)
    while start < text_len:
        end = start + chunk_size
        chunk_str = page_text[start:end]
        if chunk_str.strip():
            chunks.append({
                "id": f"{doc_id}_p{page_num}_c{start}",
                "text": chunk_str,
                "page": page_num,
            })
        start += chunk_size - overlap
    return chunks


def format_file_size(path: str) -> str:
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    else:
        return f"{size / (1024 * 1024):.1f} MB"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    doc_id: str
    query: str


class DeleteRequest(BaseModel):
    doc_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accept a PDF or .txt file, extract, chunk, embed, and store in Chroma."""
    allowed = (".pdf", ".txt")
    if not file.filename.lower().endswith(allowed):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    doc_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{doc_id}_{file.filename}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pages = extract_text_with_pages(str(save_path), file.filename)

    if not pages:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not extract any text from this file.")

    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_text(page["text"], page["page"], doc_id))

    if not all_chunks:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Document produced no usable chunks.")

    vectors = []
    texts = [c["text"] for c in all_chunks]
    embeddings = embedding_fn(texts)          # batch embed all chunks
    for c, emb in zip(all_chunks, embeddings):
        vectors.append({
            "id": c["id"],
            "values": emb,
            "metadata": {"page": c["page"], "doc_id": doc_id, "filename": file.filename, "text": c["text"]}
        })
    index.upsert(vectors=vectors)

    file_type = "pdf" if file.filename.lower().endswith(".pdf") else "txt"
    doc_meta = {
        "filename": file.filename,
        "num_pages": len(pages),
        "num_chunks": len(all_chunks),
        "uploaded_at": datetime.utcnow().isoformat(),
        "file_size": format_file_size(str(save_path)),
        "file_type": file_type,
    }

    DOCUMENTS[doc_id] = doc_meta
    save_registry(DOCUMENTS)

    return {"doc_id": doc_id, **doc_meta}


@app.get("/api/documents")
async def list_documents():
    """Return all previously uploaded documents (persisted across restarts)."""
    result = []
    for doc_id, meta in DOCUMENTS.items():
        result.append({"doc_id": doc_id, **meta})
    # Sort by upload time descending (newest first)
    result.sort(key=lambda x: x.get("uploaded_at", ""), reverse=True)
    return result


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the registry and vector store."""
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found.")

    filename = DOCUMENTS[doc_id]["filename"]

    # Remove from Chroma
    try:
        # Replace with:
        # Pinecone doesn't support filter-based delete on serverless without fetching IDs first
        # Easiest approach: store chunk IDs in the registry at upload time
        # Or use a namespace per doc_id instead:
        index.delete(delete_all=True, namespace=doc_id)
    except Exception:
        pass

    # Remove uploaded file
    save_path = UPLOAD_DIR / f"{doc_id}_{filename}"
    if save_path.exists():
        save_path.unlink()

    del DOCUMENTS[doc_id]
    save_registry(DOCUMENTS)

    return {"deleted": doc_id}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Retrieve relevant chunks, call LLM with context, return answer + citations."""
    if req.doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Document not found. Upload a document first.")

    query_embedding = embedding_fn([req.query])[0]
    results = index.query(
        vector=query_embedding,
        top_k=4,
        filter={"doc_id": req.doc_id},
        include_metadata=True
    )
    retrieved_docs = [m["metadata"]["text"] for m in results["matches"]]
    retrieved_meta = [m["metadata"] for m in results["matches"]]

    retrieved_docs = results["documents"][0] if results["documents"] else []
    retrieved_meta = results["metadatas"][0] if results["metadatas"] else []

    if not retrieved_docs:
        return {
            "answer": "I couldn't find relevant content in the document for that question.",
            "sources": [],
        }

    context_blocks = []
    sources = []
    for i, (text, meta) in enumerate(zip(retrieved_docs, retrieved_meta)):
        context_blocks.append(f"[Source {i+1} — Page {meta['page']}]\n{text}")
        sources.append({
            "index": i + 1,
            "page": meta["page"],
            "snippet": text[:220] + ("..." if len(text) > 220 else ""),
        })

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant that answers questions strictly using the provided document context. "
        "Cite the source number (e.g., 'according to Source 2') when using information from it. "
        "If the answer isn't in the context, say so honestly rather than guessing."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {req.query}\n\nAnswer using only the context above."

    answer = call_llm(system_prompt, user_prompt)

    return {"answer": answer, "sources": sources}


@app.get("/api/health")
async def health():
    return {"status": "ok", "documents": len(DOCUMENTS)}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

static_dir = FRONTEND_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# Catch-all for any other routes (SPA fallback)
@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    raise HTTPException(status_code=404, detail="Not found")
