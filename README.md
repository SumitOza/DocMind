---
title: DocMind — Document AI Assistant
emoji: 📄
colorFrom: slate
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# DocMind — Document-Based AI Assistant

A full-stack RAG (Retrieval-Augmented Generation) application. Upload a PDF or text file, ask questions, and get answers grounded in your document with page-level citations.

Built for the **E2M Solutions AI Engineering Take-Home Assessment**.

---

## Features

- **Upload & Index** — PDF and `.txt` files are extracted, chunked (800 chars, 150-char overlap), and embedded into a persistent ChromaDB vector store
- **Grounded Q&A** — Every answer is retrieved from the actual document; the LLM is instructed not to guess
- **Page citations** — Click source chips to see the exact excerpt and page number used
- **Previous Documents tab** — Browse, load, and delete all previously uploaded documents (persists across restarts)
- **Multi-provider LLM** — Switch between Gemini, Groq, or Cerebras via a single env variable

---

## Tech Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI + uvicorn |
| Vector DB | ChromaDB (persistent, local) |
| Embeddings | Gemini `text-embedding-004` |
| LLM | Gemini `gemini-2.0-flash` (or Groq / Cerebras) |
| Frontend | Single-file HTML + Tailwind CDN + vanilla JS |
| PDF parsing | pypdf (page-aware extraction) |

---

## Setup

### Local

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # fill in your API key
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000` — the backend serves the frontend directly.

### HuggingFace Spaces

1. Create a new **Docker** Space on HuggingFace
2. Push this repo
3. Add your API key as a **Space Secret**:
   - `GEMINI_API_KEY` = your Gemini API key
   - `LLM_PROVIDER` = `gemini` (or `groq` / `cerebras`)
   - `GROQ_API_KEY` = (if using Groq)

The `Dockerfile` handles everything else.

---

## Architecture

```
Upload:  File → extract pages → slide-window chunk → embed → ChromaDB
Chat:    Query → embed → top-4 similarity search → LLM with context → answer + citations
Docs:    JSON registry (documents_registry.json) persists metadata across restarts
```

---

## Switching LLM Provider

Edit `.env` (local) or Space Secrets (HuggingFace):

```
LLM_PROVIDER=gemini   # or "groq" / "cerebras"
```
