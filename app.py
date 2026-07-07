"""
HuggingFace Spaces entry point for DocMind.

HF Spaces runs this file. It starts the FastAPI app via uvicorn on port 7860.
Set your API keys as HF Space Secrets (not in code):
  - GEMINI_API_KEY
  - LLM_PROVIDER (optional, defaults to "gemini")
"""

import os
import sys

# Add backend to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# Change working directory to backend so relative paths (uploads/, chroma_store/) work
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))

import uvicorn
from backend.main import app  # noqa: F401 — imported so uvicorn can find it

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
