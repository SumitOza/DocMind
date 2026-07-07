"""
Chroma-compatible embedding function backed by the Gemini embedding API.

Rationale: avoids downloading a local ONNX model at runtime (unreliable on
restricted/offline networks) and keeps embedding + generation on one provider.
"""

import os
import time
from typing import List
from dotenv import load_dotenv

load_dotenv()


class GeminiEmbeddingFunction:
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self._genai = genai

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            # Gemini's embedding endpoint has per-minute rate limits on the
            # free tier; a small delay avoids 429s during bulk chunk upload.
            for attempt in range(3):
                try:
                    result = self._genai.embed_content(
                        model="models/gemini-embedding-001",
                        content=text,
                        task_type="retrieval_document",
                    )
                    embeddings.append(result["embedding"])
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        return embeddings
