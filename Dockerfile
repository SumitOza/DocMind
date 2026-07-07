# DocMind — HuggingFace Spaces Dockerfile
# HF Spaces expects the app to listen on port 7860

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for chromadb / pypdf
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create persistent directories (HF Spaces mounts /data for persistence)
# If running on HF with persistent storage, symlink to /data
RUN mkdir -p ./backend/uploads ./backend/chroma_store

# Expose port 7860 (HuggingFace default)
EXPOSE 7860

# Fix static files path — main.py serves frontend from ../frontend relative to backend/
# We run from /app so paths resolve correctly
ENV PYTHONPATH=/app
ENV PORT=7860

# Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "backend"]
