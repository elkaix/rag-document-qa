# Production backend image — RAG Q&A API
#
# WHY Python 3.12: ChromaDB ships pre-built wheels for 3.12.
#     Python 3.14 (host version) lacks ChromaDB wheel support in Docker images.
#
# Build:  docker build -t rag-api .
# Run:    docker run -p 8001:8001 -v app-data:/app/data rag-api

FROM python:3.12-slim AS base

WORKDIR /app

# System deps for document parsing (pypdf, python-docx, bs4)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ src/

# Ensure data directory exists
RUN mkdir -p data

EXPOSE 8001

# WHY --workers 1: ChromaDB PersistentClient is single-writer.
#     Multiple workers would corrupt the embedded database.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
