# RAG Document Q&A System — Project Plan

**Timeline:** 12 days  
**Difficulty:** Intermediate–Advanced  
**Goal:** Build a production-grade Retrieval-Augmented Generation system capable of answering questions over uploaded documents with citations, hybrid retrieval, and multi-LLM support.

---

## Table of Contents

1. [Phase 1: Architecture Design](#phase-1-architecture-design-day-1)
2. [Phase 2: Document Processing Pipeline](#phase-2-document-processing-pipeline-days-2-3)
3. [Phase 3: Vector Database Setup](#phase-3-vector-database-setup-days-3-4)
4. [Phase 4: Retrieval Strategy](#phase-4-retrieval-strategy-days-4-5)
5. [Phase 5: LLM Integration](#phase-5-llm-integration-days-5-6)
6. [Phase 6: FastAPI Backend](#phase-6-fastapi-backend-days-6-8)
7. [Phase 7: Frontend Dashboard](#phase-7-frontend-dashboard-days-8-10)
8. [Phase 8: Evaluation](#phase-8-evaluation-days-10-11)
9. [Phase 9: Deployment](#phase-9-deployment-days-11-12)
10. [Dependencies & Setup](#dependencies--setup)

---

## Phase 1: Architecture Design (Day 1)

### System Overview

The RAG system follows a classic retrieve-then-read architecture with enhancements for hybrid retrieval and multi-LLM support.

```
INDEXING PIPELINE
─────────────────
Document File
    │
    ▼
Document Loader (format detection)
    │
    ▼
Text Splitter (chunking strategy)
    │
    ▼
Metadata Extractor
    │
    ▼
Embedding Model (sentence-transformers / OpenAI)
    │
    ▼
Vector DB (Qdrant — HNSW index)

QUERY PIPELINE
──────────────
User Query
    │
    ├──► Dense Retriever (Qdrant cosine similarity)
    │         │
    ├──► BM25 Sparse Retriever
    │         │
    └──► Hybrid Fusion (RRF)
              │
              ▼
         Cross-Encoder Re-Ranker
              │
              ▼
         Top-K Contexts
              │
              ▼
         Prompt Builder
              │
              ▼
         LLM (GPT-4 / Claude / Llama 3)
              │
              ▼
         Response + Citations
```

### Directory Structure

```
src/
├── document_loader.py        # Format-agnostic loading + chunking
├── embeddings.py             # Embedding generation (HF + OpenAI)
├── vector_store.py           # Qdrant client wrapper
├── retriever.py              # Hybrid retrieval + re-ranking
├── llm_handler.py            # LLM abstraction + streaming
├── evaluation.py             # RAGAS-based evaluation
├── config.py                 # Settings via pydantic-settings
├── api/
│   ├── main.py               # FastAPI app + middleware
│   ├── schemas.py            # Pydantic request/response models
│   └── routes/
│       ├── upload.py         # /api/upload
│       ├── query.py          # /api/query
│       └── documents.py      # /api/documents
└── dashboard/
    ├── app.py                # Streamlit entry point
    └── pages/
        ├── chat.py           # Chat interface
        ├── documents.py      # Document library
        └── settings.py       # Settings panel
```

### Configuration (src/config.py)

```python
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    default_llm: Literal["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "llama3"] = "gpt-4"
    temperature: float = 0.7

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    use_openai_embeddings: bool = False

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection_name: str = "rag_documents"

    # Retrieval
    top_k: int = 5
    retrieval_type: Literal["dense", "sparse", "hybrid"] = "hybrid"
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Chunking
    chunking_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## Phase 2: Document Processing Pipeline (Days 2–3)

### Supported Formats

| Format | Library | Notes |
|--------|---------|-------|
| PDF | pypdf / pdfplumber | Handles multi-column, tables |
| DOCX | python-docx | Preserves headings |
| TXT | Built-in | Direct read |
| MD | markdown-it-py | Strips markdown syntax |
| HTML | BeautifulSoup4 | Removes scripts/styles |
| CSV | pandas | Converts rows to text |

### File: src/document_loader.py

```python
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import hashlib

class DocumentLoader:
    """Format-agnostic document loader with chunking support."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".csv"}

    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        Auto-detect format and load document content.

        Args:
            file_path: Path to the document file.

        Returns:
            Dict with keys: text (str), metadata (dict), chunks (list)

        Raises:
            ValueError: If file format is unsupported.
            FileNotFoundError: If file doesn't exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {ext}. Supported: {self.SUPPORTED_EXTENSIONS}")

        loaders = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".txt": self._load_txt,
            ".md": self._load_markdown,
            ".html": self._load_html,
            ".htm": self._load_html,
            ".csv": self._load_csv,
        }

        text = loaders[ext](file_path)
        metadata = self.extract_metadata(file_path)
        return {"text": text, "metadata": metadata}

    def _load_pdf(self, file_path: str) -> str:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            pages.append(f"[Page {i+1}]\n{page_text}")
        return "\n\n".join(pages)

    def _load_docx(self, file_path: str) -> str:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    def _load_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _load_markdown(self, file_path: str) -> str:
        import re
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Strip markdown syntax for plain text
        text = re.sub(r"#{1,6}\s+", "", text)  # headings
        text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)  # bold/italic
        text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)  # code
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # images
        text = re.sub(r"\[(.+?)\]\(.*?\)", r"\1", text)  # links
        return text.strip()

    def _load_html(self, file_path: str) -> str:
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    def _load_csv(self, file_path: str) -> str:
        import pandas as pd
        df = pd.read_csv(file_path)
        # Convert each row to a readable sentence
        rows = []
        for _, row in df.iterrows():
            row_str = " | ".join(f"{col}: {val}" for col, val in row.items())
            rows.append(row_str)
        return "\n".join(rows)

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
        strategy: str = "recursive",
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Full document text.
            chunk_size: Target chunk size in characters (approx 1 token ~ 4 chars).
            overlap: Number of overlapping characters between chunks.
            strategy: One of "fixed", "recursive", "semantic".

        Returns:
            List of dicts: {text, chunk_index, start_char, end_char}
        """
        if strategy == "fixed":
            return self._chunk_fixed(text, chunk_size, overlap)
        elif strategy == "recursive":
            return self._chunk_recursive(text, chunk_size, overlap)
        elif strategy == "semantic":
            return self._chunk_semantic(text, chunk_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _chunk_fixed(self, text: str, size: int, overlap: int) -> List[Dict]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append({
                "text": text[start:end],
                "chunk_index": idx,
                "start_char": start,
                "end_char": end,
            })
            start += size - overlap
            idx += 1
        return chunks

    def _chunk_recursive(self, text: str, size: int, overlap: int) -> List[Dict]:
        """Split on paragraph > sentence > word boundaries."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        raw_chunks = splitter.split_text(text)
        result = []
        cursor = 0
        for i, chunk in enumerate(raw_chunks):
            start = text.find(chunk, cursor)
            if start == -1:
                start = cursor
            end = start + len(chunk)
            result.append({
                "text": chunk,
                "chunk_index": i,
                "start_char": start,
                "end_char": end,
            })
            cursor = max(cursor, start + len(chunk) - overlap)
        return result

    def _chunk_semantic(self, text: str, max_size: int) -> List[Dict]:
        """Group semantically similar sentences using embeddings."""
        import nltk
        from sentence_transformers import SentenceTransformer
        import numpy as np

        nltk.download("punkt", quiet=True)
        sentences = nltk.sent_tokenize(text)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)

        chunks, current_chunk, current_len = [], [], 0
        chunk_idx = 0
        for i, (sentence, emb) in enumerate(zip(sentences, embeddings)):
            current_chunk.append(sentence)
            current_len += len(sentence)
            if current_len >= max_size and i < len(sentences) - 1:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": chunk_idx,
                    "start_char": text.find(current_chunk[0]),
                    "end_char": text.find(current_chunk[-1]) + len(current_chunk[-1]),
                })
                current_chunk, current_len = [], 0
                chunk_idx += 1
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_idx,
                "start_char": text.find(current_chunk[0]),
                "end_char": len(text),
            })
        return chunks

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract file metadata including hash, size, timestamps.

        Returns:
            Dict with: source, filename, extension, size_bytes, file_hash,
                       created_at, modified_at, document_type
        """
        path = Path(file_path)
        stat = path.stat()

        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        doc_type_map = {
            ".pdf": "pdf", ".docx": "word", ".txt": "text",
            ".md": "markdown", ".html": "html", ".htm": "html", ".csv": "csv",
        }

        return {
            "source": str(path.resolve()),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "file_hash": file_hash,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "document_type": doc_type_map.get(path.suffix.lower(), "unknown"),
            "indexed_at": datetime.utcnow().isoformat(),
        }
```

### File: src/embeddings.py

```python
from typing import List, Union
import numpy as np

class EmbeddingModel:
    """Unified embedding interface for HuggingFace and OpenAI models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_openai: bool = False):
        self.model_name = model_name
        self.use_openai = use_openai
        self._model = None  # Lazy load

    def _load_model(self):
        if self.use_openai:
            from openai import OpenAI
            self._client = OpenAI()
        else:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def get_embeddings(self, texts: List[str], model: str = None) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of strings to embed.
            model: Override model name (optional).

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if self._model is None and not self.use_openai:
            self._load_model()

        if self.use_openai:
            return self._openai_embed(texts, model or "text-embedding-3-small")
        else:
            return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _openai_embed(self, texts: List[str], model: str) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI()
        response = client.embeddings.create(input=texts, model=model)
        return np.array([item.embedding for item in response.data])

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Process texts in batches to avoid memory issues.

        Args:
            texts: All texts to embed.
            batch_size: Number of texts per batch.

        Returns:
            Concatenated numpy array of all embeddings.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            embeddings = self.get_embeddings(batch)
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.get_embeddings([query])[0]

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        dims = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return dims.get(self.model_name, 384)
```

---

## Phase 3: Vector Database Setup (Days 3–4)

### Qdrant Setup via Docker

```yaml
# docker-compose.yml (Qdrant service)
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 5
volumes:
  qdrant_data:
```

### Collection Schema

Each Qdrant point has:
- **vector**: 384-dim float array (embedding)
- **payload**: JSON object with:
  - `text`: chunk text content
  - `chunk_index`: position in document
  - `source`: file path
  - `filename`: document name
  - `document_id`: UUID
  - `document_type`: pdf/docx/etc
  - `page_number`: for PDFs
  - `indexed_at`: ISO timestamp

### File: src/vector_store.py

```python
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    HnswConfigDiff, OptimizersConfigDiff,
)
import uuid
import numpy as np

class VectorStore:
    """Qdrant vector database operations."""

    def __init__(self, url: str = "http://localhost:6333", api_key: str = ""):
        self.client = QdrantClient(url=url, api_key=api_key or None)

    def create_collection(
        self,
        collection_name: str,
        vector_dim: int = 384,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """
        Create a Qdrant collection with HNSW indexing.

        Args:
            collection_name: Name for the collection.
            vector_dim: Dimensionality of the embedding vectors.
            distance: Distance metric (COSINE, DOT, EUCLID).

        Returns:
            True if created, False if already exists.
        """
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name in existing:
            return False

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=distance),
            hnsw_config=HnswConfigDiff(
                m=16,             # Number of edges per node
                ef_construct=100, # Build-time exploration factor
                full_scan_threshold=10000,
            ),
            optimizers_config=OptimizersConfigDiff(
                default_segment_number=4,
                memmap_threshold=20000,
            ),
        )
        return True

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
        metadata: Dict[str, Any],
        collection_name: str = "rag_documents",
    ) -> List[str]:
        """
        Index document chunks with their embeddings.

        Args:
            documents: List of chunks: [{text, chunk_index, ...}]
            embeddings: numpy array (n_chunks, dim)
            metadata: Document-level metadata (filename, source, etc.)
            collection_name: Target Qdrant collection.

        Returns:
            List of point IDs (UUIDs).
        """
        points = []
        point_ids = []
        for doc, vector in zip(documents, embeddings):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            payload = {
                **metadata,
                "text": doc["text"],
                "chunk_index": doc.get("chunk_index", 0),
                "start_char": doc.get("start_char", 0),
                "end_char": doc.get("end_char", 0),
            }
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            ))

        # Batch upsert in chunks of 100
        for i in range(0, len(points), 100):
            self.client.upsert(
                collection_name=collection_name,
                points=points[i:i+100],
                wait=True,
            )
        return point_ids

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        collection_name: str = "rag_documents",
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector (1D numpy array).
            top_k: Number of results to return.
            filters: Optional payload filters (e.g., {"filename": "report.pdf"}).
            collection_name: Target collection.

        Returns:
            List of results: [{text, score, metadata}]
        """
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            {
                "text": r.payload.get("text", ""),
                "score": r.score,
                "id": r.id,
                "metadata": {k: v for k, v in r.payload.items() if k != "text"},
            }
            for r in results
        ]

    def delete_collection(self, collection_name: str) -> bool:
        """Delete an entire collection."""
        self.client.delete_collection(collection_name)
        return True

    def delete_documents(self, document_ids: List[str], collection_name: str = "rag_documents"):
        """Delete specific documents by their point IDs."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=document_ids,
            wait=True,
        )

    def get_collection_stats(self, collection_name: str = "rag_documents") -> Dict[str, Any]:
        """Return collection statistics."""
        info = self.client.get_collection(collection_name)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "status": info.status,
        }

    def list_documents(self, collection_name: str = "rag_documents") -> List[Dict]:
        """List all unique documents (grouped by filename)."""
        results, offset = [], None
        seen = set()
        while True:
            response, next_offset = self.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in response:
                fname = point.payload.get("filename", "")
                if fname not in seen:
                    seen.add(fname)
                    results.append({
                        "filename": fname,
                        "source": point.payload.get("source", ""),
                        "indexed_at": point.payload.get("indexed_at", ""),
                        "document_type": point.payload.get("document_type", ""),
                    })
            if next_offset is None:
                break
            offset = next_offset
        return results
```

---

## Phase 4: Retrieval Strategy (Days 4–5)

### Hybrid Retrieval Architecture

Hybrid retrieval combines dense (embedding-based) and sparse (keyword-based BM25) retrieval using Reciprocal Rank Fusion (RRF):

```
score_rrf = Σ 1 / (k + rank_i)   where k=60 (smoothing constant)
```

### Cross-Encoder Re-Ranking

After initial retrieval, a cross-encoder model scores each (query, chunk) pair directly:
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast) or `cross-encoder/ms-marco-electra-base` (accurate)
- Re-ranking improves precision at the cost of latency (~50ms per chunk)

### File: src/retriever.py

```python
from typing import List, Dict, Any, Literal
import numpy as np
from rank_bm25 import BM25Okapi

class HybridRetriever:
    """Combines dense and sparse retrieval with cross-encoder re-ranking."""

    def __init__(self, vector_store, embedding_model, reranker_model: str = None):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.reranker = self._load_reranker(reranker_model)
        self._bm25_index = None
        self._bm25_docs = []

    def _load_reranker(self, model_name: str = None):
        if model_name is None:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)

    def build_bm25_index(self, documents: List[str]):
        """Build BM25 index from raw text chunks."""
        tokenized = [doc.lower().split() for doc in documents]
        self._bm25_index = BM25Okapi(tokenized)
        self._bm25_docs = documents

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        retrieval_type: Literal["dense", "sparse", "hybrid"] = "hybrid",
        filters: Dict = None,
        collection_name: str = "rag_documents",
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval function.

        Args:
            query: User's question.
            top_k: Number of final results.
            retrieval_type: "dense", "sparse", or "hybrid".
            filters: Optional metadata filters.
            collection_name: Qdrant collection.

        Returns:
            Ranked list of document chunks with scores.
        """
        fetch_k = top_k * 3  # Over-retrieve for re-ranking

        if retrieval_type == "dense":
            candidates = self._dense_retrieve(query, fetch_k, filters, collection_name)
        elif retrieval_type == "sparse":
            candidates = self._sparse_retrieve(query, fetch_k)
        else:  # hybrid
            dense_results = self._dense_retrieve(query, fetch_k, filters, collection_name)
            sparse_results = self._sparse_retrieve(query, fetch_k)
            candidates = self._reciprocal_rank_fusion([dense_results, sparse_results], top_k=fetch_k)

        # Re-rank with cross-encoder
        reranked = self.rerank(query, candidates, top_k=top_k)
        return reranked

    def _dense_retrieve(self, query: str, top_k: int, filters, collection_name: str) -> List[Dict]:
        query_emb = self.embedding_model.get_query_embedding(query)
        return self.vector_store.search_similar(query_emb, top_k=top_k, filters=filters,
                                                collection_name=collection_name)

    def _sparse_retrieve(self, query: str, top_k: int) -> List[Dict]:
        if self._bm25_index is None:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"text": self._bm25_docs[i], "score": float(scores[i]), "metadata": {}}
            for i in top_indices
            if scores[i] > 0
        ]

    def _reciprocal_rank_fusion(self, result_lists: List[List[Dict]], top_k: int, k: int = 60) -> List[Dict]:
        """Merge ranked lists using RRF."""
        scores = {}
        docs = {}
        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                text = doc["text"]
                if text not in scores:
                    scores[text] = 0.0
                    docs[text] = doc
                scores[text] += 1.0 / (k + rank + 1)

        sorted_texts = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        result = []
        for text in sorted_texts[:top_k]:
            doc = docs[text].copy()
            doc["score"] = scores[text]
            result.append(doc)
        return result

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-rank candidates using a cross-encoder model.

        Args:
            query: Original user query.
            documents: Candidate documents from initial retrieval.
            top_k: Final number of results.

        Returns:
            Top-k documents sorted by cross-encoder score.
        """
        if not documents:
            return []

        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.reranker.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        return reranked[:top_k]

    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Format retrieved contexts into a RAG prompt.

        Args:
            query: User question.
            contexts: Retrieved and re-ranked chunks.

        Returns:
            Formatted prompt string for LLM.
        """
        context_text = ""
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get("metadata", {}).get("filename", "Unknown")
            context_text += f"\n[Source {i}: {source}]\n{ctx['text']}\n"

        return f"""You are a helpful assistant that answers questions based on the provided document context.
Use ONLY the information in the context below to answer. If the answer is not in the context, say so.
Always cite your sources using [Source N] notation.

Context:
{context_text}

Question: {query}

Answer:"""
```

---

## Phase 5: LLM Integration (Days 5–6)

### Supported LLMs

| Model | Provider | Notes |
|-------|----------|-------|
| gpt-4 / gpt-4-turbo | OpenAI API | Best quality |
| gpt-3.5-turbo | OpenAI API | Faster, cheaper |
| claude-3-opus | Anthropic API | Strong reasoning |
| llama3:8b | Ollama (local) | Privacy, no cost |
| llama3:70b | Ollama (local) | High quality local |

### File: src/llm_handler.py

```python
from typing import Iterator, List, Dict, Any, Optional
import json

class LLMHandler:
    """Unified LLM interface supporting streaming and multiple providers."""

    SYSTEM_PROMPT = """You are a precise document Q&A assistant. 
Rules:
1. Answer ONLY from the provided context.
2. Cite sources using [Source N] notation inline.
3. If the answer isn't in the context, say "I couldn't find this information in the provided documents."
4. Be concise and factual. Do not speculate.
5. Format lists and tables in Markdown when appropriate."""

    def generate_response(
        self,
        prompt: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Generate LLM response.

        Args:
            prompt: Full prompt including context and question.
            model: Model identifier.
            temperature: Sampling temperature (0=deterministic, 1=creative).
            max_tokens: Maximum response length.
            stream: If True, return a token iterator.

        Returns:
            Complete response string, or Iterator[str] if streaming.
        """
        if model.startswith("gpt"):
            return self._openai_generate(prompt, model, temperature, max_tokens, stream)
        elif model.startswith("claude"):
            return self._anthropic_generate(prompt, model, temperature, max_tokens, stream)
        elif model in ("llama3", "llama3:8b", "llama3:70b", "mistral"):
            return self._ollama_generate(prompt, model, temperature, stream)
        else:
            raise ValueError(f"Unknown model: {model}")

    def generate_with_context(
        self,
        query: str,
        contexts: List[Dict],
        model: str = "gpt-4",
        system_prompt: str = None,
        temperature: float = 0.3,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """
        Generate response with retrieved context (full RAG pipeline).

        Args:
            query: User question.
            contexts: Retrieved document chunks.
            model: LLM model name.
            system_prompt: Override default system prompt.
            temperature: Lower = more faithful to context.
            stream: Enable streaming output.
        """
        system = system_prompt or self.SYSTEM_PROMPT
        context_block = self._build_context_block(contexts)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
        ]

        if model.startswith("gpt"):
            return self._openai_chat(messages, model, temperature, stream=stream)
        elif model.startswith("claude"):
            return self._anthropic_chat(messages, model, temperature, stream=stream)
        else:
            prompt = f"{system}\n\nContext:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
            return self._ollama_generate(prompt, model, temperature, stream)

    def _openai_chat(self, messages, model, temperature, max_tokens=1024, stream=False):
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            def generator():
                for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            return generator()
        return response.choices[0].message.content

    def _anthropic_chat(self, messages, model, temperature, max_tokens=1024, stream=False):
        import anthropic
        client = anthropic.Anthropic()
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_messages = [m for m in messages if m["role"] != "system"]

        if stream:
            def generator():
                with client.messages.stream(
                    model=model, max_tokens=max_tokens,
                    system=system, messages=user_messages,
                ) as s:
                    for text in s.text_stream:
                        yield text
            return generator()

        response = client.messages.create(
            model=model, max_tokens=max_tokens,
            system=system, messages=user_messages,
        )
        return response.content[0].text

    def _ollama_generate(self, prompt: str, model: str, temperature: float, stream: bool):
        import requests
        payload = {"model": model, "prompt": prompt, "stream": stream,
                   "options": {"temperature": temperature}}
        if stream:
            def generator():
                with requests.post("http://localhost:11434/api/generate",
                                   json=payload, stream=True) as r:
                    for line in r.iter_lines():
                        if line:
                            data = json.loads(line)
                            if not data.get("done"):
                                yield data.get("response", "")
            return generator()
        r = requests.post("http://localhost:11434/api/generate", json={**payload, "stream": False})
        return r.json()["response"]

    def _build_context_block(self, contexts: List[Dict]) -> str:
        parts = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx.get("metadata", {}).get("filename", "Document")
            parts.append(f"[Source {i}: {source}]\n{ctx['text']}")
        return "\n\n".join(parts)

    def format_sources(self, contexts: List[Dict]) -> List[Dict]:
        """Format source citations for display."""
        return [
            {
                "source_number": i + 1,
                "filename": ctx.get("metadata", {}).get("filename", "Unknown"),
                "page": ctx.get("metadata", {}).get("page_number"),
                "score": round(ctx.get("rerank_score", ctx.get("score", 0)), 4),
                "excerpt": ctx["text"][:200] + "..." if len(ctx["text"]) > 200 else ctx["text"],
            }
            for i, ctx in enumerate(contexts)
        ]
```

---

## Phase 6: FastAPI Backend (Days 6–8)

### API Endpoints

| Method | Path | Description | Auth |
|--------|------|-------------|------|
| POST | /api/upload | Upload & index document | Optional |
| POST | /api/query | Ask a question | Optional |
| GET | /api/documents | List indexed docs | Optional |
| DELETE | /api/documents/{id} | Remove document | Optional |
| GET | /api/health | Health check | None |
| WS | /api/chat | Streaming chat | Optional |

### Pydantic Schemas (src/api/schemas.py)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    model: Literal["gpt-4", "gpt-3.5-turbo", "claude-3-opus", "llama3"] = "gpt-4"
    retrieval_type: Literal["dense", "sparse", "hybrid"] = "hybrid"
    collection_name: str = "rag_documents"
    filters: Optional[dict] = None
    stream: bool = False

class SourceCitation(BaseModel):
    source_number: int
    filename: str
    page: Optional[int]
    score: float
    excerpt: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    model_used: str
    retrieval_type: str
    latency_ms: int
    query_id: str

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int
    collection_name: str
    indexed_at: datetime
    status: Literal["success", "error"]
    message: str = ""

class DocumentInfo(BaseModel):
    filename: str
    source: str
    indexed_at: str
    document_type: str
    chunks_count: Optional[int]
```

### File: src/api/main.py

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .routes import upload, query, documents

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RAG Document Q&A API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "https://your-domain.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(documents.router, prefix="/api")

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
```

### WebSocket Streaming (src/api/routes/query.py)

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json, uuid, time

router = APIRouter()

@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query_id = str(uuid.uuid4())
            await websocket.send_json({"type": "start", "query_id": query_id})

            # Retrieve context
            stream = pipeline.stream_answer(data["query"], data.get("top_k", 5))
            async for token in stream:
                await websocket.send_json({"type": "token", "content": token})

            await websocket.send_json({"type": "done", "query_id": query_id})
    except WebSocketDisconnect:
        pass
```

---

## Phase 7: Frontend Dashboard (Days 8–10)

### Dashboard Pages

| Page | File | Features |
|------|------|---------|
| Main | app.py | Navigation, sidebar, session state |
| Chat | pages/chat.py | Chat interface, streaming, citations |
| Documents | pages/documents.py | Upload, library, delete |
| Settings | pages/settings.py | Model, retrieval, chunk settings |

### Key Streamlit Components

```python
# pages/chat.py — core chat loop
import streamlit as st
import requests

def render_chat():
    st.title("💬 Ask Your Documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**[{src['source_number']}] {src['filename']}**")
                        st.caption(src["excerpt"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                response = requests.post(
                    "http://localhost:8000/api/query",
                    json={"query": prompt, "model": st.session_state.get("model", "gpt-4")},
                ).json()

            st.markdown(response["answer"])
            with st.expander("📚 Sources"):
                for src in response["sources"]:
                    st.markdown(f"**[{src['source_number']}] {src['filename']}**")
                    st.caption(src["excerpt"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
        })
```

---

## Phase 8: Evaluation (Days 10–11)

### RAGAS Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Faithfulness | Are answers grounded in context? | ≥ 0.85 |
| Answer Relevancy | Does answer address the question? | ≥ 0.85 |
| Context Precision | Are retrieved chunks relevant? | ≥ 0.80 |
| Context Recall | Are all relevant chunks retrieved? | ≥ 0.80 |

### File: src/evaluation.py

```python
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

class RAGEvaluator:

    def evaluate_rag(
        self,
        test_queries: List[str],
        ground_truth_answers: List[str],
        pipeline,
    ) -> Dict[str, float]:
        """
        Run full RAGAS evaluation.

        Args:
            test_queries: List of test questions.
            ground_truth_answers: Expected correct answers.
            pipeline: RAG pipeline instance.

        Returns:
            Dict of metric names to scores.
        """
        results = {"questions": [], "answers": [], "contexts": [], "ground_truths": []}

        for query, truth in zip(test_queries, ground_truth_answers):
            answer, contexts = pipeline.query(query, return_contexts=True)
            results["questions"].append(query)
            results["answers"].append(answer)
            results["contexts"].append([c["text"] for c in contexts])
            results["ground_truths"].append(truth)

        dataset = Dataset.from_dict(results)
        scores = evaluate(dataset, metrics=[
            faithfulness, answer_relevancy, context_precision, context_recall
        ])
        return scores

    def generate_test_dataset(self, documents: List[str], n_questions: int = 50) -> List[Dict]:
        """Generate synthetic Q&A pairs from documents using LLM."""
        from openai import OpenAI
        client = OpenAI()
        qa_pairs = []

        for doc_chunk in documents[:n_questions]:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "user",
                    "content": f"""Generate 1 question and its answer from this text.
Return JSON: {{"question": "...", "answer": "..."}}

Text: {doc_chunk[:500]}"""
                }],
            )
            import json
            try:
                qa = json.loads(response.choices[0].message.content)
                qa_pairs.append(qa)
            except json.JSONDecodeError:
                continue
        return qa_pairs

    def compare_retrieval_strategies(
        self, strategies: List[str], test_queries: List[str], pipeline
    ) -> Dict[str, Dict]:
        """A/B test dense vs sparse vs hybrid retrieval."""
        results = {}
        for strategy in strategies:
            pipeline.retrieval_type = strategy
            scores = self.evaluate_rag(test_queries, [], pipeline)
            results[strategy] = scores
        return results
```

---

## Phase 9: Deployment (Days 11–12)

### Docker Compose (Full Stack)

```yaml
# docker-compose.yml
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]
    restart: unless-stopped

  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      qdrant:
        condition: service_healthy
    restart: unless-stopped

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports: ["8501:8501"]
    environment:
      - API_URL=http://api:8000
    depends_on: [api]
    restart: unless-stopped

volumes:
  qdrant_data:
```

### GitHub Actions CI/CD (.github/workflows/ci.yml)

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short
      - run: ruff check src/
      - run: mypy src/ --ignore-missing-imports
```

---

## Dependencies & Setup

### requirements.txt

```
langchain>=0.1.0
langchain-community>=0.0.10
sentence-transformers>=2.3.0
qdrant-client>=1.7.0
fastapi>=0.109.0
uvicorn>=0.27.0
streamlit>=1.30.0
pypdf>=4.0.0
python-docx>=1.0.0
beautifulsoup4>=4.12.0
ragas>=0.1.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-multipart>=0.0.6
websockets>=12.0
rank-bm25>=0.2.2
openai>=1.6.0
anthropic>=0.18.0
slowapi>=0.1.9
nltk>=3.8.0
numpy>=1.24.0
```

### Environment Variables (.env.example)

```bash
# LLM Provider Keys
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Vector DB
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
COLLECTION_NAME=rag_documents

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2
USE_OPENAI_EMBEDDINGS=false

# App Config
DEFAULT_LLM=gpt-4
CHUNK_SIZE=512
CHUNK_OVERLAP=50
RETRIEVAL_TYPE=hybrid
TOP_K=5
```

---

## Success Criteria

- [ ] Documents upload and index in < 30 seconds for 10-page PDF
- [ ] Query latency < 5 seconds end-to-end
- [ ] RAGAS faithfulness score ≥ 0.85
- [ ] RAGAS answer relevancy score ≥ 0.85
- [ ] Supports all 6 document formats without errors
- [ ] Docker Compose deployment works in single command
- [ ] WebSocket streaming produces first token < 1 second
- [ ] Re-ranking improves precision by ≥ 10% over dense-only
