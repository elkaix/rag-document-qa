"""
Pydantic v2 request/response models for the RAG API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""

    query: str = Field(..., min_length=1, max_length=4096, description="The user question.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve.")
    strategy: str = Field(
        default="dense",
        description="Retrieval strategy: 'dense', 'hybrid', or 'rerank'.",
    )
    model: str = Field(
        default="gpt-4",
        description="LLM model name to use for answer generation.",
    )

    model_config = {"json_schema_extra": {"examples": [{"query": "What is RAG?", "top_k": 5, "strategy": "dense", "model": "gpt-4"}]}}


class SourceInfo(BaseModel):
    """Metadata about a single retrieved source chunk."""

    doc_id: str
    chunk_id: str
    filename: Optional[str] = None
    score: float
    excerpt: str = Field(description="Short excerpt from the source chunk.")


class QueryResponse(BaseModel):
    """Response body for the query endpoint."""

    answer: str = Field(description="Generated answer.")
    sources: List[SourceInfo] = Field(default_factory=list, description="Retrieved source chunks.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Estimated answer confidence (0–1)."
    )
    latency_ms: float = Field(description="Total request latency in milliseconds.")


class UploadResponse(BaseModel):
    """Response body for the file upload endpoint."""

    document_id: str = Field(description="Unique identifier assigned to the uploaded document.")
    filename: str = Field(description="Original filename.")
    chunks_count: int = Field(ge=0, description="Number of chunks indexed.")
    status: str = Field(description="Processing status: 'success' or 'error'.")
    message: Optional[str] = Field(default=None, description="Optional detail message.")


class DocumentInfo(BaseModel):
    """Metadata for a single indexed document."""

    doc_id: str
    filename: str
    chunks: int = Field(ge=0, description="Number of indexed chunks.")
    upload_date: datetime
    file_type: Optional[str] = None
    file_size_bytes: Optional[int] = None


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str = Field(description="Short error code or type.")
    detail: Optional[str] = Field(default=None, description="Detailed error message.")
    request_id: Optional[str] = Field(default=None, description="Optional request trace ID.")
