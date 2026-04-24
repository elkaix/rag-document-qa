"""
Pydantic v2 request/response models for the RAG API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.config import DEFAULT_MODEL


class QueryRequest(BaseModel):
    """Request body for the query endpoint.

    BUG FIX: The `model` default used to be a hard-coded "gpt-4" which
    drifted from the app-wide DEFAULT_MODEL ("gpt-5-mini") in src/config.py,
    so /api/query and the WebSocket path answered with different models
    for the same user. Now both read from the same constant.

    BUG FIX: Removed the `strategy` field — it was declared but never read
    anywhere in the routes or backend, so accepting it was a lie about API
    capabilities. Add it back once hybrid/rerank is actually wired up.
    """

    query: str = Field(..., min_length=1, max_length=4096, description="The user question.")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of chunks to retrieve.")
    model: str = Field(
        default=DEFAULT_MODEL,
        description="LLM model name to use for answer generation.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"query": "What is RAG?", "top_k": 5, "model": DEFAULT_MODEL}]
        }
    }


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


# --------------------------------------------------------------------------- #
# Conversation models                                                          #
# --------------------------------------------------------------------------- #

# WHY: Separate Create/Update models from Summary/Detail models.
#      Create/Update carry user input (validated, constrained).
#      Summary/Detail carry server output (includes generated fields like id, timestamps).
#      This is the standard Pydantic "schema separation" pattern.


class ConversationCreate(BaseModel):
    """Request body for creating a new conversation."""

    title: str = Field(default="New Chat", max_length=200, description="Conversation title.")


class ConversationUpdate(BaseModel):
    """Request body for updating a conversation (partial update via PATCH).

    Both fields are optional — only provided fields are applied.
    This is the standard "partial update" pattern for PATCH endpoints.
    """

    title: Optional[str] = Field(default=None, max_length=200, description="New title.")
    pinned: Optional[bool] = Field(default=None, description="Pin/unpin the conversation.")


class ConversationSummary(BaseModel):
    """Lightweight conversation metadata returned in list endpoints.

    WHY separate from ConversationDetail: List endpoints return many conversations.
    Including full message history in each would be wasteful. Summary has just
    enough for a sidebar listing; Detail adds the messages array.
    """

    id: str
    title: str
    pinned: bool
    created_at: str = Field(description="ISO-8601 UTC timestamp.")
    updated_at: str = Field(description="ISO-8601 UTC timestamp.")
    share_token: Optional[str] = Field(
        default=None,
        description="Opaque share token for read-only public access.",
    )


class MessageInfo(BaseModel):
    """A single message within a conversation detail response.

    Mirrors the Message SQLModel table but as a plain Pydantic model —
    decoupling the API response shape from the database schema.
    """

    id: str
    role: str = Field(description="'user' or 'assistant'.")
    content: str
    model: Optional[str] = Field(default=None, description="LLM model (assistant messages only).")
    created_at: str = Field(description="ISO-8601 UTC timestamp.")
    sources: List[SourceInfo] = Field(
        default_factory=list,
        description="Document chunks cited by this message.",
    )


class ConversationDetail(BaseModel):
    """Full conversation with messages — returned by the detail endpoint.

    Extends ConversationSummary conceptually, but uses composition rather than
    inheritance to keep the Pydantic JSON schema clean and explicit.
    """

    id: str
    title: str
    pinned: bool
    created_at: str
    updated_at: str
    share_token: Optional[str] = None
    messages: List[MessageInfo] = Field(
        default_factory=list,
        description="Chronologically ordered messages in this conversation.",
    )
