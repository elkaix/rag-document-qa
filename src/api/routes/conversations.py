"""
Conversation CRUD routes — manage chat sessions and their messages.

RAG Pipeline Position:
  Document -> Chunks -> Embeddings -> Vector Store -> Retrieval -> Generator
                                         |
                                   [CONVERSATIONS] <-> Messages <-> Sources
                                         |
                                    (CRUD via these REST endpoints)

What concept it teaches:
  Full REST resource management with FastAPI's APIRouter. Each conversation
  is a resource with standard CRUD operations plus specialized actions
  (search, export, share). The routes are thin — all business logic lives
  in RAGBackend.

Why this approach over alternatives:
  Using Depends(get_backend) instead of request.app.state.backend makes
  dependencies explicit in function signatures. This is the modern FastAPI
  pattern — it enables easy mocking in tests and self-documenting routes.

Where it fits in the RAG pipeline:
  These routes sit in the API layer between the frontend (React) and the
  backend (RAGBackend). They validate input via Pydantic models, delegate
  to the backend, and format responses.
"""

from __future__ import annotations

import logging
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import PlainTextResponse

from src.api.dependencies import get_backend
from src.api.models import (
    ConversationCreate,
    ConversationDetail,
    ConversationSummary,
    ConversationUpdate,
    MessageInfo,
    SourceInfo,
)
from src.backend import RAGBackend

logger = logging.getLogger(__name__)

# WHY: prefix="/api" groups all conversation routes under /api/conversations.
#      tags=["conversations"] groups them in the OpenAPI docs sidebar.
router = APIRouter(prefix="/api", tags=["conversations"])

# PATTERN: Annotated dependency — the modern FastAPI way to declare dependencies.
#          Instead of `backend = Depends(get_backend)` as a default param, we use
#          Annotated[RAGBackend, Depends(get_backend)] which is clearer in type
#          checkers and avoids the "mutable default argument" anti-pattern.
BackendDep = Annotated[RAGBackend, Depends(get_backend)]


# --------------------------------------------------------------------------- #
# Helper: convert backend dict -> Pydantic ConversationSummary                 #
# --------------------------------------------------------------------------- #

def _to_summary(data: dict) -> ConversationSummary:
    """Convert a backend conversation dict to a ConversationSummary response model.

    WHY: The backend returns plain dicts (not Pydantic models) to stay
    framework-agnostic. This helper bridges the gap, handling optional
    fields like share_token that may be absent from the dict.
    """
    return ConversationSummary(
        id=data["id"],
        title=data["title"],
        pinned=data["pinned"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        share_token=data.get("share_token"),
    )


def _to_detail(data: dict) -> ConversationDetail:
    """Convert a backend conversation dict (with messages) to ConversationDetail.

    Each message's sources are mapped to SourceInfo Pydantic models.
    """
    messages = [
        MessageInfo(
            id=msg["id"],
            role=msg["role"],
            content=msg["content"],
            model=msg.get("model"),
            created_at=msg["created_at"],
            sources=[
                SourceInfo(
                    doc_id=s["doc_id"],
                    chunk_id=s["chunk_id"],
                    filename=s.get("filename"),
                    score=s.get("score", 0.0),
                    excerpt=s.get("excerpt", ""),
                )
                for s in msg.get("sources", [])
            ],
        )
        for msg in data.get("messages", [])
    ]

    return ConversationDetail(
        id=data["id"],
        title=data["title"],
        pinned=data["pinned"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        share_token=data.get("share_token"),
        messages=messages,
    )


# --------------------------------------------------------------------------- #
# List & Create                                                                #
# --------------------------------------------------------------------------- #

@router.get(
    "/conversations",
    response_model=List[ConversationSummary],
    summary="List all conversations",
)
def list_conversations(backend: BackendDep) -> List[ConversationSummary]:
    """Return all conversations, pinned first, then by most recently updated.

    WHY sync def: The backend performs synchronous SQLite queries. FastAPI
    automatically runs sync handlers in a threadpool, so they don't block
    the async event loop.
    """
    conversations = backend.list_conversations()
    return [_to_summary(c) for c in conversations]


@router.post(
    "/conversations",
    response_model=ConversationSummary,
    summary="Create a new conversation",
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    body: ConversationCreate,
    backend: BackendDep,
) -> ConversationSummary:
    """Create a new empty conversation with an optional title.

    The frontend calls this when the user clicks "New Chat" in the sidebar.
    """
    data = backend.create_conversation(title=body.title)
    return _to_summary(data)


# --------------------------------------------------------------------------- #
# Search — MUST be defined BEFORE /{conversation_id} routes                    #
# --------------------------------------------------------------------------- #
# WHY: FastAPI matches routes in definition order. If /{conversation_id}
#      were defined first, a request to /conversations/search would match
#      with conversation_id="search" and return 404. Defining /search
#      first ensures it matches literal "search" before the path param.

@router.get(
    "/conversations/search",
    response_model=List[ConversationSummary],
    summary="Search conversations by title or message content",
)
def search_conversations(
    backend: BackendDep,
    q: str = Query(..., min_length=1, max_length=500, description="Search query string."),
) -> List[ConversationSummary]:
    """Search conversations by title or message content using substring matching.

    TRADE-OFF: Uses SQL LIKE for simplicity. Production would use SQLite FTS5
    for full-text search with ranking. Adequate for a portfolio demo.
    """
    results = backend.search_conversations(q)
    return [_to_summary(c) for c in results]


# --------------------------------------------------------------------------- #
# Detail, Update, Delete — parameterised by {conversation_id}                  #
# --------------------------------------------------------------------------- #

@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get conversation with messages",
)
def get_conversation(
    conversation_id: str,
    backend: BackendDep,
) -> ConversationDetail:
    """Return a single conversation including all messages and their sources.

    The frontend calls this when the user selects a conversation from the
    sidebar to load the full chat history.
    """
    data = backend.get_conversation(conversation_id)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return _to_detail(data)


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationSummary,
    summary="Update conversation title or pinned status",
)
def update_conversation(
    conversation_id: str,
    body: ConversationUpdate,
    backend: BackendDep,
) -> ConversationSummary:
    """Partially update a conversation. Only provided fields are changed.

    PATTERN: PATCH (not PUT) because only a subset of fields can be updated.
    PUT would imply replacing the entire resource, which doesn't make sense
    for conversations that accumulate messages over time.
    """
    data = backend.update_conversation(
        conversation_id,
        title=body.title,
        pinned=body.pinned,
    )
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return _to_summary(data)


@router.delete(
    "/conversations/{conversation_id}",
    summary="Delete a conversation and all its messages",
    status_code=status.HTTP_200_OK,
)
def delete_conversation(
    conversation_id: str,
    backend: BackendDep,
) -> dict:
    """Delete a conversation, cascading to all messages and sources.

    WHY cascade: ON DELETE CASCADE in the FK definitions handles cleanup
    automatically. See database.py for the PRAGMA foreign_keys=ON listener.
    """
    deleted = backend.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return {"id": conversation_id, "status": "deleted"}


# --------------------------------------------------------------------------- #
# Export & Share                                                                #
# --------------------------------------------------------------------------- #

@router.get(
    "/conversations/{conversation_id}/export",
    response_class=PlainTextResponse,
    summary="Export conversation as Markdown",
)
def export_conversation(
    conversation_id: str,
    backend: BackendDep,
) -> PlainTextResponse:
    """Export a conversation as a Markdown-formatted plain text document.

    Returns Content-Type: text/plain with the full conversation in a
    human-readable format suitable for saving or sharing.
    """
    markdown = backend.export_conversation(conversation_id)
    if markdown is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return PlainTextResponse(content=markdown, media_type="text/markdown")


@router.post(
    "/conversations/{conversation_id}/share",
    summary="Generate a share token for read-only access",
    status_code=status.HTTP_201_CREATED,
)
def create_share_token(
    conversation_id: str,
    backend: BackendDep,
    request: Request,
) -> dict:
    """Generate an opaque, unguessable share token for a conversation.

    Anyone with the token can view the conversation read-only via the
    /api/shared/{token} endpoint. Tokens are UUID4 — unguessable and
    not sequential.

    SECURITY: The token is the only access control. Do not expose it in
    URLs that get logged (e.g. query parameters). The frontend should
    store it in the path segment only.
    """
    token = backend.create_share_token(conversation_id)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    # WHY: The frontend writes share_url straight to the clipboard. Building
    #      it server-side from the request's scheme+host keeps it correct
    #      behind proxies/reverse-proxies that rewrite the public origin.
    base = str(request.base_url).rstrip("/")
    share_url = f"{base}/shared/{token}"
    return {
        "conversation_id": conversation_id,
        "share_token": token,
        "share_url": share_url,
    }


# --------------------------------------------------------------------------- #
# Shared (public read-only) — uses /api/shared/{token} path                    #
# --------------------------------------------------------------------------- #

@router.get(
    "/shared/{token}",
    response_model=ConversationDetail,
    summary="View a shared conversation (read-only, no auth)",
)
def get_shared_conversation(
    token: str,
    backend: BackendDep,
) -> ConversationDetail:
    """Retrieve a conversation by its share token for public read-only access.

    WHY separate endpoint: Shared conversations are accessed by token (not ID),
    so they need their own route. This endpoint requires no authentication —
    the token itself serves as a capability-based access control.
    """
    data = backend.get_shared_conversation(token)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Shared conversation not found or token is invalid.",
        )
    return _to_detail(data)
