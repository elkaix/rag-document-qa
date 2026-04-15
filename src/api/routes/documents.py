"""
Document management routes:
    GET    /api/documents
    DELETE /api/documents/{doc_id}
    GET    /api/documents/{doc_id}/chunks

Uses the shared RAGBackend (app.state.backend) for all storage operations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, status

from src.api.models import DocumentInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])


@router.get(
    "/documents",
    response_model=List[DocumentInfo],
    summary="List all indexed documents",
)
async def list_documents(request: Request) -> List[DocumentInfo]:
    """Return metadata for every document currently indexed in the vector store."""
    backend = request.app.state.backend
    entries = backend.list_documents()

    # BUG FIX: Backend returns "id" and "chunks_count" (matching DocumentRecord
    #          field names), not "doc_id" and "chunks" (the old in-memory format).
    return [
        DocumentInfo(
            doc_id=entry["id"],
            filename=entry["filename"],
            chunks=entry["chunks_count"],
            upload_date=datetime.fromisoformat(entry["upload_date"]),
            file_type=entry.get("file_type"),
            file_size_bytes=entry.get("file_size_bytes"),
        )
        for entry in entries
    ]


@router.delete(
    "/documents/{doc_id}",
    summary="Delete a document and all its chunks",
    status_code=status.HTTP_200_OK,
)
async def delete_document(doc_id: str, request: Request) -> Dict[str, Any]:
    """Delete a document (and all its indexed chunks) by doc_id.

    Returns a JSON object with 'doc_id', 'chunks_deleted', and 'status'.
    """
    backend = request.app.state.backend

    # Check existence
    docs = {d["id"]: d for d in backend.list_documents()}
    if doc_id not in docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    try:
        chunks_deleted = backend.delete_document(doc_id)
    except Exception as exc:
        logger.error("Failed to delete document %s: %s", doc_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion failed: {exc}",
        ) from exc

    logger.info("Deleted doc_id=%s (%d chunks)", doc_id, chunks_deleted)
    return {"doc_id": doc_id, "chunks_deleted": chunks_deleted, "status": "deleted"}


@router.get(
    "/documents/{doc_id}/chunks",
    summary="List all chunks for a document",
)
async def get_document_chunks(doc_id: str, request: Request) -> Dict[str, Any]:
    """Return all indexed chunks for a specific document.

    The response includes the doc_id, filename, and a list of chunk dicts
    containing chunk_id, content excerpt, and metadata.
    """
    backend = request.app.state.backend

    # Check existence
    docs = {d["id"]: d for d in backend.list_documents()}
    if doc_id not in docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    raw_chunks = backend.get_document_chunks(doc_id)

    chunks = [
        {
            "chunk_id": c["chunk_id"],
            "excerpt": c["text"][:300],
            "metadata": c["metadata"],
        }
        for c in raw_chunks
    ]

    return {
        "doc_id": doc_id,
        "filename": docs[doc_id]["filename"],
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
