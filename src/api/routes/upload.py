"""
Upload routes: POST /api/upload and POST /api/upload/batch

Uses the shared RAGBackend (app.state.backend) for ingestion.
Validation (file size, extension) is done here; temp-file handling is
delegated to backend.ingest_bytes().
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Request, UploadFile, status

from src.api.models import UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])

from src.document_loader import SUPPORTED_EXTENSIONS as ALLOWED_EXTENSIONS
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def _validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )


async def _process_upload(file: UploadFile, request: Request) -> UploadResponse:
    """Validate and ingest a single uploaded file via the backend."""
    filename = file.filename or "upload"
    _validate_extension(filename)

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    backend = request.app.state.backend

    try:
        result = backend.ingest_bytes(filename, contents)
    except Exception as exc:
        logger.error("Upload processing failed for '%s': %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return UploadResponse(
        document_id=result["doc_id"],
        filename=result["filename"],
        chunks_count=result["chunks_count"],
        status=result["status"],
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a single document",
    status_code=status.HTTP_201_CREATED,
)
async def upload_single(file: UploadFile, request: Request) -> UploadResponse:
    """Upload and index a single document file.

    - **file**: Multipart file upload (PDF, DOCX, TXT, MD, HTML, CSV, JSON).

    Returns document ID, filename, chunk count, and status.
    """
    return await _process_upload(file, request)


@router.post(
    "/upload/batch",
    response_model=List[UploadResponse],
    summary="Upload multiple documents",
    status_code=status.HTTP_201_CREATED,
)
async def upload_batch(files: List[UploadFile], request: Request) -> List[UploadResponse]:
    """Upload and index multiple document files in one request.

    - **files**: List of multipart file uploads.

    Returns a list of upload results (one per file; errors are included inline).
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided.",
        )

    results: List[UploadResponse] = []
    for file in files:
        try:
            result = await _process_upload(file, request)
            results.append(result)
        except HTTPException as exc:
            results.append(
                UploadResponse(
                    document_id="",
                    filename=file.filename or "unknown",
                    chunks_count=0,
                    status="error",
                    message=exc.detail,
                )
            )
    return results
