# Routes package
from .documents import router as documents_router
from .query import router as query_router
from .upload import router as upload_router

__all__ = ["upload_router", "query_router", "documents_router"]
