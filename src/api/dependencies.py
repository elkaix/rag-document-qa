"""
FastAPI dependency functions for injecting shared resources into route handlers.

RAG Pipeline Position:
  [DEPENDENCIES] sit between the lifespan (which creates resources) and
  the routes (which consume them). They are the "wiring" that makes
  dependency injection work in FastAPI.

  Lifespan (creates backend) -> Dependencies (provides backend) -> Routes (use backend)

What concept it teaches:
  FastAPI's Depends() system. Instead of reaching into request.app.state
  directly, routes declare their dependencies as function parameters.
  This makes dependencies explicit, testable, and swappable.

Why this approach over alternatives:
  Accessing request.app.state.backend directly works, but:
  1. It's implicit — the route signature doesn't show what it needs.
  2. It's harder to mock in tests — you have to patch app.state.
  With Depends(), you can override get_backend in tests with a mock.
"""

from __future__ import annotations

from fastapi import Request

from src.backend import RAGBackend


def get_backend(request: Request) -> RAGBackend:
    """Extract the shared RAGBackend instance from FastAPI application state.

    The backend is created once during the lifespan startup hook and stored
    on app.state. This dependency function simply retrieves it, making the
    dependency explicit in route handler signatures.

    Args:
        request: The incoming HTTP request (injected by FastAPI).

    Returns:
        The shared RAGBackend instance.

    PATTERN: Thin dependency — no logic, just extraction. This keeps the
    dependency function trivial to understand and test. All business logic
    stays in RAGBackend.
    """
    return request.app.state.backend
