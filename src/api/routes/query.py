"""
Query routes: POST /api/query and WebSocket /api/chat

Uses the shared RAGBackend (app.state.backend) for retrieval and generation.
No shared mutable state is touched — per-request LLM handlers are created
when the requested model differs from the default.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status

from src.api.models import QueryRequest, QueryResponse, SourceInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against indexed documents",
)
async def query(request_body: QueryRequest, request: Request) -> QueryResponse:
    """Submit a question and receive a RAG-generated answer with source citations.

    - **query**: The user question.
    - **top_k**: Number of document chunks to retrieve (default 5).
    - **model**: LLM model name for answer generation.
    """
    start = time.perf_counter()

    backend = request.app.state.backend
    result = backend.query(
        request_body.query,
        top_k=request_body.top_k,
        model=request_body.model,
    )

    sources = [
        SourceInfo(
            doc_id=s["doc_id"],
            chunk_id=s["chunk_id"],
            filename=s.get("filename"),
            score=s["score"],
            excerpt=s["excerpt"][:200],
        )
        for s in result.get("sources", [])
    ]

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        confidence=result.get("confidence", 0.0),
        latency_ms=latency_ms,
    )


@router.websocket("/chat")
async def chat_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming chat responses with chain-of-thought.

    Send a JSON message:
        {"query": "...", "top_k": 5, "model": "gpt-4", "conversation_id": "..."}

    Receive a stream of events describing what the agent is doing, followed
    by reasoning tokens, then answer tokens, then a final done message:
        {"type": "status",    "content": "Searching indexed documents..."}
        {"type": "status",    "content": "Retrieved 5 chunk(s) across 2 file(s): ..."}
        {"type": "status",    "content": "Analyzing retrieved context..."}
        {"type": "reasoning", "content": "..."}   ← CoT tokens (streamed)
        {"type": "status",    "content": "Composing answer..."}
        {"type": "token",     "content": "..."}   ← answer tokens (streamed)
        {"type": "done",      "sources": [...], "message_id": "...", "conversation_id": "..."}
    """
    await websocket.accept()
    backend = websocket.app.state.backend

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "content": "Invalid JSON."})
                continue

            query_text = payload.get("query", "").strip()
            if not query_text:
                await websocket.send_json({"type": "error", "content": "Empty query."})
                continue

            top_k = int(payload.get("top_k", 5))
            model = payload.get("model")

            # WHY: conversation_id links the WebSocket chat to a persisted
            #      conversation in SQLite. When provided, stream_query saves
            #      user/assistant messages and returns message_id + conversation_id
            #      in the done event so the frontend can update its local state.
            conversation_id = payload.get("conversation_id")

            # WHY run_in_executor: stream_query() is a synchronous generator
            #      that makes blocking HTTP calls to the LLM API. Iterating
            #      it directly in this async handler would block the event
            #      loop, preventing WebSocket frames from being flushed
            #      between tokens — the user would see everything at once
            #      instead of real-time streaming. Running next() in a thread
            #      keeps the event loop free so each send_json flushes
            #      immediately.
            loop = asyncio.get_running_loop()
            gen = backend.stream_query(
                query_text, top_k=top_k, model=model,
                conversation_id=conversation_id,
            )
            _sentinel = object()

            try:
                while True:
                    item = await loop.run_in_executor(
                        None, next, gen, _sentinel
                    )
                    if item is _sentinel:
                        break
                    event_type, data = item

                    if event_type == "token":
                        await websocket.send_json({"type": "token", "content": data})
                    elif event_type == "reasoning":
                        await websocket.send_json({"type": "reasoning", "content": data})
                    elif event_type == "status":
                        await websocket.send_json({"type": "status", "content": data})
                    elif event_type == "done":
                        await websocket.send_json({
                            "type": "done",
                            "sources": data.get("sources", []),
                            "message_id": data.get("message_id"),
                            "conversation_id": data.get("conversation_id"),
                        })
            except Exception as exc:
                logger.error("Streaming error: %s", exc)
                await websocket.send_json(
                    {"type": "error", "content": f"Streaming error: {exc}"}
                )
            finally:
                gen.close()

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
