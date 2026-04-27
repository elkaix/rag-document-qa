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
    # WHY query_with_telemetry: replaces the plain query() call so we get
    #     per-stage timing and token-cost numbers in the response. The
    #     result_dict has the same shape as before — only telemetry is new.
    result, telemetry = backend.query_with_telemetry(
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
        # ADDITIVE: telemetry field added in Task 5 (Sub-plan 1D).
        telemetry=telemetry,
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

            # BUG FIX: `int(payload.get("top_k", 5))` sat OUTSIDE the inner
            #          try/except below, so a bad client value raised
            #          ValueError up through the outer WebSocketDisconnect
            #          handler, which silently killed the socket. Validate
            #          + bounds-check here and emit a structured error event
            #          so the client can surface it without dropping the
            #          connection. Matches QueryRequest's le=50 bound.
            raw_top_k = payload.get("top_k", 5)
            try:
                top_k = int(raw_top_k)
            except (TypeError, ValueError):
                await websocket.send_json({
                    "type": "error",
                    "content": f"Invalid top_k: {raw_top_k!r}",
                })
                continue
            if not (1 <= top_k <= 50):
                await websocket.send_json({
                    "type": "error",
                    "content": "top_k must be between 1 and 50.",
                })
                continue
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

            # Accumulate answer tokens and retrieved contexts so we can run
            # faithfulness evaluation after the stream completes without
            # blocking or delaying any part of the answer delivery.
            full_answer_parts: list[str] = []
            retrieved_contexts: list[str] = []
            done_data: dict = {}

            try:
                while True:
                    item = await loop.run_in_executor(
                        None, next, gen, _sentinel
                    )
                    if item is _sentinel:
                        break
                    event_type, data = item

                    if event_type == "token":
                        full_answer_parts.append(data)
                        await websocket.send_json({"type": "token", "content": data})
                    elif event_type == "reasoning":
                        await websocket.send_json({"type": "reasoning", "content": data})
                    elif event_type == "status":
                        await websocket.send_json({"type": "status", "content": data})
                    elif event_type == "done":
                        done_data = data
                        retrieved_contexts = [
                            s.get("excerpt", "") for s in data.get("sources", [])
                        ]
                        await websocket.send_json({
                            "type": "done",
                            "sources": data.get("sources", []),
                            "message_id": data.get("message_id"),
                            "conversation_id": data.get("conversation_id"),
                        })
                    elif event_type == "telemetry":
                        # WHY: stream_query yields ("telemetry", StageTelemetry.model_dump())
                        #      after the done event. Forward it verbatim so the frontend
                        #      can render per-stage timing and cost without polling.
                        await websocket.send_json({"type": "telemetry", "content": data})
            except Exception as exc:
                logger.error("Streaming error: %s", exc)
                await websocket.send_json(
                    {"type": "error", "content": f"Streaming error: {exc}"}
                )
            finally:
                gen.close()

            # WHY: Real-time faithfulness fires AFTER streaming completes so
            #      the user sees the answer immediately. The evaluation result
            #      is sent as a separate WebSocket event that the frontend
            #      uses to update the inline faithfulness badge.
            message_id = done_data.get("message_id")
            full_answer = "".join(full_answer_parts)
            if message_id and full_answer and retrieved_contexts:
                try:
                    loop_ref = asyncio.get_running_loop()
                    eval_result = await loop_ref.run_in_executor(
                        None,
                        backend.evaluate_faithfulness_realtime,
                        message_id,
                        full_answer,
                        retrieved_contexts,
                    )
                    await websocket.send_json({
                        "type": "evaluation",
                        "content": eval_result,
                    })
                except Exception as exc:
                    logger.warning("Real-time faithfulness evaluation failed: %s", exc)

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
