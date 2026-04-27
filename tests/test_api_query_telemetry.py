"""Tests for telemetry surfacing in the REST and WebSocket query routes.

API Layer Position:
  RAGBackend.query_with_telemetry → (result_dict, StageTelemetry)
  POST /api/query → QueryResponse with `telemetry` field   [REST]
  GET  /api/chat  → WebSocket stream ends with telemetry event  [WS]

What concept it teaches:
  Route-layer serialization testing: mock the backend dependency, assert the
  route converts the returned StageTelemetry into the right JSON shape.
  This isolates the serialization concern from real LLM / ChromaDB I/O.

Why mock approach instead of full-ingest:
  - RAGBackend telemetry behaviour is already tested in test_backend_telemetry.py.
  - Task 5 only changes the route layer (query.py + models.py).
  - Mocking the backend keeps tests fast (<1 s) and dependency-free.
"""

from __future__ import annotations

import json
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FAKE_TELEMETRY = {
    "retrieve_ms": 42.5,
    "generate_ms": 310.0,
    "prompt_tokens": 512,
    "completion_tokens": 128,
    "cost_usd": 0.0034,
}

_FAKE_RESULT = {
    "answer": "RAG combines retrieval with generation.",
    "sources": [
        {
            "doc_id": "doc-1",
            "chunk_id": "chunk-1",
            "filename": "rag.txt",
            "score": 0.92,
            "excerpt": "RAG stands for Retrieval-Augmented Generation.",
        }
    ],
    "confidence": 0.85,
}


def _make_telemetry_model():
    """Return a StageTelemetry instance with the fake values."""
    from src.api.schemas.telemetry import StageTelemetry
    return StageTelemetry(**_FAKE_TELEMETRY)


# ---------------------------------------------------------------------------
# REST endpoint — POST /api/query
# ---------------------------------------------------------------------------


class TestRestTelemetry:
    """POST /api/query response must include a well-formed `telemetry` field."""

    @pytest.fixture
    def client(self):
        """TestClient with a mocked backend attached to app.state.

        WHY: TestClient runs the FastAPI lifespan, which creates a real
        RAGBackend on app.state. We replace it with a MagicMock after
        startup so query_with_telemetry() returns predictable values
        without touching ChromaDB or an LLM.
        """
        from src.api.main import app
        with TestClient(app) as c:
            mock_backend = MagicMock()
            mock_backend.query_with_telemetry.return_value = (
                _FAKE_RESULT, _make_telemetry_model()
            )
            # evaluate_faithfulness_realtime must not raise during WS teardown
            mock_backend.evaluate_faithfulness_realtime.return_value = {}
            app.state.backend = mock_backend
            yield c

    def test_response_includes_telemetry_field(self, client):
        """The telemetry object appears at the top level of the JSON response."""
        r = client.post("/api/query", json={"query": "What is RAG?"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "telemetry" in body, f"'telemetry' key missing from response: {body}"

    def test_telemetry_has_all_five_fields(self, client):
        """All five StageTelemetry fields are present in the response body."""
        r = client.post("/api/query", json={"query": "What is RAG?"})
        assert r.status_code == 200
        t = r.json()["telemetry"]
        for field in ("retrieve_ms", "generate_ms", "prompt_tokens",
                      "completion_tokens", "cost_usd"):
            assert field in t, f"Missing telemetry field: {field}"
            assert t[field] >= 0, f"Telemetry field {field} must be >= 0"

    def test_telemetry_values_match_backend(self, client):
        """Telemetry values round-trip correctly from backend to JSON response."""
        r = client.post("/api/query", json={"query": "What is RAG?"})
        t = r.json()["telemetry"]
        assert t["retrieve_ms"] == _FAKE_TELEMETRY["retrieve_ms"]
        assert t["generate_ms"] == _FAKE_TELEMETRY["generate_ms"]
        assert t["prompt_tokens"] == _FAKE_TELEMETRY["prompt_tokens"]
        assert t["completion_tokens"] == _FAKE_TELEMETRY["completion_tokens"]
        assert t["cost_usd"] == pytest.approx(_FAKE_TELEMETRY["cost_usd"])

    def test_existing_response_fields_unchanged(self, client):
        """Adding telemetry must not drop or alter existing response fields."""
        r = client.post("/api/query", json={"query": "What is RAG?"})
        body = r.json()
        assert "answer" in body
        assert "sources" in body
        assert "confidence" in body
        assert "latency_ms" in body

    def test_query_with_telemetry_is_called_not_query(self, client):
        """The route must call query_with_telemetry, not the plain query().

        WHY: If someone reverts to backend.query() the telemetry field would
        silently become None (the Optional default). This assertion catches it.
        """
        from src.api.main import app
        client.post("/api/query", json={"query": "test"})
        app.state.backend.query_with_telemetry.assert_called_once()
        app.state.backend.query.assert_not_called()


# ---------------------------------------------------------------------------
# WebSocket endpoint — /api/chat
# ---------------------------------------------------------------------------


class TestWebSocketTelemetry:
    """WebSocket /api/chat must forward the telemetry event from stream_query."""

    def _make_stream(self) -> list[tuple[str, object]]:
        """Minimal event stream: status → done → telemetry."""
        return [
            ("status", "Searching indexed documents..."),
            ("token", "RAG combines retrieval with generation."),
            ("done", {
                "sources": [
                    {
                        "doc_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "filename": "rag.txt",
                        "score": 0.92,
                        "excerpt": "RAG stands for Retrieval-Augmented Generation.",
                    }
                ],
                "message_id": "msg-abc",
                "conversation_id": "conv-xyz",
            }),
            ("telemetry", _FAKE_TELEMETRY),
        ]

    @pytest.fixture
    def ws_client(self):
        """TestClient with a mocked streaming backend."""
        from src.api.main import app
        with TestClient(app) as c:
            mock_backend = MagicMock()

            def _fake_stream(*args, **kwargs) -> Iterator:
                yield from self._make_stream()

            mock_backend.stream_query.side_effect = _fake_stream
            mock_backend.evaluate_faithfulness_realtime.return_value = {}
            app.state.backend = mock_backend
            yield c

    def test_stream_emits_telemetry_event(self, ws_client):
        """The WebSocket stream must include a telemetry event."""
        from src.api.main import app
        with ws_client.websocket_connect("/api/chat") as ws:
            ws.send_json({"query": "What is RAG?", "top_k": 3})
            events = []
            # Drain all events until we see telemetry or hit 20 messages.
            # WHY cap: if telemetry is never emitted, we stop instead of hanging.
            for _ in range(20):
                try:
                    msg = ws.receive_json()
                    events.append(msg)
                    if msg.get("type") == "telemetry":
                        break
                except Exception:
                    break

        assert any(e.get("type") == "telemetry" for e in events), (
            f"No telemetry event received. Got event types: {[e.get('type') for e in events]}"
        )

    def test_telemetry_event_has_content_key(self, ws_client):
        """The telemetry event must be shaped: {type: 'telemetry', content: {...}}."""
        with ws_client.websocket_connect("/api/chat") as ws:
            ws.send_json({"query": "What is RAG?", "top_k": 3})
            tele_event = None
            for _ in range(20):
                try:
                    msg = ws.receive_json()
                    if msg.get("type") == "telemetry":
                        tele_event = msg
                        break
                except Exception:
                    break

        assert tele_event is not None, "No telemetry event received."
        assert "content" in tele_event, f"telemetry event missing 'content' key: {tele_event}"

    def test_telemetry_content_has_all_five_fields(self, ws_client):
        """All five StageTelemetry fields must appear inside the content dict."""
        with ws_client.websocket_connect("/api/chat") as ws:
            ws.send_json({"query": "What is RAG?", "top_k": 3})
            tele_event = None
            for _ in range(20):
                try:
                    msg = ws.receive_json()
                    if msg.get("type") == "telemetry":
                        tele_event = msg
                        break
                except Exception:
                    break

        content = tele_event["content"]
        for field in ("retrieve_ms", "generate_ms", "prompt_tokens",
                      "completion_tokens", "cost_usd"):
            assert field in content, f"Missing telemetry field: {field}"
            assert content[field] >= 0, f"Telemetry field {field} must be >= 0"

    def test_done_event_shape_unchanged(self, ws_client):
        """Adding telemetry must not modify the done event shape.

        STRICT: done must still carry sources, message_id, conversation_id.
        This guards against accidentally merging telemetry into done.
        """
        with ws_client.websocket_connect("/api/chat") as ws:
            ws.send_json({"query": "What is RAG?", "top_k": 3})
            done_event = None
            for _ in range(20):
                try:
                    msg = ws.receive_json()
                    if msg.get("type") == "done":
                        done_event = msg
                    if msg.get("type") == "telemetry":
                        break
                except Exception:
                    break

        assert done_event is not None, "No done event received."
        assert "sources" in done_event, f"done event missing 'sources': {done_event}"
        assert "message_id" in done_event
        assert "conversation_id" in done_event
        # telemetry must NOT be embedded inside done
        assert "telemetry" not in done_event
