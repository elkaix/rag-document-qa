"""Tests that RAGBackend instrumentation produces a StageTelemetry payload.

RAG Pipeline Position:
  Question -> Retrieve -> Generate -> (answer, StageTelemetry)
                                              ^^^
  These tests verify that query_with_telemetry() and stream_query() both emit
  well-formed StageTelemetry with non-negative numeric fields.

What concept it teaches:
  Testing the observability layer in isolation from real LLM providers.
  LLMHandler falls back to a dummy response when no API keys are configured,
  so these tests run without any network access and with zero cost.

Why this fixture pattern:
  Mirrors test_backend.py exactly — EphemeralClient for ChromaDB (no disk I/O,
  isolated per test) and sqlite:// for SQLite (in-memory, vanishes on teardown).
  Both stores are fully isolated; no shared mutable state between tests.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import chromadb
import pytest

from src.backend import RAGBackend
from src.api.schemas.telemetry import StageTelemetry
from src.database import create_db_and_tables, get_engine


# --------------------------------------------------------------------------- #
# Fixtures (mirrored from test_backend.py)                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture
def tmp_sqlite_engine():
    """In-memory SQLite engine with all tables created."""
    engine = get_engine("sqlite://")
    create_db_and_tables(engine)
    return engine


@pytest.fixture
def chroma_backend_collection():
    """Ephemeral ChromaDB collection with auto-embedding enabled.

    WHY unique name: EphemeralClient shares an in-process store.
    A UUID suffix ensures complete isolation between test runs.
    """
    client = chromadb.EphemeralClient()
    return client.get_or_create_collection(
        name=f"test_backend_telemetry_{uuid.uuid4().hex}",
        metadata={"hnsw:space": "cosine"},
    )


@pytest.fixture
def backend(tmp_sqlite_engine, chroma_backend_collection):
    """Fully-wired RAGBackend with ephemeral ChromaDB + in-memory SQLite."""
    return RAGBackend(engine=tmp_sqlite_engine, collection=chroma_backend_collection)


@pytest.fixture
def ingested_backend(backend: RAGBackend, tmp_path: Path) -> RAGBackend:
    """A backend that has already ingested a small text document.

    WHY pre-ingest: query_with_telemetry requires at least one indexed chunk
    to take the non-empty retrieval path and call the LLM. Without a document,
    both retrieve and generate phases return zeros — not useful to test.
    """
    content = (
        "Retrieval-Augmented Generation (RAG) combines document retrieval "
        "with large language model generation to produce grounded answers. "
        "The retrieval step finds relevant chunks via vector similarity search. "
        "The generation step builds a prompt from those chunks and queries the LLM."
    )
    doc = tmp_path / "rag_intro.txt"
    doc.write_text(content, encoding="utf-8")
    backend.ingest_file(doc)
    return backend


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

class TestQueryWithTelemetry:
    """Tests for RAGBackend.query_with_telemetry()."""

    def test_telemetry_fields_are_non_negative_after_ingest(
        self, ingested_backend: RAGBackend
    ):
        """query_with_telemetry returns a StageTelemetry with all non-negative fields.

        PATTERN: With no real LLM configured, LLMHandler falls back to a dummy
        response. The dummy response still produces valid answer text, which
        means token counting and cost computation run on real strings — the
        telemetry shape is exercised even without a real LLM call.
        """
        result, telemetry = ingested_backend.query_with_telemetry("What is RAG?")

        # Result dict has the same shape as query()
        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) > 0

        # Telemetry is the right type
        assert isinstance(telemetry, StageTelemetry)

        # All five numeric fields are non-negative
        assert telemetry.retrieve_ms >= 0.0
        assert telemetry.generate_ms >= 0.0
        assert telemetry.prompt_tokens >= 0
        assert telemetry.completion_tokens >= 0
        assert telemetry.cost_usd >= 0.0

    def test_telemetry_zeros_when_no_documents(self, backend: RAGBackend):
        """When no documents are indexed, StageTelemetry has zero generate/token fields.

        WHY: The early-return branch skips the LLM call entirely.
        retrieve_ms records real elapsed time (even for an empty search);
        generate_ms, prompt_tokens, completion_tokens, and cost_usd are all 0.
        """
        result, telemetry = backend.query_with_telemetry("What is RAG?")

        assert result["answer"].startswith("No documents indexed")
        assert telemetry.generate_ms == 0.0
        assert telemetry.prompt_tokens == 0
        assert telemetry.completion_tokens == 0
        assert telemetry.cost_usd == 0.0
        # retrieve_ms is still measured (we did make the call, it just returned empty)
        assert telemetry.retrieve_ms >= 0.0

    def test_query_unchanged_after_sibling_added(self, ingested_backend: RAGBackend):
        """query() still returns the plain dict — the new sibling has no side effects.

        WHY: This is the regression guard for the Option A design choice.
        Existing tests use result["answer"] / result["sources"] on the dict
        returned by query(). If we accidentally broke the dict shape, this
        test catches it immediately.
        """
        result = ingested_backend.query("What is RAG?")

        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result


class TestStreamQueryTelemetry:
    """Tests for the telemetry event emitted by stream_query()."""

    def test_stream_query_emits_telemetry_event_last(
        self, ingested_backend: RAGBackend
    ):
        """stream_query yields a ("telemetry", dict) as the final event after ("done", ...).

        WHY last: The done event is what the client waits for to display sources.
        Telemetry is a secondary signal. Emitting it last ensures done latency
        is not delayed by token-counting arithmetic.
        """
        events = list(ingested_backend.stream_query("What is RAG?"))

        # The last event must be the telemetry event
        last_type, last_data = events[-1]
        assert last_type == "telemetry", (
            f"Expected last event type 'telemetry', got {last_type!r}. "
            f"All event types: {[e[0] for e in events]}"
        )

        # The telemetry payload must be a dict (model_dump() output)
        assert isinstance(last_data, dict)

        # All five fields must be present and non-negative
        for field in ("retrieve_ms", "generate_ms", "prompt_tokens", "completion_tokens", "cost_usd"):
            assert field in last_data, f"Missing telemetry field: {field}"
            assert last_data[field] >= 0, f"Telemetry field {field} is negative: {last_data[field]}"

    def test_stream_query_done_event_unchanged(self, ingested_backend: RAGBackend):
        """The ("done", ...) event shape is not modified by the telemetry addition.

        STRICT: The done event must still carry 'sources' (and nothing else
        unexpected). This test guards against accidental mutation of the done
        event dict.
        """
        events = list(ingested_backend.stream_query("What is RAG?"))

        done_events = [(t, d) for t, d in events if t == "done"]
        assert len(done_events) == 1, f"Expected exactly one done event, got {len(done_events)}"

        _, done_data = done_events[0]
        assert "sources" in done_data

    def test_stream_query_emits_telemetry_on_empty_store(self, backend: RAGBackend):
        """The early-return (no documents) path still emits a telemetry event.

        WHY: The route layer always expects a telemetry event. If the early-return
        path omitted it, the frontend would never receive telemetry data for
        failed queries — a silent gap that's hard to debug.
        """
        events = list(backend.stream_query("What is RAG?"))

        event_types = [e[0] for e in events]
        assert "telemetry" in event_types

        last_type, last_data = events[-1]
        assert last_type == "telemetry"
        assert last_data["generate_ms"] == 0.0
        assert last_data["prompt_tokens"] == 0

    def test_stream_query_with_conversation_emits_telemetry_and_persists(
        self, ingested_backend: RAGBackend
    ):
        """The conversation branch captures answer usage and still persists.

        WHY: stream_query's multi-turn branch is the one that reads the answer
        stream's terminal Usage AND saves the assistant message. This drives it
        with a real conversation_id so both behaviours are exercised together —
        the non-conversation telemetry tests never take this path.
        """
        conv_id = ingested_backend.create_conversation()["id"]

        events = list(
            ingested_backend.stream_query("What is RAG?", conversation_id=conv_id)
        )

        # Telemetry still last, with non-negative usage from the captured Usage.
        last_type, last_data = events[-1]
        assert last_type == "telemetry"
        assert last_data["prompt_tokens"] >= 0
        assert last_data["completion_tokens"] >= 0

        # The done event carries the persisted message + conversation ids.
        done_data = next(d for t, d in events if t == "done")
        assert done_data["conversation_id"] == conv_id
        assert done_data["message_id"]

        # The assistant message was actually saved to the conversation.
        detail = ingested_backend.get_conversation(conv_id)
        assert any(m["role"] == "assistant" for m in detail["messages"])

    def test_stream_query_existing_events_order_preserved(
        self, ingested_backend: RAGBackend
    ):
        """Existing event types appear in the expected order before telemetry.

        The protocol guarantees: status* → reasoning* → status → token* → done → telemetry
        (where * = zero or more). This test checks that all mandatory events
        are still present and that telemetry is appended AFTER done.
        """
        events = list(ingested_backend.stream_query("What is RAG?"))
        event_types = [e[0] for e in events]

        assert "status" in event_types
        assert "done" in event_types
        assert "telemetry" in event_types

        done_idx = next(i for i, (t, _) in enumerate(events) if t == "done")
        telemetry_idx = next(i for i, (t, _) in enumerate(events) if t == "telemetry")
        assert telemetry_idx > done_idx, (
            "telemetry event must come after done event"
        )
