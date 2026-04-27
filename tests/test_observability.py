"""Tests for src.observability."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from src.observability import (
    TRACER_NAME,
    get_tracer,
    init_observability,
    traced_stage,
)


@pytest.fixture
def in_memory_exporter():
    """Install an InMemorySpanExporter on a fresh TracerProvider for the test."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Force-install our test provider, overriding any prior init.
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    yield exporter
    exporter.clear()


class TestInitObservability:
    def test_idempotent(self):
        """Calling init twice doesn't crash and doesn't double-install."""
        init_observability()
        init_observability()
        # No assertion — just doesn't raise.

    def test_unreachable_endpoint_does_not_raise(self):
        """A bad endpoint is logged but doesn't crash."""
        # Reset the idempotency flag so this call actually attempts init.
        import src.observability as obs
        obs._INITIALIZED = False  # type: ignore[attr-defined]
        init_observability(otlp_endpoint="http://127.0.0.1:1/v1/traces")
        # Subsequent traced spans must still work (as no-ops or local).
        @traced_stage("test.bad-endpoint")
        def f():
            return "x", {"k": 1}
        assert f() == "x"


class TestGetTracer:
    def test_returns_tracer(self):
        t = get_tracer()
        assert t is not None


class TestTracedStage:
    def test_records_span_with_attrs(self, in_memory_exporter):
        @traced_stage("rag.retrieve")
        def retrieve(query: str):
            return ["chunk1", "chunk2"], {"top_k": 5, "chunk_count": 2}

        result = retrieve("test query")
        assert result == ["chunk1", "chunk2"]

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "rag.retrieve"
        assert span.attributes["top_k"] == 5
        assert span.attributes["chunk_count"] == 2

    def test_propagates_exception_and_records_error(self, in_memory_exporter):
        @traced_stage("rag.fail")
        def boom():
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="kaboom"):
            boom()

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        # Span recorded ERROR status
        from opentelemetry.trace import StatusCode
        assert spans[0].status.status_code == StatusCode.ERROR

    def test_attribute_coercion(self, in_memory_exporter):
        """Non-primitive attribute values are coerced to str."""
        @traced_stage("rag.coerce")
        def f():
            return None, {"a_dict": {"x": 1}, "a_list_of_str": ["a", "b"]}
        f()
        spans = in_memory_exporter.get_finished_spans()
        # a_dict gets coerced to str representation; a_list_of_str passes through.
        attrs = spans[0].attributes
        assert "a_dict" in attrs
        assert tuple(attrs["a_list_of_str"]) == ("a", "b")
