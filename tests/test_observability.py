"""Tests for src.observability."""

from __future__ import annotations

from src.observability import TRACER_NAME, get_tracer, init_observability


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
        # Subsequent spans must still work (as no-ops or local).
        with get_tracer().start_as_current_span("test.bad-endpoint") as span:
            span.set_attribute("k", 1)


class TestGetTracer:
    def test_returns_tracer(self):
        t = get_tracer()
        assert t is not None
