"""
Observability — OpenTelemetry tracer initialization.

API Layer Position:
  RAGBackend.query → tracer.start_as_current_span("rag.retrieve") → OTLP
                                                   → Phoenix UI at :6006

Design decisions:
  - Idempotent init via a module-level flag — multiple imports / lifespan
    callbacks won't double-install processors.
  - Fail QUIETLY on exporter / endpoint errors. Eval and chat must work
    when Phoenix is down; observability is opt-in.
  - Spans are opened inline at each call site (get_tracer().start_as_current_span(...))
    rather than via a decorator — RAGBackend's stages are one-liners with
    per-call attributes, so inline spans are shorter than a wrapper.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

TRACER_NAME = "rag-qa"
_INITIALIZED = False


def init_observability(otlp_endpoint: str | None = None) -> None:
    """Initialize global TracerProvider + OTLP exporter.

    Idempotent — safe to call multiple times.
    ``otlp_endpoint`` defaults to the ``OTLP_ENDPOINT`` env var, or
    ``http://localhost:6006/v1/traces`` when neither is set.

    On import errors or connection failures, fails QUIETLY (logs warning,
    spans become no-ops). The system never crashes due to OTel.

    Args:
        otlp_endpoint: OTLP HTTP traces endpoint URL.  Pass ``None`` to
            use the default derived from the environment variable.
    """
    # PATTERN: module-level flag makes this idempotent — calling from
    # multiple lifespan callbacks or test setups is safe.
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    endpoint = otlp_endpoint or os.getenv(
        "OTLP_ENDPOINT", "http://localhost:6006/v1/traces"
    )

    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        import opentelemetry.trace as otel_trace

        # WHY: OTLPSpanExporter is lazy — it won't attempt a connection
        # until the first batch is flushed, so construction never raises
        # even if the endpoint is unreachable.
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)

    except Exception as exc:  # noqa: BLE001
        # TRADE-OFF: We catch broadly here because we never want Phoenix
        # being unavailable to crash the RAG service.  A warning is enough.
        logger.warning(
            "Observability init failed — spans will be no-ops. Reason: %s", exc
        )


def get_tracer():
    """Return the rag-qa tracer (always works; no-op if init never called).

    Returns:
        An ``opentelemetry.trace.Tracer`` instance bound to TRACER_NAME.
        When no provider has been installed this returns the global no-op
        tracer, so call sites never need to guard for None.
    """
    import opentelemetry.trace as otel_trace

    # WHY: get_tracer() delegates to whatever TracerProvider is currently
    # registered globally.  If init_observability was never called that's
    # the SDK default (no-op), which is perfectly fine.
    return otel_trace.get_tracer(TRACER_NAME)
