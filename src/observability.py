"""
Observability — OpenTelemetry tracer initialization and per-stage
span decorator.

API Layer Position:
  RAGBackend.query → @traced_stage("rag.retrieve") → span exported via OTLP
                                                   → Phoenix UI at :6006

Design decisions:
  - Idempotent init via a module-level flag — multiple imports / lifespan
    callbacks won't double-install processors.
  - Fail QUIETLY on exporter / endpoint errors. Eval and chat must work
    when Phoenix is down; observability is opt-in.
  - traced_stage uses the (payload, attrs) return convention so existing
    return shapes don't change at call sites; the decorator strips attrs.
  - Attribute coercion: OTel restricts attribute types. Non-primitives get
    str()-coerced rather than dropped (visibility > strictness).
"""

from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

TRACER_NAME = "rag-qa"
_INITIALIZED = False

# OTel primitive types that span.set_attribute accepts as scalars.
_OTEL_SCALARS = (str, int, float, bool)


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


def _coerce_attr(value: Any) -> Any:
    """Coerce a value to an OTel-legal span attribute type.

    OTel accepts: str, int, float, bool, or homogeneous Sequence thereof.
    Everything else is str()-converted so we retain visibility over dropping.

    Args:
        value: Raw attribute value from the decorated function.

    Returns:
        A value safe to pass to ``span.set_attribute``.
    """
    if isinstance(value, _OTEL_SCALARS):
        return value

    # Sequences: pass through only when every element is a scalar of the
    # same OTel-legal type.  Mixed or complex elements fall back to str().
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(el, _OTEL_SCALARS) for el in value):
            # OTel SDK coerces list → tuple internally; list is fine here.
            return value
        # Empty or mixed-type list — convert whole thing.
        return str(value)

    # Dicts, None, and anything else: str-coerce for visibility.
    return str(value)


def traced_stage(name: str):
    """Decorator that opens a span around the wrapped function.

    The wrapped function MUST return ``(payload, attrs_dict)``.  The
    decorator opens a span named ``name``, calls the function, records
    each ``attrs_dict`` entry as a span attribute, then returns only
    ``payload`` — callers see the same shape as before instrumentation.

    On exception the span is marked ERROR and the exception is re-raised.

    Args:
        name: Span name (e.g. ``"rag.retrieve"``).

    Returns:
        A decorator that wraps ``f(*args, **kwargs) -> tuple[Any, dict]``
        and exposes only the payload to callers.

    Example::

        @traced_stage("rag.retrieve")
        def retrieve(query: str):
            chunks = _do_search(query)
            return chunks, {"chunk_count": len(chunks)}

        result = retrieve("what is RAG?")  # returns chunks directly
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            with tracer.start_as_current_span(name) as span:
                try:
                    payload, attrs = f(*args, **kwargs)
                except Exception as exc:
                    # PATTERN: Record exception details on the span so
                    # Phoenix shows the stack trace, then propagate.
                    from opentelemetry.trace import Status, StatusCode
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR))
                    raise

                # Set attributes after successful return.
                for key, raw_val in attrs.items():
                    coerced = _coerce_attr(raw_val)
                    try:
                        span.set_attribute(key, coerced)
                    except Exception:  # noqa: BLE001
                        logger.warning(
                            "traced_stage: could not set attribute %r=%r, skipping",
                            key,
                            coerced,
                        )

                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.OK))

            return payload

        return wrapper

    return decorator
