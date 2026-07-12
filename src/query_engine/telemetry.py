"""Telemetry assembly — turn a generation's Usage into a StageTelemetry payload.

RAG Pipeline Position:
    (retrieve_ms, generate_ms, model, Usage) -> [TELEMETRY] -> StageTelemetry

Design Decision:
    Before step 4 the StageTelemetry construction was duplicated across four call
    sites in the backend. The QueryEngine assembles it once, here, from the
    provider-reported ``Usage`` (ADR 0003) and the core pricing table (ADR 0003)
    — never from a reconstructed prompt.
"""

from __future__ import annotations

from src.api.schemas.telemetry import StageTelemetry
from src.llm_handler import Usage
from src.telemetry.pricing import cost_usd


def assemble(
    retrieve_ms: float, generate_ms: float, model: str, usage: Usage,
) -> StageTelemetry:
    """Build a StageTelemetry from stage timings and provider-reported usage.

    Args:
        retrieve_ms: Retrieval wall time in milliseconds.
        generate_ms: Generation wall time in milliseconds.
        model: The model whose pricing applies to ``usage``.
        usage: Provider-reported (or adapter-counted) token counts.

    Returns:
        StageTelemetry with cost priced from ``model`` and ``usage``.
    """
    return StageTelemetry(
        retrieve_ms=round(retrieve_ms, 2),
        generate_ms=round(generate_ms, 2),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cost_usd=cost_usd(model, usage.prompt_tokens, usage.completion_tokens),
    )


def zero(retrieve_ms: float) -> StageTelemetry:
    """Telemetry for a path that never called the LLM (no docs, or refusal).

    Retrieval time is real; every generation field is zero.
    """
    return StageTelemetry(
        retrieve_ms=round(retrieve_ms, 2),
        generate_ms=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        cost_usd=0.0,
    )
