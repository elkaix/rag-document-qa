"""Core telemetry — token counting and cost pricing.

RAG Pipeline Position:
    Generation (reported usage) -> [TELEMETRY] -> StageTelemetry / EvalResult

Owned by the core so both the production telemetry assembly and the eval harness
depend on one source of truth. Production no longer imports these from the eval
package (the dependency direction the architecture-deepening spec inverts).

See [ADR 0003](../../docs/adr/0003-telemetry-ownership.md).
"""

from __future__ import annotations

from src.telemetry.pricing import MODEL_PRICES, ModelPrice, cost_usd
from src.telemetry.tokens import count_tokens

__all__ = ["MODEL_PRICES", "ModelPrice", "cost_usd", "count_tokens"]
