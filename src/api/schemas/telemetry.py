"""
StageTelemetry — per-stage timing, token, and cost numbers returned to the
chat client alongside the answer.

API Layer Position:
  RAGBackend.query → returns (answer, sources, StageTelemetry)
  /api/query response includes StageTelemetry as `telemetry` field
  WebSocket emits a final `telemetry` event with this payload after `done`

Frontend renders these as a muted footer under each assistant chat bubble:
  "Retrieve 142ms · Generate 2.1s · 4,217 tok · $0.0083"
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StageTelemetry(BaseModel):
    """Per-stage observability numbers for one chat turn."""

    retrieve_ms: float = Field(ge=0.0)
    generate_ms: float = Field(ge=0.0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
