"""Streaming support for the QueryEngine — the terminal payload and pure helpers.

The engine's `ask_stream` yields display events (status/reasoning/token) plus one
terminal `("result", StreamResult)` so the facade can persist the turn and emit
its own done/telemetry events without the engine knowing about conversations.
This module holds that payload type, the event alias, and the small pure helpers
the streaming path uses. (Step 6 of issue #16 will add the shared streaming event
vocabulary here.)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.api.schemas.telemetry import StageTelemetry
from src.query_engine.prompt import ANSWER_SYSTEM_PROMPT, build_answer_user_prompt
from src.vector_store import SearchResult


@dataclass(frozen=True)
class StreamResult:
    """The terminal payload of `ask_stream`, carried on the ("result", ...) event."""

    results: list[SearchResult]
    telemetry: StageTelemetry
    model: str


# One streamed event: a label plus either a display string or the terminal result.
StreamEvent = tuple[str, "str | StreamResult"]


def retrieval_summary(results: list[SearchResult]) -> str:
    """Return a one-liner naming the files that contributed to the retrieval."""
    unique_files = sorted(
        {r.metadata.get("filename", "unknown") for r in results if r.metadata.get("filename")}
    )
    summary = ", ".join(unique_files[:3])
    if len(unique_files) > 3:
        summary += f" (+{len(unique_files) - 3} more)"
    return f"Retrieved {len(results)} chunk(s) across {len(unique_files)} file(s): {summary}"


def build_answer_messages(
    history: list[dict[str, str]], context: str, question: str,
) -> list[dict[str, str]]:
    """Build the multi-turn messages list: system + prior turns + current user."""
    messages: list[dict[str, str]] = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": build_answer_user_prompt(context, question)})
    return messages
