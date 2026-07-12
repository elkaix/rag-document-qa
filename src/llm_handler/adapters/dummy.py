"""Dummy fallback adapter — the answer when no provider is configured.

RAG Pipeline Position:
    ... -> [DUMMY ADAPTER] -> placeholder answer
    LLMHandler routes here when the selected provider raises
    ``ProviderUnavailableError`` (missing SDK, missing key, local server down),
    so the demo stays usable — the retrieval half still returns real sources.

Design Decision:
    Usage is always the local-count fallback: there is no provider to report it.
"""

from __future__ import annotations

from typing import Iterator

from .base import (
    GenerationResult,
    Usage,
    counted_usage,
    join_message_text,
)


class DummyAdapter:
    """Return a safe placeholder response for any messages list."""

    def __init__(self, model: str) -> None:
        """Store the model name (used only to pick a tokenizer for usage counts)."""
        self.model = model

    def generate(self, messages: list[dict], **kwargs: object) -> GenerationResult:
        """Return the placeholder answer with locally-counted usage."""
        text = self._placeholder(messages)
        usage = counted_usage(join_message_text(messages), text, self.model)
        return GenerationResult(text=text, usage=usage)

    def stream(self, messages: list[dict], **kwargs: object) -> Iterator[str | Usage]:
        """Stream the placeholder word by word, then a terminal Usage."""
        text = self._placeholder(messages)
        for word in text.split():
            yield word + " "
        yield counted_usage(join_message_text(messages), text, self.model)

    @staticmethod
    def _placeholder(messages: list[dict]) -> str:
        """Build the sentinel string, sized from the last user turn."""
        last_user_content = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                last_user_content = message.get("content", "")
                break
        return (
            "[LLM unavailable] This is a placeholder response. "
            f"Your last message was {len(last_user_content)} characters "
            f"(conversation has {len(messages)} turn(s)). "
            "Please configure a valid LLM provider (OpenAI, Anthropic, or Ollama)."
        )
