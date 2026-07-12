"""Anthropic provider adapter.

RAG Pipeline Position:
    Messages -> [ANTHROPIC ADAPTER] -> Answer + Usage

Design Decision:
    Anthropic's API does not accept ``system`` role entries inside the messages
    list — the system prompt is a separate top-level ``system=`` argument. This
    adapter owns that split so no caller has to know the quirk. Multiple system
    messages are joined with newlines (well-formed histories have at most one).

    Usage maps Anthropic's ``input_tokens`` / ``output_tokens`` onto the shared
    ``Usage`` prompt/completion fields, with a local count as fallback.
"""

from __future__ import annotations

from typing import Callable, Iterator

from .base import (
    GenerationResult,
    Usage,
    counted_usage,
    join_message_text,
)


class AnthropicAdapter:
    """Messages adapter for Anthropic's Claude models."""

    def __init__(
        self,
        model: str,
        max_tokens: int,
        client_factory: Callable[[], object],
    ) -> None:
        """Store generation settings and the injected client factory.

        Args:
            model: Claude model name.
            max_tokens: Output token cap.
            client_factory: Zero-arg builder for an Anthropic-SDK-shaped client,
                called once per request.

        Note:
            No ``temperature`` — the previous implementation never sent one to
            Anthropic, and this adapter preserves that exactly.
        """
        self.model = model
        self.max_tokens = max_tokens
        self._client_factory = client_factory

    def _request_kwargs(self, messages: list[dict]) -> dict[str, object]:
        """Split system messages out and build the create()/stream() kwargs."""
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": non_system,
        }
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)
        return kwargs

    def generate(self, messages: list[dict], **kwargs: object) -> GenerationResult:
        """Call messages.create and return text plus usage."""
        client = self._client_factory()
        response = client.messages.create(**self._request_kwargs(messages))
        block = response.content[0]
        text = block.text if hasattr(block, "text") else str(block)
        usage = _usage_from(response)
        if usage is None:
            usage = counted_usage(join_message_text(messages), text, self.model)
        return GenerationResult(text=text, usage=usage)

    def stream(self, messages: list[dict], **kwargs: object) -> Iterator[str | Usage]:
        """Stream tokens, then a terminal Usage (reported or counted)."""
        client = self._client_factory()
        collected: list[str] = []
        with client.messages.stream(**self._request_kwargs(messages)) as stream:
            for text in stream.text_stream:
                collected.append(text)
                yield text
            usage = _usage_from(stream.get_final_message())
        yield usage or counted_usage(
            join_message_text(messages), "".join(collected), self.model
        )


def _usage_from(message: object) -> Usage | None:
    """Map Anthropic's input/output token usage, or None when absent."""
    usage = getattr(message, "usage", None)
    if usage is None:
        return None
    return Usage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
    )
