"""OpenAI-compatible provider adapter (serves both OpenAI and GLM).

RAG Pipeline Position:
    Messages -> [OPENAI-COMPATIBLE ADAPTER] -> Answer + Usage

Design Decision:
    OpenAI and Zhipu's GLM speak the same chat-completions wire protocol, so one
    adapter serves both. The only difference is the client's base URL and key,
    which LLMHandler bakes into the injected ``client_factory`` — so GLM support
    is a factory concern, not a second code path here.

    The token-parameter and temperature quirks DO live here:
      - Always send ``max_completion_tokens`` (the gpt-5 / o-series families
        require it; modern gpt-4* accept it).
      - Omit ``temperature`` for constrained models (gpt-5*, o1*, o3*, o4*) —
        they reject any non-default value with a 400.

    Usage is provider-reported (``response.usage`` / the stream's terminal usage
    chunk requested via ``stream_options``) with a local count as fallback.
"""

from __future__ import annotations

from typing import Callable, Iterator

from .base import (
    GenerationResult,
    Usage,
    counted_usage,
    join_message_text,
)


def _is_constrained(model: str) -> bool:
    """Return True if the model rejects a ``temperature`` override.

    WHY these prefixes: the GPT-5 family and the o-series reasoning models only
    accept the default temperature; older gpt-4* families accept the full range.
    """
    lower = model.lower()
    return (
        lower.startswith("gpt-5")
        or lower.startswith("o1")
        or lower.startswith("o3")
        or lower.startswith("o4")
    )


class OpenAICompatibleAdapter:
    """Chat-completions adapter for OpenAI and any OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        client_factory: Callable[[], object],
    ) -> None:
        """Store generation settings and the injected client factory.

        Args:
            model: Model name (also decides the constrained-model quirks).
            temperature: Sampling temperature (dropped for constrained models).
            max_tokens: Output cap, sent as ``max_completion_tokens``.
            client_factory: Zero-arg builder for an OpenAI-SDK-shaped client.
                Called once per request (no caching) so a rotated key or a test
                fake is always picked up.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client_factory = client_factory

    def _request_kwargs(self, **extra: object) -> dict[str, object]:
        """Assemble the create() kwargs shared by generate() and stream()."""
        kwargs: dict[str, object] = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            **extra,
        }
        if not _is_constrained(self.model):
            kwargs["temperature"] = self.temperature
        return kwargs

    def generate(self, messages: list[dict], **kwargs: object) -> GenerationResult:
        """Call chat.completions.create and return text plus usage."""
        client = self._client_factory()
        response = client.chat.completions.create(
            **self._request_kwargs(messages=messages)
        )
        text = response.choices[0].message.content or ""
        usage = _usage_from_response(response)
        if usage is None:
            usage = counted_usage(join_message_text(messages), text, self.model)
        return GenerationResult(text=text, usage=usage)

    def stream(self, messages: list[dict], **kwargs: object) -> Iterator[str | Usage]:
        """Stream tokens, then a terminal Usage (reported or counted)."""
        client = self._client_factory()
        stream = client.chat.completions.create(
            **self._request_kwargs(
                messages=messages,
                stream=True,
                # WHY: without include_usage the stream carries no token counts,
                #      forcing the local-count fallback. Requesting it lets the
                #      provider's own numbers flow through.
                stream_options={"include_usage": True},
            )
        )
        collected: list[str] = []
        reported: Usage | None = None
        for chunk in stream:
            reported = _usage_from_response(chunk) or reported
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                collected.append(delta.content)
                yield delta.content
        yield reported or counted_usage(
            join_message_text(messages), "".join(collected), self.model
        )


def _usage_from_response(response: object) -> Usage | None:
    """Extract OpenAI-style usage, or None when the provider omits it."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
