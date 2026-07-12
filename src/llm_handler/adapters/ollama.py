"""Ollama provider adapter — local models over HTTP.

RAG Pipeline Position:
    Messages -> [OLLAMA ADAPTER] -> Answer + Usage

Design Decision:
    Ollama is a local server, so a refused connection or an unknown model is
    treated as "unconfigured" (``ProviderUnavailableError``) — the same bucket as
    a missing SDK — letting LLMHandler fall back to the dummy rather than 500.

    This adapter speaks only ``/api/chat`` (the messages-native endpoint). The
    older single-prompt ``/api/generate`` path is gone: LLMHandler now translates
    every single-prompt call into a messages list before it reaches an adapter,
    so one endpoint covers both.

    Usage comes from Ollama's ``prompt_eval_count`` / ``eval_count`` fields (on
    the response, or the terminal streamed line), with a local count as fallback.
"""

from __future__ import annotations

import json
from typing import Callable, Iterator

from .base import (
    GenerationResult,
    ProviderUnavailableError,
    Usage,
    counted_usage,
    join_message_text,
)

# Requests errors (connection refused, timeout, HTTP error) all subclass
# RequestException. Catching it narrowly wraps a down/unknown local server
# without swallowing genuine bugs.
_REQUEST_EXCEPTION: type[Exception] | tuple[type[Exception], ...]
try:
    import requests

    _REQUEST_EXCEPTION = requests.exceptions.RequestException
except ImportError:  # pragma: no cover - requests is a hard dependency
    # If requests is missing, LLMHandler's factory raises ProviderUnavailableError
    # before this adapter ever posts, so this except clause never fires.
    _REQUEST_EXCEPTION = ()


class OllamaAdapter:
    """Messages adapter for a local Ollama server."""

    def __init__(
        self,
        model: str,
        temperature: float,
        max_tokens: int,
        base_url: str,
        client_factory: Callable[[], object],
    ) -> None:
        """Store generation settings, the server URL, and the injected factory.

        Args:
            model: Ollama model tag (e.g. ``llama3``).
            temperature: Sampling temperature, sent under ``options``.
            max_tokens: Output cap, sent as ``options.num_predict``.
            base_url: Ollama server base URL (no trailing slash).
            client_factory: Zero-arg builder for an HTTP client exposing
                ``post`` (the ``requests`` module by default), called per request.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._client_factory = client_factory

    def _payload(self, messages: list[dict], *, stream: bool) -> dict[str, object]:
        """Build the /api/chat request body."""
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

    def generate(self, messages: list[dict], **kwargs: object) -> GenerationResult:
        """POST to /api/chat and return content plus usage."""
        http = self._client_factory()
        try:
            resp = http.post(
                f"{self.base_url}/api/chat",
                json=self._payload(messages, stream=False),
                timeout=120,
            )
            resp.raise_for_status()
        except _REQUEST_EXCEPTION as exc:
            raise _unavailable(exc) from exc

        data = resp.json()
        text = data.get("message", {}).get("content", "")
        usage = _usage_from(data)
        if usage is None:
            usage = counted_usage(join_message_text(messages), text, self.model)
        return GenerationResult(text=text, usage=usage)

    def stream(self, messages: list[dict], **kwargs: object) -> Iterator[str | Usage]:
        """Stream tokens from /api/chat, then a terminal Usage."""
        http = self._client_factory()
        collected: list[str] = []
        reported: Usage | None = None
        try:
            with http.post(
                f"{self.base_url}/api/chat",
                json=self._payload(messages, stream=True),
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        collected.append(token)
                        yield token
                    if data.get("done"):
                        reported = _usage_from(data)
                        break
        except _REQUEST_EXCEPTION as exc:
            raise _unavailable(exc) from exc

        yield reported or counted_usage(
            join_message_text(messages), "".join(collected), self.model
        )


def _usage_from(data: dict) -> Usage | None:
    """Read Ollama's token counts, or None when the line omits them."""
    if "prompt_eval_count" not in data and "eval_count" not in data:
        return None
    return Usage(
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
    )


def _unavailable(exc: Exception) -> ProviderUnavailableError:
    """Wrap a requests failure as a provider-unavailable signal."""
    return ProviderUnavailableError(f"Ollama unavailable: {exc}")
