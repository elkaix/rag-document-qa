"""LLM handler — provider selection over a set of swappable adapters.

RAG Pipeline Position:
    Query + Context -> Prompt -> [LLM HANDLER] -> Answer (+ Usage)
                                      ^^^
    LLMHandler owns the single-prompt -> messages translation and the
    unconfigured-provider fallback. Provider selection and SDK client
    construction live in ``providers`` (via ``build_adapter``); the actual SDK
    calls live in one adapter per provider (see ``adapters/``).

Design Decision:
    Before this refactor a single class repeated the same provider dispatch four
    times and built its clients from module globals, so two of three real
    providers had no test coverage. Now one adapter is selected at construction
    and every provider is unit-testable with a fake client (see
    ``tests/test_llm_adapters.py``).

    The public interface (generate / stream / *_messages / generate_with_usage /
    list_models) is unchanged for callers. Prompt variants are thin translations
    to the messages form the adapters speak.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

# Load .env from the project root (two levels up from this package).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

from .adapters.base import (
    GenerationResult,
    ProviderAdapter,
    ProviderUnavailableError,
    Usage,
)
from .adapters.dummy import DummyAdapter
from .providers import build_adapter, detect_provider, list_models

logger = logging.getLogger(__name__)

__all__ = [
    "LLMHandler",
    "GenerationResult",
    "ProviderAdapter",
    "ProviderUnavailableError",
    "Usage",
]


def _to_messages(prompt: str, system_prompt: str | None) -> list[dict]:
    """Translate a single prompt (+ optional system) into a messages list."""
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


class LLMHandler:
    """Unified LLM interface backed by one provider adapter per instance.

    Provider selection is automatic from the model prefix (gpt/o1/o3 -> OpenAI,
    claude -> Anthropic, glm -> GLM, else Ollama). If the selected provider is
    unconfigured, calls fall back to a dummy response so the demo stays usable.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        # WHY 4096: 1024 truncates detailed Q&A answers mid-sentence; 4096 gives
        #           room for thorough, formatted answers while bounding cost.
        max_tokens: int = 4096,
        api_key: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        """Configure the handler and select its provider adapter.

        Args:
            model: Model name (determines provider).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            api_key: API key (falls back to provider-specific env vars).
            ollama_base_url: Base URL for a local Ollama server.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.ollama_base_url = ollama_base_url.rstrip("/")

        self._provider = detect_provider(model)
        self._adapter = build_adapter(
            model, temperature, max_tokens, api_key, self.ollama_base_url
        )
        # The fallback is always ready — no client, no configuration.
        self._dummy = DummyAdapter(model)
        logger.info("LLMHandler initialised: model=%s provider=%s", model, self._provider)

    # ------------------------------------------------------------------ #
    # Public API — non-streaming                                          #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate a response for a single prompt."""
        return self._generate(_to_messages(prompt, system_prompt)).text

    def generate_messages(self, messages: list[dict]) -> str:
        """Generate a response for a full OpenAI-style messages list."""
        return self._generate(messages).text

    def generate_with_usage(
        self, prompt: str, system_prompt: str | None = None
    ) -> tuple[str, int, int]:
        """Generate a response and return (text, prompt_tokens, completion_tokens).

        Usage is provider-reported where available, adapter-counted otherwise —
        callers no longer rebuild the prompt just to estimate token counts.
        """
        result = self._generate(_to_messages(prompt, system_prompt))
        return result.text, result.usage.prompt_tokens, result.usage.completion_tokens

    def _generate(self, messages: list[dict]) -> GenerationResult:
        """Call the selected adapter, falling back to dummy if unconfigured."""
        try:
            return self._adapter.generate(messages)
        except ProviderUnavailableError as exc:
            logger.warning("Provider unavailable, using dummy: %s", exc)
            return self._dummy.generate(messages)

    # ------------------------------------------------------------------ #
    # Public API — streaming                                              #
    # ------------------------------------------------------------------ #

    def stream_response(
        self, prompt: str, system_prompt: str | None = None
    ) -> Iterator[str | Usage]:
        """Stream a response for a single prompt.

        Yields text chunks, then a terminal ``Usage``. Callers discriminate the
        terminal event with ``isinstance(item, Usage)``.
        """
        yield from self._stream(_to_messages(prompt, system_prompt))

    def stream_messages(self, messages: list[dict]) -> Iterator[str | Usage]:
        """Stream a response for a messages list.

        Yields text chunks, then a terminal ``Usage`` (see ``stream_response``).
        """
        yield from self._stream(messages)

    def _stream(self, messages: list[dict]) -> Iterator[str | Usage]:
        """Stream text chunks then a terminal Usage from the adapter.

        Falls back to the dummy adapter (which reports counted usage) when the
        provider is unconfigured. The terminal ``Usage`` flows through to the
        backend's telemetry assembly.
        """
        try:
            yield from self._adapter.stream(messages)
        except ProviderUnavailableError as exc:
            logger.warning("Provider unavailable, using dummy stream: %s", exc)
            yield from self._dummy.stream(messages)

    # ------------------------------------------------------------------ #
    # Model listing (out of scope for the adapter refactor — preserved)   #
    # ------------------------------------------------------------------ #

    def list_models(self) -> list[str]:
        """Return available model names for the current provider."""
        return list_models(
            self._provider, self.model, self.api_key, self.ollama_base_url
        )
