"""LLM handler — provider selection over a set of swappable adapters.

RAG Pipeline Position:
    Query + Context -> Prompt -> [LLM HANDLER] -> Answer (+ Usage)
                                      ^^^
    LLMHandler owns provider *selection* (from the model-name prefix), the
    single-prompt -> messages translation, and the unconfigured-provider
    fallback. The actual SDK calls live in one adapter per provider (see
    ``adapters/``), each constructed with an injected client factory.

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
import os
from pathlib import Path
from typing import Iterator, List, Optional

# Load .env from the project root (two levels up from this package).
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

from .adapters.anthropic import AnthropicAdapter
from .adapters.base import (
    GenerationResult,
    ProviderAdapter,
    ProviderUnavailableError,
    Usage,
)
from .adapters.dummy import DummyAdapter
from .adapters.ollama import OllamaAdapter
from .adapters.openai_compatible import OpenAICompatibleAdapter

logger = logging.getLogger(__name__)

# Backwards-compatible alias: the error used to be module-private under this name.
_ProviderUnavailableError = ProviderUnavailableError

__all__ = [
    "LLMHandler",
    "GenerationResult",
    "ProviderAdapter",
    "ProviderUnavailableError",
    "Usage",
]

# --------------------------------------------------------------------------- #
# Provider SDK availability (checked once at import, no hard dependency)       #
# --------------------------------------------------------------------------- #

try:
    import openai as _openai_module

    _OPENAI_AVAILABLE = True
except ImportError:
    _openai_module = None  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False

try:
    import anthropic as _anthropic_module

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic_module = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False

try:
    import requests as _requests_module

    _REQUESTS_AVAILABLE = True
except ImportError:
    _requests_module = None  # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Provider constants                                                           #
# --------------------------------------------------------------------------- #

OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
ANTHROPIC_MODELS = [
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
]
OLLAMA_DEFAULT_MODELS = ["llama3", "mistral", "gemma", "phi3"]

# GLM / Zhipu AI — OpenAI-compatible endpoint, overridable via env.
GLM_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def _detect_provider(model: str) -> str:
    """Infer the provider from the model-name prefix.

    GLM (Zhipu AI) exposes an OpenAI-compatible API, so it routes to the
    OpenAI-compatible adapter with a custom base URL (see the client factory).
    """
    lower = model.lower()
    if lower.startswith("gpt") or lower.startswith("o1") or lower.startswith("o3"):
        return "openai"
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith("glm"):
        return "glm"
    return "ollama"


def _to_messages(prompt: str, system_prompt: Optional[str]) -> list[dict]:
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
        api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        """
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

        self._provider = _detect_provider(model)
        self._adapter = self._build_adapter()
        # The fallback is always ready — no client, no configuration.
        self._dummy = DummyAdapter(model)
        logger.info("LLMHandler initialised: model=%s provider=%s", model, self._provider)

    # ------------------------------------------------------------------ #
    # Adapter selection + client factories                                #
    # ------------------------------------------------------------------ #

    def _build_adapter(self) -> ProviderAdapter:
        """Select and construct the one adapter for this handler's provider."""
        if self._provider in ("openai", "glm"):
            return OpenAICompatibleAdapter(
                self.model, self.temperature, self.max_tokens, self._openai_client
            )
        if self._provider == "anthropic":
            return AnthropicAdapter(self.model, self.max_tokens, self._anthropic_client)
        # Only "ollama" remains — _detect_provider returns nothing else.
        return OllamaAdapter(
            self.model, self.temperature, self.max_tokens,
            self.ollama_base_url, self._ollama_client,
        )

    def _openai_client(self):
        """Build an OpenAI-SDK client, redirected to GLM's endpoint when needed.

        Missing SDK -> ProviderUnavailableError (falls back to dummy). Missing
        GLM key -> ProviderUnavailableError. A missing OpenAI key is left to the
        SDK, which raises its own error that propagates (unchanged behaviour).
        """
        if not _OPENAI_AVAILABLE:
            raise ProviderUnavailableError("openai package not installed")
        if self._provider == "glm":
            api_key = self.api_key or os.getenv("GLM_API_KEY")
            if not api_key:
                raise ProviderUnavailableError("GLM_API_KEY not set")
            base_url = os.getenv("GLM_BASE_URL", GLM_DEFAULT_BASE_URL)
            return _openai_module.OpenAI(api_key=api_key, base_url=base_url)
        return _openai_module.OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_KEY"))

    def _anthropic_client(self):
        """Build an Anthropic-SDK client (missing SDK -> ProviderUnavailableError)."""
        if not _ANTHROPIC_AVAILABLE:
            raise ProviderUnavailableError("anthropic package not installed")
        return _anthropic_module.Anthropic(
            api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    def _ollama_client(self):
        """Return the HTTP client for Ollama (the requests module)."""
        if not _REQUESTS_AVAILABLE:
            raise ProviderUnavailableError("requests package not installed")
        return _requests_module

    # ------------------------------------------------------------------ #
    # Public API — non-streaming                                          #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response for a single prompt."""
        return self._generate(_to_messages(prompt, system_prompt)).text

    def generate_messages(self, messages: List[dict]) -> str:
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

    def _generate(self, messages: List[dict]) -> GenerationResult:
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
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Iterator[str | Usage]:
        """Stream a response for a single prompt.

        Yields text chunks, then a terminal ``Usage``. Callers discriminate the
        terminal event with ``isinstance(item, Usage)``.
        """
        yield from self._stream(_to_messages(prompt, system_prompt))

    def stream_messages(self, messages: List[dict]) -> Iterator[str | Usage]:
        """Stream a response for a messages list.

        Yields text chunks, then a terminal ``Usage`` (see ``stream_response``).
        """
        yield from self._stream(messages)

    def _stream(self, messages: List[dict]) -> Iterator[str | Usage]:
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

    def list_models(self) -> List[str]:
        """Return available model names for the current provider."""
        if self._provider == "openai":
            return self._openai_list_models()
        if self._provider == "anthropic":
            return list(ANTHROPIC_MODELS)
        if self._provider == "ollama":
            return self._ollama_list_models()
        return ["dummy-model"]

    def _openai_list_models(self) -> List[str]:
        """Query OpenAI for GPT models, falling back to the static list."""
        if not _OPENAI_AVAILABLE:
            return list(OPENAI_MODELS)
        try:
            client = _openai_module.OpenAI(
                api_key=self.api_key or os.getenv("OPENAI_API_KEY")
            )
            models = client.models.list()
            return [m.id for m in models.data if "gpt" in m.id]
        except Exception as exc:
            logger.warning("Could not list OpenAI models: %s", exc)
            return list(OPENAI_MODELS)

    def _ollama_list_models(self) -> List[str]:
        """Query the local Ollama server for tags, falling back to defaults."""
        if not _REQUESTS_AVAILABLE:
            return list(OLLAMA_DEFAULT_MODELS)
        try:
            resp = _requests_module.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            logger.warning("Could not list Ollama models: %s", exc)
            return list(OLLAMA_DEFAULT_MODELS)
