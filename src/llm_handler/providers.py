"""Provider selection and client construction for LLMHandler.

RAG Pipeline Position:
    model name -> [PROVIDER WIRING] -> a configured ProviderAdapter

Owns: which provider a model routes to, how to build that provider's SDK client
(the injected factory), the model-listing calls, and the graceful optional-import
guards. Split out of the LLMHandler facade so each module has one responsibility
and stays under the line ceiling.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from types import ModuleType

from .adapters.anthropic import AnthropicAdapter
from .adapters.base import ProviderAdapter, ProviderUnavailableError
from .adapters.ollama import OllamaAdapter
from .adapters.openai_compatible import OpenAICompatibleAdapter

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional SDK availability (checked once at import, no hard dependency)       #
# --------------------------------------------------------------------------- #

_openai_module: ModuleType | None
try:
    import openai as _openai_module
except ImportError:
    _openai_module = None

_anthropic_module: ModuleType | None
try:
    import anthropic as _anthropic_module
except ImportError:
    _anthropic_module = None

_requests_module: ModuleType | None
try:
    import requests as _requests_module
except ImportError:
    _requests_module = None


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


def detect_provider(model: str) -> str:
    """Infer the provider from the model-name prefix.

    GLM (Zhipu AI) exposes an OpenAI-compatible API, so it routes to the
    OpenAI-compatible adapter with a custom base URL (see the client factory).

    Args:
        model: Model name.

    Returns:
        One of ``"openai"``, ``"anthropic"``, ``"glm"``, ``"ollama"``.
    """
    lower = model.lower()
    if lower.startswith(("gpt", "o1", "o3")):
        return "openai"
    if lower.startswith("claude"):
        return "anthropic"
    if lower.startswith("glm"):
        return "glm"
    return "ollama"


def build_adapter(
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str | None,
    ollama_base_url: str,
) -> ProviderAdapter:
    """Select and construct the one adapter for a model's provider.

    Each adapter is given an injected zero-arg client factory (built here) so the
    SDK client is constructed per request and can be faked in tests.

    Args:
        model: Model name (also decides the provider and constrained-model quirks).
        temperature: Sampling temperature.
        max_tokens: Output token cap.
        api_key: Explicit key, or None to read the provider's env var.
        ollama_base_url: Base URL for a local Ollama server.

    Returns:
        A ready-to-use ``ProviderAdapter``.
    """
    provider = detect_provider(model)
    if provider in ("openai", "glm"):
        return OpenAICompatibleAdapter(
            model, temperature, max_tokens, _openai_client_factory(provider, api_key)
        )
    if provider == "anthropic":
        return AnthropicAdapter(model, max_tokens, _anthropic_client_factory(api_key))
    return OllamaAdapter(
        model, temperature, max_tokens, ollama_base_url, _ollama_client_factory()
    )


# Environment variable names, spelled once. Provider credentials are resolved
# lazily at client-construction time rather than at import, so a missing key
# only matters when that provider is actually selected.
API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "glm": "GLM_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def resolve_api_key(provider: str, api_key: str | None = None) -> str | None:
    """Return the API key for a provider: explicit argument, else environment.

    Args:
        provider: One of ``openai``, ``glm``, ``anthropic``.
        api_key: An explicitly supplied key, which always wins.

    Returns:
        The key, or None when neither source has one.
    """
    if api_key:
        return api_key
    env_name = API_KEY_ENV.get(provider)
    return os.getenv(env_name) if env_name else None


def _openai_client_factory(provider: str, api_key: str | None) -> Callable[[], object]:
    """Build a factory for an OpenAI-SDK client, redirected to GLM when needed.

    Missing SDK -> ProviderUnavailableError (dummy fallback). Missing GLM key ->
    ProviderUnavailableError. A missing OpenAI key is left to the SDK, which
    raises its own error that propagates (unchanged behaviour).
    """
    def factory() -> object:
        if _openai_module is None:
            raise ProviderUnavailableError("openai package not installed")
        if provider == "glm":
            key = resolve_api_key("glm", api_key)
            if not key:
                raise ProviderUnavailableError("GLM_API_KEY not set")
            base_url = os.getenv("GLM_BASE_URL", GLM_DEFAULT_BASE_URL)
            return _openai_module.OpenAI(api_key=key, base_url=base_url)
        return _openai_module.OpenAI(api_key=resolve_api_key("openai", api_key))

    return factory


def _anthropic_client_factory(api_key: str | None) -> Callable[[], object]:
    """Build a factory for an Anthropic-SDK client (missing SDK -> unavailable)."""
    def factory() -> object:
        if _anthropic_module is None:
            raise ProviderUnavailableError("anthropic package not installed")
        return _anthropic_module.Anthropic(
            api_key=resolve_api_key("anthropic", api_key)
        )

    return factory


def _ollama_client_factory() -> Callable[[], object]:
    """Build a factory returning the HTTP client for Ollama (the requests module)."""
    def factory() -> object:
        if _requests_module is None:
            raise ProviderUnavailableError("requests package not installed")
        return _requests_module

    return factory


def list_models(
    provider: str, model: str, api_key: str | None, ollama_base_url: str
) -> list[str]:
    """Return available model names for a provider (out of scope; preserved)."""
    if provider == "openai":
        return _openai_list_models(api_key)
    if provider == "anthropic":
        return list(ANTHROPIC_MODELS)
    if provider == "ollama":
        return _ollama_list_models(ollama_base_url)
    return ["dummy-model"]


def _openai_list_models(api_key: str | None) -> list[str]:
    """Query OpenAI for GPT models, falling back to the static list."""
    if _openai_module is None:
        return list(OPENAI_MODELS)
    try:
        client = _openai_module.OpenAI(api_key=resolve_api_key("openai", api_key))
        models = client.models.list()
        return [m.id for m in models.data if "gpt" in m.id]
    except Exception as exc:
        logger.warning("Could not list OpenAI models: %s", exc)
        return list(OPENAI_MODELS)


def _ollama_list_models(ollama_base_url: str) -> list[str]:
    """Query the local Ollama server for tags, falling back to defaults."""
    if _requests_module is None:
        return list(OLLAMA_DEFAULT_MODELS)
    try:
        resp = _requests_module.get(f"{ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:
        logger.warning("Could not list Ollama models: %s", exc)
        return list(OLLAMA_DEFAULT_MODELS)
