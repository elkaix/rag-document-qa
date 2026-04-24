"""
LLM handler module for RAG pipeline.

Supports OpenAI, Anthropic, and Ollama providers with graceful fallback
to a dummy response generator when providers are unavailable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Generator, List, Optional

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)


# BUG FIX: `generate()` and `stream_response()` used to swallow every exception
# and return a dummy response. That turned auth failures, quota exhaustion,
# and network outages into HTTP 200 "Your prompt was N characters" answers —
# dangerous in production. The fallback now fires *only* when the provider is
# genuinely unconfigured (module missing, api key missing). Real API errors
# propagate to the caller so the API returns 5xx.
class _ProviderUnavailableError(RuntimeError):
    """Raised when a provider module is not installed or has no API key."""

# --------------------------------------------------------------------------- #
# Provider detection at import time (no hard dependency)                      #
# --------------------------------------------------------------------------- #

try:
    import openai as _openai_module

    _OPENAI_AVAILABLE = True
except ImportError:
    _openai_module = None  # type: ignore
    _OPENAI_AVAILABLE = False

try:
    import anthropic as _anthropic_module

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic_module = None  # type: ignore
    _ANTHROPIC_AVAILABLE = False

try:
    import requests as _requests_module

    _REQUESTS_AVAILABLE = True
except ImportError:
    _requests_module = None  # type: ignore
    _REQUESTS_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Provider constants                                                           #
# --------------------------------------------------------------------------- #

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]

ANTHROPIC_MODELS = [
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
]

OLLAMA_DEFAULT_MODELS = [
    "llama3",
    "mistral",
    "gemma",
    "phi3",
]


class LLMHandler:
    """Unified LLM interface supporting OpenAI, Anthropic, and Ollama.

    Provider selection is automatic based on the model prefix:
    - Models starting with 'gpt' / 'o1' / 'o3' → OpenAI
    - Models starting with 'claude' → Anthropic
    - Others → Ollama (localhost)

    Falls back to a dummy response if the selected provider is unavailable.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        # WHY: 1024 tokens is too low for detailed Q&A responses — the LLM
        #      hits the limit mid-sentence and truncates the answer.
        #      4096 gives enough room for thorough, formatted answers while
        #      still bounding cost and latency.
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        """
        Args:
            model: Model name (determines provider).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            api_key: API key (falls back to env vars OPENAI_API_KEY / ANTHROPIC_API_KEY).
            ollama_base_url: Base URL for Ollama server.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.ollama_base_url = ollama_base_url.rstrip("/")

        self._provider = self._detect_provider(model)
        logger.info("LLMHandler initialised: model=%s provider=%s", model, self._provider)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a response for a prompt.

        Args:
            prompt: User message.
            system_prompt: Optional system instructions.

        Returns:
            Model response string.
        """
        try:
            if self._provider in ("openai", "glm"):
                return self._openai_generate(prompt, system_prompt)
            if self._provider == "anthropic":
                return self._anthropic_generate(prompt, system_prompt)
            if self._provider == "ollama":
                return self._ollama_generate(prompt, system_prompt)
        except _ProviderUnavailableError as exc:
            # Genuinely unconfigured — silent fallback keeps the demo usable
            # without any API keys (the TF-IDF path still returns sources).
            logger.warning("Provider unavailable, using dummy: %s", exc)
            return self._dummy_response(prompt)
        # Real errors (auth, quota, network) re-raise so the API returns 5xx
        # instead of a fake success.
        return self._dummy_response(prompt)

    def generate_with_context(self, query: str, context: str) -> str:
        """Generate a RAG response given a user query and retrieved context.

        Args:
            query: User question.
            context: Retrieved document context.

        Returns:
            Answer string.
        """
        system_prompt = (
            "You are a helpful assistant. Answer the user's question based solely on the "
            "provided context. If the context does not contain enough information, say so."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        return self.generate(user_prompt, system_prompt=system_prompt)

    def stream_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream model output token by token.

        Args:
            prompt: User message.
            system_prompt: Optional system instructions.

        Yields:
            Text chunks as they arrive.
        """
        try:
            if self._provider in ("openai", "glm"):
                yield from self._openai_stream(prompt, system_prompt)
                return
            if self._provider == "anthropic":
                yield from self._anthropic_stream(prompt, system_prompt)
                return
            if self._provider == "ollama":
                yield from self._ollama_stream(prompt, system_prompt)
                return
        except _ProviderUnavailableError as exc:
            logger.warning("Provider unavailable, using dummy stream: %s", exc)
            for word in self._dummy_response(prompt).split():
                yield word + " "
            return
        # Any other provider error (auth/quota/network) re-raises so the
        # WebSocket handler can surface it as a structured error event.

    def generate_messages(self, messages: List[dict]) -> str:
        """Generate a response given a full conversation messages list.

        This is the messages-list counterpart to generate(). It accepts the
        standard OpenAI-style messages format (list of role/content dicts),
        which carries the full sliding-window chat history.

        RAG Pipeline Position:
          Chat History → [generate_messages] → Provider → Answer
                              ^^^
          Whereas generate() wraps a single prompt string, this method passes
          the conversation history straight through to the provider — enabling
          multi-turn awareness at the LLM level.

        Args:
            messages: Conversation history as OpenAI-style dicts, e.g.:
                [
                    {"role": "system", "content": "..."},
                    {"role": "user",   "content": "..."},
                    {"role": "assistant", "content": "..."},
                    {"role": "user",   "content": "..."},  ← latest turn
                ]

        Returns:
            Model response string, or dummy fallback if provider fails.
        """
        try:
            if self._provider in ("openai", "glm"):
                return self._openai_generate_messages(messages)
            if self._provider == "anthropic":
                return self._anthropic_generate_messages(messages)
            if self._provider == "ollama":
                return self._ollama_generate_messages(messages)
        except _ProviderUnavailableError as exc:
            logger.warning("Provider unavailable, using dummy: %s", exc)
            return self._dummy_messages_response(messages)
        # Real provider errors propagate so the API can return a proper 5xx.
        return self._dummy_messages_response(messages)

    def stream_messages(
        self, messages: List[dict]
    ) -> Generator[str, None, None]:
        """Stream a response given a full conversation messages list.

        Streaming counterpart to generate_messages(). Yields text chunks as
        they arrive so the frontend can display tokens incrementally without
        waiting for the full response.

        Args:
            messages: Conversation history as OpenAI-style dicts (same shape
                as generate_messages()).

        Yields:
            Text chunks from the model, or dummy fallback tokens if the
            provider fails.
        """
        try:
            if self._provider in ("openai", "glm"):
                yield from self._openai_stream_messages(messages)
                return
            if self._provider == "anthropic":
                yield from self._anthropic_stream_messages(messages)
                return
            if self._provider == "ollama":
                yield from self._ollama_stream_messages(messages)
                return
        except _ProviderUnavailableError as exc:
            logger.warning("Provider unavailable, using dummy stream: %s", exc)
            for word in self._dummy_messages_response(messages).split():
                yield word + " "
            return
        # Real provider errors propagate — the WebSocket handler turns them
        # into a structured {"type": "error"} event for the client.

    def list_models(self) -> List[str]:
        """Return a list of available models for the current provider.

        Returns:
            List of model name strings.
        """
        if self._provider == "openai":
            return self._openai_list_models()
        if self._provider == "anthropic":
            return list(ANTHROPIC_MODELS)
        if self._provider == "ollama":
            return self._ollama_list_models()
        return ["dummy-model"]

    # ------------------------------------------------------------------ #
    # OpenAI                                                               #
    # ------------------------------------------------------------------ #

    def _is_constrained_openai_model(self) -> bool:
        """Return True if the current model rejects `temperature` overrides.

        WHY: GPT-5 family and the o-series reasoning models only accept the
        default temperature (1). Sending any other value raises a 400.
        Older families (gpt-4, gpt-4o, gpt-4.1) still accept the full range.
        """
        m = self.model.lower()
        return (
            m.startswith("gpt-5")
            or m.startswith("o1")
            or m.startswith("o3")
            or m.startswith("o4")
        )

    def _openai_kwargs(self, **extra: Any) -> dict:
        """Build kwargs for an OpenAI chat completions call.

        PATTERN: Centralised so every call site picks up the same constraints:
          - Always use `max_completion_tokens` (gpt-5*/o* require it; modern
            gpt-4* accept it).
          - Omit `temperature` for constrained models rather than sending the
            default, so a future SDK default change doesn't surprise us.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            **extra,
        }
        if not self._is_constrained_openai_model():
            kwargs["temperature"] = self.temperature
        return kwargs

    def _openai_generate(self, prompt: str, system_prompt: Optional[str]) -> str:
        client = self._openai_client_for_provider()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            **self._openai_kwargs(messages=messages)
        )
        return response.choices[0].message.content or ""

    def _openai_stream(
        self, prompt: str, system_prompt: Optional[str]
    ) -> Generator[str, None, None]:
        client = self._openai_client_for_provider()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = client.chat.completions.create(
            **self._openai_kwargs(messages=messages, stream=True)
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def _openai_list_models(self) -> List[str]:
        if not _OPENAI_AVAILABLE:
            return list(OPENAI_MODELS)
        try:
            client = _openai_module.OpenAI(  # type: ignore[union-attr]
                api_key=self.api_key or os.getenv("OPENAI_API_KEY")
            )
            models = client.models.list()
            return [m.id for m in models.data if "gpt" in m.id]
        except Exception as exc:
            logger.warning("Could not list OpenAI models: %s", exc)
            return list(OPENAI_MODELS)

    # ------------------------------------------------------------------ #
    # Anthropic                                                            #
    # ------------------------------------------------------------------ #

    def _anthropic_generate(self, prompt: str, system_prompt: Optional[str]) -> str:
        if not _ANTHROPIC_AVAILABLE:
            raise _ProviderUnavailableError("anthropic package not installed")

        client = _anthropic_module.Anthropic(  # type: ignore[union-attr]
            api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        block = response.content[0]
        return block.text if hasattr(block, "text") else str(block)

    def _anthropic_stream(
        self, prompt: str, system_prompt: Optional[str]
    ) -> Generator[str, None, None]:
        if not _ANTHROPIC_AVAILABLE:
            raise _ProviderUnavailableError("anthropic package not installed")

        client = _anthropic_module.Anthropic(  # type: ignore[union-attr]
            api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    # ------------------------------------------------------------------ #
    # Ollama                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wrap_ollama_errors(exc: Exception) -> _ProviderUnavailableError:
        """Convert a requests error from a local Ollama into unavailable.

        WHY: A local Ollama that's offline, returns 404 for an unknown model,
        or refuses connections is functionally "not configured" — same bucket
        as a missing package or missing API key. Wrapping these as
        _ProviderUnavailableError lets the top-level dispatch fall back to
        the dummy response without suppressing real provider auth/quota
        errors from OpenAI, Anthropic, or GLM.
        """
        return _ProviderUnavailableError(f"Ollama unavailable: {exc}")

    def _ollama_generate(self, prompt: str, system_prompt: Optional[str]) -> str:
        if not _REQUESTS_AVAILABLE:
            raise _ProviderUnavailableError("requests package not installed")

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = _requests_module.post(  # type: ignore[union-attr]
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except _requests_module.exceptions.RequestException as exc:  # type: ignore[union-attr]
            raise self._wrap_ollama_errors(exc) from exc
        return resp.json().get("response", "")

    def _ollama_stream(
        self, prompt: str, system_prompt: Optional[str]
    ) -> Generator[str, None, None]:
        if not _REQUESTS_AVAILABLE:
            raise _ProviderUnavailableError("requests package not installed")

        import json

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            with _requests_module.post(  # type: ignore[union-attr]
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
        except _requests_module.exceptions.RequestException as exc:  # type: ignore[union-attr]
            raise self._wrap_ollama_errors(exc) from exc

    # ------------------------------------------------------------------ #
    # OpenAI — messages-list API                                          #
    # ------------------------------------------------------------------ #

    def _openai_generate_messages(self, messages: List[dict]) -> str:
        """Pass a messages list directly to the OpenAI chat completions API.

        WHY: Unlike _openai_generate() which builds the messages list itself
        from a single prompt, this method receives the already-assembled list
        so the caller controls the full conversation history.
        """
        client = self._openai_client_for_provider()
        response = client.chat.completions.create(
            **self._openai_kwargs(messages=messages)  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""

    def _openai_stream_messages(
        self, messages: List[dict]
    ) -> Generator[str, None, None]:
        """Stream a response from OpenAI given a messages list."""
        client = self._openai_client_for_provider()
        stream = client.chat.completions.create(
            **self._openai_kwargs(messages=messages, stream=True)  # type: ignore[arg-type]
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    # ------------------------------------------------------------------ #
    # Anthropic — messages-list API                                       #
    # ------------------------------------------------------------------ #

    def _anthropic_generate_messages(self, messages: List[dict]) -> str:
        """Generate a response from Anthropic given a messages list.

        WHY: Anthropic's API does NOT accept "system" role messages inside the
        messages list — it expects system instructions as a separate top-level
        `system=` kwarg. We therefore filter out system messages first and join
        their content, then pass only user/assistant turns in `messages`.

        TRADE-OFF: Multiple system messages are joined with newlines. In practice,
        well-formed chat histories have at most one system message (the first one),
        but defensive handling is safer.
        """
        if not _ANTHROPIC_AVAILABLE:
            raise _ProviderUnavailableError("anthropic package not installed")

        client = _anthropic_module.Anthropic(  # type: ignore[union-attr]
            api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY")
        )

        # PATTERN: Separate system messages from conversation turns.
        #          Anthropic requires this split at the API level.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=non_system,
        )
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)

        response = client.messages.create(**kwargs)
        block = response.content[0]
        return block.text if hasattr(block, "text") else str(block)

    def _anthropic_stream_messages(
        self, messages: List[dict]
    ) -> Generator[str, None, None]:
        """Stream a response from Anthropic given a messages list.

        See _anthropic_generate_messages() for the system-message separation
        rationale — the same split applies here.
        """
        if not _ANTHROPIC_AVAILABLE:
            raise _ProviderUnavailableError("anthropic package not installed")

        client = _anthropic_module.Anthropic(  # type: ignore[union-attr]
            api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY")
        )

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]

        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=non_system,
        )
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    # ------------------------------------------------------------------ #
    # Ollama — messages-list API  (/api/chat)                             #
    # ------------------------------------------------------------------ #

    def _ollama_generate_messages(self, messages: List[dict]) -> str:
        """Generate a response from Ollama using the /api/chat endpoint.

        WHY: The existing _ollama_generate() uses /api/generate, which accepts
        a single prompt string. /api/chat is the correct Ollama endpoint for
        multi-turn conversations — it accepts an OpenAI-style messages list
        and understands role context (system/user/assistant).

        Response format differs from /api/generate:
          /api/generate  → {"response": "token", "done": false}
          /api/chat      → {"message": {"role": "assistant", "content": "token"}, "done": false}
        """
        if not _REQUESTS_AVAILABLE:
            raise _ProviderUnavailableError("requests package not installed")

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            resp = _requests_module.post(  # type: ignore[union-attr]
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except _requests_module.exceptions.RequestException as exc:  # type: ignore[union-attr]
            raise self._wrap_ollama_errors(exc) from exc
        # WHY: /api/chat wraps the response inside a "message" object, unlike
        #      /api/generate which puts it at the top-level "response" key.
        return resp.json().get("message", {}).get("content", "")

    def _ollama_stream_messages(
        self, messages: List[dict]
    ) -> Generator[str, None, None]:
        """Stream a response from Ollama /api/chat endpoint.

        See _ollama_generate_messages() for endpoint and response-format details.
        """
        if not _REQUESTS_AVAILABLE:
            raise _ProviderUnavailableError("requests package not installed")

        import json

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            with _requests_module.post(  # type: ignore[union-attr]
                f"{self.ollama_base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        data = json.loads(line)
                        # WHY: /api/chat nests the token inside data["message"]["content"],
                        #      not at data["response"] like /api/generate does.
                        token = data.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
        except _requests_module.exceptions.RequestException as exc:  # type: ignore[union-attr]
            raise self._wrap_ollama_errors(exc) from exc

    def _ollama_list_models(self) -> List[str]:
        if not _REQUESTS_AVAILABLE:
            return list(OLLAMA_DEFAULT_MODELS)
        try:
            resp = _requests_module.get(  # type: ignore[union-attr]
                f"{self.ollama_base_url}/api/tags", timeout=5
            )
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception as exc:
            logger.warning("Could not list Ollama models: %s", exc)
            return list(OLLAMA_DEFAULT_MODELS)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_provider(model: str) -> str:
        """Infer provider from model name.

        BUG FIX: `glm*` used to fall through to Ollama, which silently bricked
        `GLM_API_KEY` configuration in docker-compose.prod.yml. GLM (Zhipu AI)
        exposes an OpenAI-compatible API, so we route it through the OpenAI
        SDK with a custom base_url (see `_openai_client_for_provider`).
        """
        lower = model.lower()
        if lower.startswith("gpt") or lower.startswith("o1") or lower.startswith("o3"):
            return "openai"
        if lower.startswith("claude"):
            return "anthropic"
        if lower.startswith("glm"):
            return "glm"
        return "ollama"

    # GLM / Zhipu AI — OpenAI-compatible endpoint. Configurable via env so
    # the same class works for Zhipu's public API or any self-hosted proxy.
    GLM_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    def _openai_client_for_provider(self):
        """Build an OpenAI-SDK client pointed at the right endpoint.

        WHY: Zhipu's GLM API is wire-compatible with OpenAI's chat completions
        endpoint, so we reuse the same SDK and just redirect base_url+api_key
        when the provider is "glm". Keeping this in one place means every
        chat method (generate, stream, messages) stays a one-liner.
        """
        if not _OPENAI_AVAILABLE:
            raise _ProviderUnavailableError("openai package not installed")
        if self._provider == "glm":
            api_key = self.api_key or os.getenv("GLM_API_KEY")
            if not api_key:
                raise _ProviderUnavailableError("GLM_API_KEY not set")
            base_url = os.getenv("GLM_BASE_URL", self.GLM_DEFAULT_BASE_URL)
            return _openai_module.OpenAI(  # type: ignore[union-attr]
                api_key=api_key, base_url=base_url,
            )
        return _openai_module.OpenAI(  # type: ignore[union-attr]
            api_key=self.api_key or os.getenv("OPENAI_API_KEY")
        )

    @staticmethod
    def _dummy_response(prompt: str) -> str:
        """Return a safe fallback response when no provider is available."""
        return (
            "[LLM unavailable] This is a placeholder response. "
            f"Your prompt was {len(prompt)} characters. "
            "Please configure a valid LLM provider (OpenAI, Anthropic, or Ollama)."
        )

    @staticmethod
    def _dummy_messages_response(messages: List[dict]) -> str:
        """Return a safe fallback response for the messages-list API.

        WHY: generate_messages() receives a List[dict] instead of a plain string,
        so we can't call _dummy_response(prompt) directly. We derive a meaningful
        length metric from the last user message (same sentinel, different context).

        PATTERN: Extract last user turn for the character count so the fallback
        message is as informative as the prompt-based one.
        """
        # Find the last user message to report its length in the fallback string
        last_user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_content = msg.get("content", "")
                break

        return (
            "[LLM unavailable] This is a placeholder response. "
            f"Your last message was {len(last_user_content)} characters "
            f"(conversation has {len(messages)} turn(s)). "
            "Please configure a valid LLM provider (OpenAI, Anthropic, or Ollama)."
        )
