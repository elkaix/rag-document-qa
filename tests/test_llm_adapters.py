"""Contract tests for the LLM provider adapters.

What concept this teaches:
    Each provider lives behind one adapter that speaks the same messages-first
    interface (``generate`` / ``stream``). Because the SDK client is *injected*
    via a zero-arg factory, every provider — including Anthropic and Ollama,
    which previously had no coverage — is unit-testable with a fake client. No
    monkeypatching of module globals.

Seams under test:
    Each adapter's public ``generate`` (-> GenerationResult) and ``stream``
    (-> str chunks then a terminal Usage), plus the ProviderUnavailableError
    triggers that make LLMHandler fall back to the dummy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.llm_handler.adapters.base import (
    GenerationResult,
    ProviderAdapter,
    ProviderUnavailableError,
    Usage,
)
import requests

from src.llm_handler.adapters.anthropic import AnthropicAdapter
from src.llm_handler.adapters.dummy import DummyAdapter
from src.llm_handler.adapters.ollama import OllamaAdapter
from src.llm_handler.adapters.openai_compatible import OpenAICompatibleAdapter


# --------------------------------------------------------------------------- #
# DummyAdapter — the always-available fallback                                #
# --------------------------------------------------------------------------- #

class TestDummyAdapter:
    """The dummy adapter needs no client and always returns a placeholder."""

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(DummyAdapter("any-model"), ProviderAdapter)

    def test_generate_returns_marker_and_counted_usage(self) -> None:
        adapter = DummyAdapter("dummy-model")

        result = adapter.generate([{"role": "user", "content": "What is RAG?"}])

        assert isinstance(result, GenerationResult)
        assert "[LLM unavailable]" in result.text
        # Fallback usage is counted locally, so a non-empty prompt is > 0 tokens.
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0

    def test_stream_yields_tokens_then_terminal_usage(self) -> None:
        adapter = DummyAdapter("dummy-model")

        items = list(adapter.stream([{"role": "user", "content": "Hello"}]))

        # Exactly one terminal Usage, and it is last.
        assert isinstance(items[-1], Usage)
        assert sum(isinstance(i, Usage) for i in items) == 1
        text = "".join(i for i in items if isinstance(i, str))
        assert "[LLM unavailable]" in text


# --------------------------------------------------------------------------- #
# OpenAI-compatible adapter (serves both OpenAI and GLM)                       #
# --------------------------------------------------------------------------- #

class FakeOpenAIClient:
    """Mimics the subset of the OpenAI SDK the adapter calls.

    Records every ``chat.completions.create`` kwargs list in ``calls`` so tests
    can assert the token-parameter/temperature quirks. ``usage`` / ``stream_usage``
    of ``None`` simulate a provider that omits usage (forcing the count fallback).
    """

    def __init__(
        self,
        *,
        usage: tuple[int, int] | None = (10, 8),
        stream_usage: tuple[int, int] | None = (11, 9),
        calls: list[dict] | None = None,
    ) -> None:
        self.calls = calls if calls is not None else []
        self._usage = usage
        self._stream_usage = stream_usage
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs: object):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self._stream_chunks()
        return self._response()

    def _response(self):
        choice = SimpleNamespace(
            message=SimpleNamespace(content="OpenAI answer", role="assistant"),
            finish_reason="stop",
            index=0,
        )
        usage = (
            SimpleNamespace(prompt_tokens=self._usage[0], completion_tokens=self._usage[1])
            if self._usage
            else None
        )
        return SimpleNamespace(choices=[choice], usage=usage)

    def _stream_chunks(self):
        for piece in ("OpenAI ", "answer"):
            delta = SimpleNamespace(content=piece, role="assistant")
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=delta, finish_reason=None, index=0)],
                usage=None,
            )
        if self._stream_usage:
            # Matches the real SDK with stream_options include_usage: a final
            # chunk with empty choices carries the usage totals.
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=self._stream_usage[0],
                    completion_tokens=self._stream_usage[1],
                ),
            )


class TestOpenAICompatibleAdapter:
    """OpenAI and GLM share this adapter — GLM differs only by injected client."""

    def _adapter(self, model: str, fake: FakeOpenAIClient) -> OpenAICompatibleAdapter:
        return OpenAICompatibleAdapter(
            model=model, temperature=0.5, max_tokens=256, client_factory=lambda: fake
        )

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(self._adapter("gpt-4", FakeOpenAIClient()), ProviderAdapter)

    def test_generate_returns_reported_usage(self) -> None:
        fake = FakeOpenAIClient(usage=(10, 8))
        result = self._adapter("gpt-4", fake).generate(
            [{"role": "user", "content": "Q"}]
        )
        assert result.text == "OpenAI answer"
        assert result.usage == Usage(prompt_tokens=10, completion_tokens=8)

    def test_generate_falls_back_to_counted_usage_when_provider_omits_it(self) -> None:
        fake = FakeOpenAIClient(usage=None)
        result = self._adapter("gpt-4", fake).generate(
            [{"role": "user", "content": "Q"}]
        )
        assert result.usage.prompt_tokens > 0  # counted locally, not reported

    def test_constrained_model_omits_temperature_and_uses_completion_tokens(self) -> None:
        fake = FakeOpenAIClient()
        self._adapter("gpt-5-mini", fake).generate([{"role": "user", "content": "Q"}])
        kwargs = fake.calls[-1]
        assert "temperature" not in kwargs
        assert kwargs["max_completion_tokens"] == 256

    def test_unconstrained_model_includes_temperature(self) -> None:
        fake = FakeOpenAIClient()
        self._adapter("gpt-4", fake).generate([{"role": "user", "content": "Q"}])
        assert fake.calls[-1]["temperature"] == 0.5

    def test_stream_yields_tokens_then_reported_usage(self) -> None:
        fake = FakeOpenAIClient(stream_usage=(11, 9))
        items = list(self._adapter("gpt-4", fake).stream([{"role": "user", "content": "Q"}]))
        assert isinstance(items[-1], Usage)
        assert items[-1] == Usage(prompt_tokens=11, completion_tokens=9)
        assert "".join(i for i in items if isinstance(i, str)) == "OpenAI answer"
        # The adapter must request usage on the stream.
        assert fake.calls[-1]["stream_options"] == {"include_usage": True}

    def test_stream_falls_back_to_counted_usage_without_usage_chunk(self) -> None:
        fake = FakeOpenAIClient(stream_usage=None)
        items = list(self._adapter("gpt-4", fake).stream([{"role": "user", "content": "Q"}]))
        assert isinstance(items[-1], Usage)
        assert items[-1].completion_tokens > 0  # counted from streamed text


# --------------------------------------------------------------------------- #
# Anthropic adapter — owns the system-message split                           #
# --------------------------------------------------------------------------- #

class _FakeAnthropicStream:
    """Context-manager stream mirroring anthropic's messages.stream()."""

    def __init__(self, usage: tuple[int, int] | None) -> None:
        self._usage = usage

    def __enter__(self) -> "_FakeAnthropicStream":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    @property
    def text_stream(self):
        yield "Claude "
        yield "answer"

    def get_final_message(self):
        usage = (
            SimpleNamespace(input_tokens=self._usage[0], output_tokens=self._usage[1])
            if self._usage
            else None
        )
        return SimpleNamespace(usage=usage)


class FakeAnthropicClient:
    """Mimics the subset of the Anthropic SDK the adapter calls."""

    def __init__(
        self,
        *,
        usage: tuple[int, int] | None = (12, 7),
        stream_usage: tuple[int, int] | None = (13, 6),
        calls: list[dict] | None = None,
    ) -> None:
        self.calls = calls if calls is not None else []
        self._usage = usage
        self._stream_usage = stream_usage
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    def _create(self, **kwargs: object):
        self.calls.append(kwargs)
        usage = (
            SimpleNamespace(input_tokens=self._usage[0], output_tokens=self._usage[1])
            if self._usage
            else None
        )
        return SimpleNamespace(content=[SimpleNamespace(text="Claude answer")], usage=usage)

    def _stream(self, **kwargs: object) -> _FakeAnthropicStream:
        self.calls.append(kwargs)
        return _FakeAnthropicStream(self._stream_usage)


class TestAnthropicAdapter:
    """Anthropic needs system messages hoisted out of the messages list."""

    def _adapter(self, fake: FakeAnthropicClient) -> AnthropicAdapter:
        return AnthropicAdapter(
            model="claude-sonnet-4-5", max_tokens=256, client_factory=lambda: fake
        )

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(self._adapter(FakeAnthropicClient()), ProviderAdapter)

    def test_generate_maps_reported_usage(self) -> None:
        result = self._adapter(FakeAnthropicClient(usage=(12, 7))).generate(
            [{"role": "user", "content": "Q"}]
        )
        assert result.text == "Claude answer"
        # Anthropic reports input/output tokens -> prompt/completion.
        assert result.usage == Usage(prompt_tokens=12, completion_tokens=7)

    def test_generate_hoists_system_message(self) -> None:
        fake = FakeAnthropicClient()
        self._adapter(fake).generate(
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Q"},
            ]
        )
        kwargs = fake.calls[-1]
        assert kwargs["system"] == "Be terse."
        # The system turn must NOT remain in the messages list.
        assert all(m["role"] != "system" for m in kwargs["messages"])

    def test_stream_yields_tokens_then_reported_usage(self) -> None:
        items = list(
            self._adapter(FakeAnthropicClient(stream_usage=(13, 6))).stream(
                [{"role": "user", "content": "Q"}]
            )
        )
        assert isinstance(items[-1], Usage)
        assert items[-1] == Usage(prompt_tokens=13, completion_tokens=6)
        assert "".join(i for i in items if isinstance(i, str)) == "Claude answer"

    def test_stream_falls_back_to_counted_usage(self) -> None:
        items = list(
            self._adapter(FakeAnthropicClient(stream_usage=None)).stream(
                [{"role": "user", "content": "Q"}]
            )
        )
        assert isinstance(items[-1], Usage)
        assert items[-1].completion_tokens > 0


# --------------------------------------------------------------------------- #
# Ollama adapter — local /api/chat with prompt_eval_count / eval_count usage   #
# --------------------------------------------------------------------------- #

class _FakeOllamaResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeOllamaStreamResponse:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __enter__(self) -> "_FakeOllamaStreamResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_lines(self):
        yield from self._lines


class FakeOllamaHttp:
    """Mimics the requests.post interface the adapter calls."""

    def __init__(
        self,
        *,
        content: str = "Ollama answer",
        usage: tuple[int, int] | None = (20, 15),
        fail: bool = False,
        stream_lines: list[bytes] | None = None,
        calls: list[dict] | None = None,
    ) -> None:
        self.calls = calls if calls is not None else []
        self._content = content
        self._usage = usage
        self._fail = fail
        self._stream_lines = stream_lines

    def post(self, url: str, json: dict | None = None, timeout: int | None = None, stream: bool = False):
        self.calls.append({"url": url, "json": json, "stream": stream})
        if self._fail:
            raise requests.exceptions.ConnectionError("connection refused")
        if stream:
            return _FakeOllamaStreamResponse(self._stream_lines or [])
        payload: dict = {"message": {"content": self._content}}
        if self._usage:
            payload["prompt_eval_count"] = self._usage[0]
            payload["eval_count"] = self._usage[1]
        return _FakeOllamaResponse(payload)


class TestOllamaAdapter:
    """Ollama runs locally; connection refusal means 'unconfigured', not error."""

    def _adapter(self, http: FakeOllamaHttp) -> OllamaAdapter:
        return OllamaAdapter(
            model="llama3",
            temperature=0.5,
            max_tokens=256,
            base_url="http://localhost:11434",
            client_factory=lambda: http,
        )

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(self._adapter(FakeOllamaHttp()), ProviderAdapter)

    def test_generate_posts_to_chat_and_returns_usage(self) -> None:
        http = FakeOllamaHttp(usage=(20, 15))
        result = self._adapter(http).generate([{"role": "user", "content": "Q"}])
        assert result.text == "Ollama answer"
        assert result.usage == Usage(prompt_tokens=20, completion_tokens=15)
        assert http.calls[-1]["url"].endswith("/api/chat")

    def test_connection_failure_raises_provider_unavailable(self) -> None:
        adapter = self._adapter(FakeOllamaHttp(fail=True))
        with pytest.raises(ProviderUnavailableError):
            adapter.generate([{"role": "user", "content": "Q"}])

    def test_stream_yields_tokens_then_reported_usage(self) -> None:
        lines = [
            b'{"message": {"content": "Ollama "}, "done": false}',
            b'{"message": {"content": "answer"}, "done": false}',
            b'{"done": true, "prompt_eval_count": 21, "eval_count": 14}',
        ]
        items = list(
            self._adapter(FakeOllamaHttp(stream_lines=lines)).stream(
                [{"role": "user", "content": "Q"}]
            )
        )
        assert isinstance(items[-1], Usage)
        assert items[-1] == Usage(prompt_tokens=21, completion_tokens=14)
        assert "".join(i for i in items if isinstance(i, str)) == "Ollama answer"

    def test_stream_falls_back_to_counted_usage_without_counts(self) -> None:
        lines = [
            b'{"message": {"content": "hi"}, "done": false}',
            b'{"done": true}',
        ]
        items = list(
            self._adapter(FakeOllamaHttp(stream_lines=lines)).stream(
                [{"role": "user", "content": "Q"}]
            )
        )
        assert isinstance(items[-1], Usage)
        assert items[-1].completion_tokens > 0
