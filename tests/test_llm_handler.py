"""
Tests for LLMHandler messages-list API.

RAG Pipeline Position:
  User Query → [LLMHANDLER] → Provider → Response
                    ^^^
  Tests cover the new generate_messages() and stream_messages() methods
  that accept a full conversation history (sliding window) instead of a
  single prompt string. This is the interface the chat frontend will use.

These tests do NOT require any live LLM provider. A nonexistent model name
routes to the Ollama provider, which will fail to connect and fall back to
the dummy response — exercising the full fallback path without mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable so tests can use `from src.llm_handler import ...`
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.llm_handler import LLMHandler, Usage


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

# WHY: "nonexistent-model-xyz" doesn't start with "gpt", "claude", or "glm",
#      so _detect_provider routes it to Ollama. Ollama is not running in CI,
#      so the connection fails and the handler falls back to the dummy response.
#      This lets us test the fallback path deterministically with zero mocking.
_DUMMY_MODEL = "nonexistent-model-xyz"

# A realistic sliding-window conversation: system context + two turns of history
_SLIDING_WINDOW_MESSAGES = [
    {"role": "system", "content": "You are a helpful RAG assistant."},
    {"role": "user", "content": "What is retrieval-augmented generation?"},
    {"role": "assistant", "content": "RAG combines a retriever with a language model."},
    {"role": "user", "content": "How does the retrieval step work?"},
]


# --------------------------------------------------------------------------- #
# generate_messages() tests                                                    #
# --------------------------------------------------------------------------- #

class TestGenerateMessages:
    """Tests for the non-streaming messages-list generation method."""

    def test_generate_messages_falls_back_to_dummy(self) -> None:
        """Nonexistent model returns a dummy string containing '[LLM unavailable]'.

        WHY: If generate_messages() raises or returns empty, the chat frontend
        has no safe default to show. The dummy fallback ensures the API always
        returns a non-empty, identifiable string even when all providers fail.
        """
        handler = LLMHandler(model=_DUMMY_MODEL)
        messages = [{"role": "user", "content": "Hello, world!"}]

        result = handler.generate_messages(messages)

        # PATTERN: Assert the contract (dummy marker present), not the exact string,
        #          so minor wording changes in _dummy_response don't break the test.
        assert isinstance(result, str), "generate_messages must return a str"
        assert "[LLM unavailable]" in result, (
            "Fallback response must contain '[LLM unavailable]' marker"
        )
        assert len(result) > 0, "Fallback response must not be empty"

    def test_generate_messages_accepts_sliding_window(self) -> None:
        """Multi-turn conversation history is accepted and returns a non-empty string.

        WHY: The messages-list API must handle the full sliding window — not just
        a single user message. If the method errors on system/assistant roles, the
        chat history feature is broken. This test uses the fallback path so it
        runs without any live provider.
        """
        handler = LLMHandler(model=_DUMMY_MODEL)

        result = handler.generate_messages(_SLIDING_WINDOW_MESSAGES)

        assert isinstance(result, str), "generate_messages must return a str"
        assert len(result) > 0, "Must return non-empty result for multi-turn input"


# --------------------------------------------------------------------------- #
# stream_messages() tests                                                      #
# --------------------------------------------------------------------------- #

class TestStreamMessages:
    """Tests for the streaming messages-list generation method."""

    def test_stream_messages_falls_back_to_dummy(self) -> None:
        """Nonexistent model yields dummy tokens containing '[LLM unavailable]'.

        WHY: stream_messages() must always yield at least one token — an empty
        generator would leave the chat UI hanging with no response. The dummy
        fallback yields word-by-word tokens from the same sentinel string.
        """
        handler = LLMHandler(model=_DUMMY_MODEL)
        messages = [{"role": "user", "content": "Hello, world!"}]

        items = list(handler.stream_messages(messages))
        full_response = "".join(i for i in items if isinstance(i, str))

        assert full_response, "stream_messages must yield answer text"
        assert "[LLM unavailable]" in full_response, (
            "Streamed fallback must contain '[LLM unavailable]' marker"
        )

    def test_stream_messages_accepts_sliding_window(self) -> None:
        """Multi-turn conversation history is accepted and yields tokens.

        WHY: Validates that stream_messages handles the full OpenAI-style messages
        list (system + user + assistant + user), which is the shape the sliding
        window chat sends. A method that only handles single-user-message lists
        would silently break multi-turn conversations.
        """
        handler = LLMHandler(model=_DUMMY_MODEL)

        items = list(handler.stream_messages(_SLIDING_WINDOW_MESSAGES))
        full_response = "".join(i for i in items if isinstance(i, str))

        assert len(full_response) > 0, "Joined tokens must form a non-empty response"

    def test_stream_messages_ends_with_terminal_usage(self) -> None:
        """The stream yields text chunks, then exactly one terminal Usage.

        WHY: cost telemetry needs the token counts the provider (or the local
        fallback) reports for the streamed call, delivered as the final event so
        text chunks stay pure strings for the UI.
        """
        handler = LLMHandler(model=_DUMMY_MODEL)

        items = list(handler.stream_messages([{"role": "user", "content": "Hi"}]))

        assert isinstance(items[-1], Usage), "last streamed item must be Usage"
        assert sum(isinstance(i, Usage) for i in items) == 1
        assert all(isinstance(i, str) for i in items[:-1]), "chunks before it are text"
