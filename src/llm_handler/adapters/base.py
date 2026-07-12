"""Provider-adapter contract for the LLM layer.

RAG Pipeline Position:
    Query + Context -> Prompt -> [LLM PROVIDER ADAPTER] -> Answer + Usage
                                         ^^^
    This is the seam between LLMHandler (which owns provider *selection* and the
    prompt/messages translation) and the concrete SDK calls. Each provider lives
    behind one adapter that speaks a single messages-first interface.

Design Decision:
    A ``Protocol`` (not an ABC) defines the contract. WHY: the project standard
    prefers structural typing for narrow, swappable interfaces — an adapter is
    "a ProviderAdapter" if it has ``generate`` and ``stream`` with the right
    shapes, with no inheritance ceremony. ``@runtime_checkable`` lets tests assert
    conformance with ``isinstance``.

    Client construction is *injected*: every adapter receives a zero-arg
    ``client_factory`` so a unit test can pass a fake instead of monkeypatching a
    module global. Before this refactor, two of three real providers had zero
    coverage precisely because the only way to fake them was to reach into module
    internals.

    Usage is reported alongside text: provider-supplied counts where the SDK
    returns them, adapter-internal token counting as the fallback (see each
    adapter). Callers never rebuild a prompt just to estimate what was billed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable


class ProviderUnavailableError(RuntimeError):
    """Raised when a provider is not usable for configuration reasons.

    "Configuration reasons" means the SDK package is not installed, no API key is
    set, or a local Ollama server refuses the connection — i.e. the provider is
    *unconfigured*, not *failing*. LLMHandler catches only this error and falls
    back to the dummy adapter, so the demo stays usable with no keys.

    Genuine runtime errors (auth rejected, quota exhausted, malformed request)
    are NOT this type: they propagate so the API surfaces a real 5xx instead of a
    fake "success" dummy answer.
    """


@dataclass(frozen=True)
class Usage:
    """Token counts for one generation call.

    Attributes:
        prompt_tokens: Tokens billed for the input (system + messages).
        completion_tokens: Tokens billed for the generated output.
    """

    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class GenerationResult:
    """The text and usage returned by a non-streaming generation.

    Attributes:
        text: The model's answer.
        usage: Provider-reported usage where available, adapter-counted otherwise.
    """

    text: str
    usage: Usage


def join_message_text(messages: list[dict]) -> str:
    """Concatenate message contents for fallback token counting.

    Joins with newlines so the count matches what LLMHandler used to reconstruct
    for the same messages — keeping fallback numbers stable across the refactor.
    """
    return "\n".join(str(m.get("content", "")) for m in messages)


def counted_usage(input_text: str, output_text: str, model: str) -> Usage:
    """Estimate usage with the local tokenizer when a provider reports none.

    This is the "adapter-internal counting" fallback: the dummy adapter always
    uses it, and the real adapters fall back to it only when a provider omits
    usage (e.g. a stream that carries no terminal usage chunk).
    """
    # NOTE: token/pricing utilities move to the core `src.telemetry` package in
    #       sequencing step 3; this import becomes `from src.telemetry.tokens
    #       import count_tokens` then. It is the one transient cross-package
    #       import during step 2.
    from src.eval._telemetry import count_tokens

    return Usage(
        prompt_tokens=count_tokens(input_text, model),
        completion_tokens=count_tokens(output_text, model),
    )


@runtime_checkable
class ProviderAdapter(Protocol):
    """One provider behind a messages-first interface.

    Implementations own their SDK's request shape, response parsing, usage
    extraction, and the provider-specific quirks (base URLs, token-parameter
    names, system-message handling). LLMHandler owns provider *selection* and the
    single-prompt -> messages translation, so adapters only ever see messages.
    """

    def generate(self, messages: list[dict], **kwargs: object) -> GenerationResult:
        """Generate a full response for an OpenAI-style messages list.

        Args:
            messages: Conversation as ``[{"role": ..., "content": ...}, ...]``.
            **kwargs: Reserved for future per-call overrides.

        Returns:
            The answer text plus token usage.

        Raises:
            ProviderUnavailableError: If the provider is unconfigured.
        """
        ...

    def stream(self, messages: list[dict], **kwargs: object) -> Iterator[str | Usage]:
        """Stream a response, yielding text chunks then a terminal ``Usage``.

        Consumers discriminate the terminal event with ``isinstance(item, Usage)``:
        every ``str`` is an answer chunk; exactly one final ``Usage`` reports the
        call's token counts.

        Args:
            messages: Conversation as ``[{"role": ..., "content": ...}, ...]``.
            **kwargs: Reserved for future per-call overrides.

        Yields:
            ``str`` answer chunks, then one ``Usage``.

        Raises:
            ProviderUnavailableError: If the provider is unconfigured.
        """
        ...
