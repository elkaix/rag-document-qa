"""QueryEngine — the one module that owns retrieve->generate.

RAG Pipeline Position:
    Query -> [QUERYENGINE] -> Answer
             Retriever -> (refusal gate) -> prompt -> LLM -> telemetry

What concept it teaches:
    A *deep* module: a small interface (`ask`, `ask_stream`) hiding retrieval,
    prompt construction, generation, and telemetry assembly. Both the production
    facade and the eval harness call it, so a measured improvement is an
    improvement in the shipped system — there is no second retrieve->generate
    implementation to drift from.

Design Decisions (full rationale in ADR 0004):
    - The `Retriever` is injected (constructor injection), so dense / hybrid /
      reranked / multi-query retrieval are swapped by configuration, not code.
    - `ask` (sync) and `ask_stream` (streaming) are two methods sharing the same
      prompt / context / telemetry helpers, NOT one body: only streaming runs the
      extra planning pass, so the sync path keeps its single LLM call.
    - The refusal gate is optional and off by default, preserving production.
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

from src.api.schemas.telemetry import StageTelemetry
from src.llm_handler import LLMHandler, Usage
from src.observability import get_tracer
from src.query_engine import telemetry as telemetry_asm
from src.query_engine.prompt import (
    ANSWER_SYSTEM_PROMPT,
    NO_DOCUMENTS_ANSWER,
    REASONING_SYSTEM_PROMPT,
    build_answer_user_prompt,
    build_context,
    build_reasoning_user_prompt,
)
from src.query_engine.streaming import (
    StreamEvent,
    StreamResult,
    build_answer_messages,
    retrieval_summary,
)
from src.retrieval import RefusalHandler, Retriever
from src.vector_store import SearchResult

logger = logging.getLogger(__name__)


class QueryEngine:
    """Owns retrieve->generate for both the sync and the streaming query paths."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMHandler,
        reasoning_llm: LLMHandler,
        top_k: int,
        refusal: RefusalHandler | None = None,
    ) -> None:
        """Wire the engine to its retrieval and generation collaborators.

        Args:
            retriever: The retrieval strategy (dense by default). Injected so it
                is interchangeable behind the Retriever seam.
            llm: The answer-generation handler.
            reasoning_llm: The (cheaper) handler for the streaming planning pass.
            top_k: Default number of chunks to retrieve.
            refusal: Optional answerability gate; when None there is no gate.
        """
        self._retriever = retriever
        self._llm = llm
        self._reasoning_llm = reasoning_llm
        self._top_k = top_k
        self._refusal = refusal

    def _handler_for(self, model: str | None) -> LLMHandler:
        """Return the default handler, or a per-query one if `model` differs."""
        if model and model != self._llm.model:
            return LLMHandler(model=model)
        return self._llm

    def _refusal_text(self, results: list[SearchResult]) -> str | None:
        """Return the no-answer text if the gate refuses, else None.

        A local binding (not ``self._refusal``) so the None-narrowing is visible
        to the type checker without a cast.
        """
        gate = self._refusal
        if gate is not None and gate.should_refuse(results):
            return gate.refuse_response()[1]
        return None

    # ------------------------------------------------------------------ #
    # Synchronous path                                                     #
    # ------------------------------------------------------------------ #

    def ask(
        self, question: str, top_k: int | None = None, model: str | None = None,
    ) -> tuple[list[SearchResult], str, StageTelemetry]:
        """Retrieve, generate, and assemble telemetry for one question.

        Args:
            question: The user's natural-language question.
            top_k: Chunks to retrieve (defaults to the engine's configured top_k).
            model: Optional per-query answer-model override.

        Returns:
            (results, answer, telemetry). On an empty index or a refusal, results
            is empty and telemetry has zero generation fields (no LLM call).
        """
        k = top_k or self._top_k
        tracer = get_tracer()

        start = time.perf_counter()
        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("top_k", k)
            span.set_attribute("question_len", len(question))
            results = self._retriever.retrieve(question, top_k=k)
            span.set_attribute("results_count", len(results))
        retrieve_ms = (time.perf_counter() - start) * 1000

        # The refusal gate is checked BEFORE the no-documents branch: an empty
        # retrieval is itself an answerability signal the gate is entitled to
        # act on (should_refuse([]) is True when enabled). Production leaves the
        # gate off, so it always falls through to the no-documents notice.
        refusal_text = self._refusal_text(results)
        if refusal_text is not None:
            return [], refusal_text, telemetry_asm.zero(retrieve_ms)
        if not results:
            return [], NO_DOCUMENTS_ANSWER, telemetry_asm.zero(retrieve_ms)

        context = build_context(results)
        handler = self._handler_for(model)

        gen_start = time.perf_counter()
        with tracer.start_as_current_span("rag.generate") as span:
            span.set_attribute("model", handler.model)
            answer, prompt_tokens, completion_tokens = handler.generate_with_usage(
                build_answer_user_prompt(context, question),
                system_prompt=ANSWER_SYSTEM_PROMPT,
            )
            span.set_attribute("answer_len", len(answer))
        generate_ms = (time.perf_counter() - gen_start) * 1000

        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        return results, answer, telemetry_asm.assemble(
            retrieve_ms, generate_ms, handler.model, usage
        )

    # ------------------------------------------------------------------ #
    # Streaming path                                                       #
    # ------------------------------------------------------------------ #

    def ask_stream(
        self,
        question: str,
        top_k: int | None = None,
        model: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream retrieve -> plan -> answer as typed events.

        Yields ("status"|"reasoning"|"token", str) display events, then one
        terminal ("result", StreamResult). The facade consumes the terminal event
        to persist the turn and emit its own done/telemetry events.

        Args:
            question: The user's question.
            top_k: Chunks to retrieve (defaults to the configured top_k).
            model: Optional per-query answer-model override.
            history: Prior conversation turns (role/content dicts). When present,
                the answer pass runs multi-turn; when empty, single-turn.
        """
        k = top_k or self._top_k
        history = history or []
        tracer = get_tracer()

        yield ("status", "Searching indexed documents...")
        start = time.perf_counter()
        with tracer.start_as_current_span("rag.retrieve") as span:
            span.set_attribute("top_k", k)
            span.set_attribute("question_len", len(question))
            results = self._retriever.retrieve(question, top_k=k)
            span.set_attribute("results_count", len(results))
        retrieve_ms = (time.perf_counter() - start) * 1000

        refusal_text = self._refusal_text(results)
        if refusal_text is not None:
            yield ("token", refusal_text)
            yield ("result", StreamResult([], telemetry_asm.zero(retrieve_ms), self._llm.model))
            return
        if not results:
            yield ("status", "No indexed documents — nothing to retrieve.")
            yield ("token", NO_DOCUMENTS_ANSWER)
            yield ("result", StreamResult([], telemetry_asm.zero(retrieve_ms), self._llm.model))
            return

        yield ("status", retrieval_summary(results))
        context = build_context(results)
        handler = self._handler_for(model)

        yield from self._stream_reasoning(context, question)

        yield ("status", "Composing answer...")
        answer_usage: Usage | None = None
        gen_start = time.perf_counter()
        with tracer.start_as_current_span("rag.generate") as span:
            span.set_attribute("model", handler.model)
            span.set_attribute("has_conversation", bool(history))
            stream = (
                handler.stream_messages(build_answer_messages(history, context, question))
                if history
                else handler.stream_response(
                    build_answer_user_prompt(context, question),
                    system_prompt=ANSWER_SYSTEM_PROMPT,
                )
            )
            answer_len = 0
            for item in stream:
                if isinstance(item, Usage):
                    answer_usage = item
                    continue
                answer_len += len(item)
                yield ("token", item)
            span.set_attribute("answer_len", answer_len)
        generate_ms = (time.perf_counter() - gen_start) * 1000

        usage = answer_usage or Usage(prompt_tokens=0, completion_tokens=0)
        telemetry = telemetry_asm.assemble(retrieve_ms, generate_ms, handler.model, usage)
        yield ("result", StreamResult(results, telemetry, handler.model))

    def _stream_reasoning(self, context: str, question: str) -> Iterator[StreamEvent]:
        """Stream the planning pass; a failure here must not block the answer."""
        yield ("status", f"Analyzing retrieved context ({self._reasoning_llm.model})...")
        try:
            for item in self._reasoning_llm.stream_response(
                build_reasoning_user_prompt(context, question),
                system_prompt=REASONING_SYSTEM_PROMPT,
            ):
                if isinstance(item, Usage):
                    continue  # reasoning usage is out of telemetry scope (ADR 0003)
                yield ("reasoning", item)
        except Exception as exc:  # noqa: BLE001 — best-effort; degrade to answer.
            logger.warning("Reasoning pass failed: %s", exc)
            yield ("status", "Reasoning unavailable — skipping to answer.")
