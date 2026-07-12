# ADR 0004 — Retriever seam and the shared QueryEngine

- **Status:** Accepted
- **Sequencing:** Step 4 of the RAG architecture deepening spec ([issue #16](https://github.com/elkaix/rag-document-qa/issues/16)); resolves the Retriever-seam and shared-QueryEngine map tickets.
- **Date:** 2026-07-12

## Context

Two coupled problems, both rooted in there being no seam between retrieval and
generation:

1. **Proven retrieval levers couldn't ship.** Hybrid BM25, cross-encoder
   reranking, query rewriting, and refusal handling existed only in
   `src/eval/`, with no production interface to activate them.
2. **Three diverged answer prompts, and eval measured a different pipeline.**
   The sync query path used a *plain* answer prompt; the streaming path used a
   *Markdown* one; the eval harness carried a *third* copy ("Answer **the**
   question… say so **clearly**") and joined context with a bare newline join
   (no `[filename]` prefix) and different chunking defaults (512/64 vs 500/50).
   Eval therefore scored a pipeline that was not the one served, and telemetry
   assembly was duplicated across four backend sites.

## Decision

**A `Retriever` seam.** A runtime-checkable Protocol —
`retrieve(query, top_k) -> list[SearchResult]` — with adapters that either
conform directly or compose an inner Retriever:

- `DenseRetriever` wraps the vector store (the default).
- `BM25HybridRetriever` conforms directly.
- `RerankingRetriever` composes an inner Retriever: over-fetches, then
  re-scores with a cross-encoder.
- `MultiQueryRetriever` composes an inner Retriever: fans rewritten queries out,
  unions, and dedups by chunk_id keeping each chunk's best score.

The four eval-proven levers were **promoted from `src/eval/` to a core
`src/retrieval/` package** (via `git mv`, no shims — same dependency-direction
fix as ADR 0003), so production can activate them without importing eval.

**A deep `QueryEngine` module** owns retrieve→generate for both paths behind a
small interface (`ask` sync, `ask_stream` streaming). The two are separate
methods sharing prompt/context/telemetry helpers — only streaming runs the
planning pass, so sync keeps its single LLM call. The engine owns:

- The **single answer prompt**: the Markdown one, for every path (the sync path
  adopts it — a deliberate, product-improving change: the frontend renderer
  expects Markdown, and eval must measure the shipped prompt).
- **Filename-prefixed context** everywhere (the eval bare join is retired).
- **Telemetry assembly** once, from provider-reported `Usage` (ADR 0003).
- An **optional refusal gate**, off by default. The gate is checked *before* the
  no-documents branch, so an empty retrieval is itself an answerability signal
  the gate may act on.

**Production selects a strategy by config** (`RETRIEVER_STRATEGY`, default
`dense`) through a `build_retriever` factory: `dense` and `reranked` are wired;
`hybrid` and `multi_query` are recognised but **deferred** (see Consequences).
`RAGBackend` delegates `query`/`query_with_telemetry`/`stream_query` to the
engine and owns only conversation persistence.

**The eval harness converges onto the engine.** `EvalPipeline` composes its
levers into one Retriever behind the seam and delegates retrieve→generate to a
`QueryEngine`; its divergent prompt/context/top-k copies are deleted.
`EvalConfig` chunking/top-k/model defaults now derive from `src/config.py`
(single source of truth) — production ingestion reads the same constants. The
values are unchanged (config holds production's actual 512/64), so this is pure
single-sourcing with no behaviour delta on either side. A parity test pins eval
to the shipped prompt and context builders.

## Consequences

- **Behaviour preserved for production:** `test_backend.py` and
  `test_backend_telemetry.py` pass unchanged — the facade contract
  (`query`/`query_with_telemetry`/`stream_query` shapes, the streaming event
  protocol, the empty-store path) is intact. The sync answer becoming Markdown
  is invisible to those tests (dummy LLM) and is the intended product change.
- **Streaming persistence** now happens at one point (the terminal `result`
  event), gated on non-empty results — the empty-store conversation path
  persists nothing, as before, and conversation writes concentrate ahead of the
  step-5 ConversationStore extraction. The only residual difference from the old
  code is an untested mid-generation-crash edge (old: a dangling user message;
  now: nothing) — "don't persist a half-failed turn" is the more defensible
  behaviour.
- **Eval now measures the shipped pipeline.** Deliberate, spec-accepted changes:
  eval telemetry granularity collapses from per-lever stages
  (`rewrite`/`rerank`/`refusal_check`) to the engine's `retrieve`/`generate`;
  the dead `rewriter_cost_usd` field is dropped (verified: zero readers); and
  multi-query dedup shifts from first-seen to best-score-and-truncate (the
  *shipped* `MultiQueryRetriever` semantics) — a retrieval-metric shift that is
  correct-by-definition once eval measures production. And because the gate is
  checked before the no-documents branch, an eval run with an **empty index and
  no refusal handler** now returns the no-documents sentinel instead of
  generating from empty context (the old eval path) — latent, since eval always
  ingests before querying.
- **Deferred, with signal:** `hybrid` needs a live BM25 corpus kept in sync with
  ingestion/deletion (a genuinely new feature), and `multi_query`'s production
  wiring is held so it lands deliberately; `build_retriever` raises a clear
  error pointing here rather than silently falling back. When hybrid *is*
  enabled, `BM25HybridRetriever` emits empty `metadata`/`doc_id` (its corpus is
  `chunk_id -> text`), so citations degrade — acceptable while the lever is off
  by default.
- **New coverage:** contract tests across every adapter; engine tests with a
  fake Retriever and fake LLM asserting sync and streaming issue identical
  answer instructions; a factory strategy→type test; and the eval↔production
  parity test.
