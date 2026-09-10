# ADR 0006 — One owner for the retrieval composition rule

- **Status:** Accepted; the deferral bullet in Consequences is superseded by [ADR 0009](0009-wire-hybrid-and-multi-query.md)
- **Sequencing:** Follow-up to ADR 0004, from the 2026-09-09 architecture review. Top recommendation of that review.
- **Date:** 2026-09-09

## Context

ADR 0004 cut a `Retriever` seam and promoted the four eval-proven levers into `src/retrieval/` so production could activate them by configuration. The seam works: dense, hybrid, reranking and multi-query all present the same interface, and the composing adapters (`RerankingRetriever`, `MultiQueryRetriever`) wrap an inner Retriever.

What it did not unify is the rule for *how those adapters stack*. That knowledge stayed in two modules, selected by two different vocabularies:

- `src/retrieval/factory.py` — `build_retriever(strategy: str, ...)`. Selected by a **strategy name**. Could express `dense` and `reranked`; raised for `hybrid` and `multi_query`. Returned a Retriever, and `RAGBackend` passed `TOP_K_RESULTS` to the engine alongside it.
- `src/eval/pipeline_factory.py:_get_engine` — selected by **four boolean levers** (`hybrid.enabled`, `reranker.model`, `query_rewriter.model`, `refusal_handler.enabled`). Could express combinations production could not. And it derived the effective top-k from the composition: *when reranking is on, the final count is `final_top_k`, because the reranker over-fetches the wider `rerank_top_n` first.*

**Production had no equivalent of that last rule.** `backend.py` passed `TOP_K_RESULTS` unconditionally. The two paths agree today only because `final_top_k` (5, in all six shipped configs) and `TOP_K_RESULTS` (5) happen to be the same number. That is agreement by coincidence, not by construction: tuning either one would silently make the harness measure a different pipeline than the one served — precisely the failure ADR 0004 was written to eliminate, recurring one level up from the prompt.

Nothing would have caught it. The parity test uses a non-reranked config, so it never exercises the branch, and the composition rule at `pipeline_factory.py:244-280` had no test at all — it was a private method on an object whose construction spins up a real Chroma client and may download an 80 MB cross-encoder, so the rule could not be exercised without that I/O.

## Decision

**A `src/retrieval/composition.py` module owning the rule.**

`compose_retrieval(base=..., rewriter=..., reranker=..., top_k=..., rerank_over_fetch_n=..., rerank_final_top_k=...)` returns a **`RetrievalPlan`** — the composed `Retriever` *and* its effective top-k.

The top-k travels with the retriever because reranking changes it. A caller that receives only a Retriever cannot compute the right count without re-deriving the composition, which is exactly how the two rules diverged.

`compose_retrieval` takes an already-built **base Retriever**, not a vector store. It therefore composes without touching storage, an embedder, or a cross-encoder — which is what makes the rule unit-testable.

**Production strategy names become presets.** `build_retrieval_plan(strategy, vector_store, ...)` maps `dense` / `reranked` onto `compose_retrieval` and keeps ADR 0004's deferral of `hybrid` and `multi_query`, error message and all. *(Superseded by [ADR 0009](0009-wire-hybrid-and-multi-query.md): all four names are now presets; the deferral branch is gone.)* `src/retrieval/factory.py` is deleted rather than kept as a shim, following the ADR 0003/0004 precedent.

Both callers converge: `RAGBackend` builds a plan and feeds both halves to the engine; `EvalPipeline._get_engine` calls `compose_retrieval` with its lever objects.

## Consequences

- **The rule has one owner and, for the first time, tests.** `tests/test_retrieval_composition.py` pins the wrapping order (reranking outside rewriting outside base), that every combination still conforms to the seam, that the reranker over-fetches wider than its final count, and the effective-top-k rule in all three of its cases.
- **Deferral behaviour is unchanged.** `hybrid` and `multi_query` still raise, still name ADR 0004. This ADR moves where composition lives; it does not ship the deferred levers.
- **A structural guard.** A parity test asserts that neither `src/backend.py` nor `src/eval/pipeline_factory.py` mentions `RerankingRetriever(` or `MultiQueryRetriever(` — neither may stack adapters itself again.
- **The duplicated literals are gone too.** `rerank_top_n` and `final_top_k` in the eval config now derive from `RERANK_OVER_FETCH_N` and `TOP_K_RESULTS`; a test pins that. Previously a comment asserted they matched and nothing enforced it.
- **Naming:** `build_retriever` → `build_retrieval_plan`; CONTEXT.md and `src/config.py`'s comment updated. ADR 0004's prose still refers to `build_retriever`; it is left as written, since an ADR records the decision at its date.
- **377 tests pass.**
