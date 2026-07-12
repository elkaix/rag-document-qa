# ADR 0003 — Telemetry ownership and usage-driven cost

- **Status:** Accepted
- **Sequencing:** Step 3 of the RAG architecture deepening spec ([issue #16](https://github.com/elkaix/rag-document-qa/issues/16)); resolves [#12](https://github.com/elkaix/rag-document-qa/issues/12).
- **Date:** 2026-07-12

## Context

Two coupled problems in the cost-telemetry path:

1. **Inverted dependency.** Token counting (`count_tokens`) and pricing
   (`cost_usd`, `MODEL_PRICES`) lived in `src/eval/` (`_telemetry.py`,
   `pricing.py`), yet **production** imported them — `src/backend.py` and
   `src/llm_handler` reached into the eval package for core utilities.
2. **Cost computed from a reconstruction.** `RAGBackend` did not have the tokens
   the provider billed, so it **rebuilt the prompt string** (a third copy of the
   system + user prompt) purely to re-tokenize it with `count_tokens`. The number
   shown to users was an estimate of a *copy*, in four duplicated assembly sites
   (sync query, streaming conversation branch, streaming single-turn branch, and
   `LLMHandler.generate_with_usage`).

## Decision

**Move token/pricing utilities to a core `src/telemetry/` package** (`tokens.py`,
`pricing.py`). The eval package now imports from core (`from src.telemetry ...`),
never the reverse; `src/eval/__init__.py` re-exports the pricing symbols so its
public API is unchanged. The old `src/eval/_telemetry.py` and `src/eval/pricing.py`
are deleted (moved with `git mv`, not shimmed).

**Consume provider-reported usage** (from ADR 0002's adapters):

- The **synchronous** path calls `LLMHandler.generate_with_usage(...)`, which
  returns the answer plus real `(prompt_tokens, completion_tokens)`. The
  reconstructed prompt and its `count_tokens` calls are gone; the now-orphaned
  `generate_with_context` is deleted.
- The **streaming** path reads the `Usage` carried on the stream's terminal event
  (`LLMHandler.stream_response` / `stream_messages` now yield `str | Usage`). The
  backend discriminates with `isinstance(item, Usage)`.

`cost_usd` is still computed in the backend from those counts.

## Consequences

- Production no longer imports from the eval package — a clean dependency
  direction (verified: `grep` finds no `src.eval` import in backend/llm_handler).
- The cost footer reflects what the provider actually billed (or the adapter's
  local count when a provider omits usage), not a re-tokenized copy of a
  reconstructed prompt.
- **Behavior preserved (deliberately):** telemetry still covers the **answer pass
  only** — the streaming reasoning pass's terminal `Usage` is discarded, keeping
  the per-query cost display to one model's spend, exactly as before. The
  no-documents early-return still emits zero telemetry. `StageTelemetry`'s shape
  is unchanged, so the API and frontend are untouched. `test_backend_telemetry.py`
  passes unchanged as the behavior-preservation proof.
- The four duplicated token-assembly sites collapse to two usage reads (one per
  path); further consolidation of the answer prompt itself is step 4's
  QueryEngine work, not this step.
