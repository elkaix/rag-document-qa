# ADR 0007 — Splitting the RAGBackend facade

- **Status:** Accepted
- **Sequencing:** Step 5 of issue #16, executed with the evidence from the 2026-09-09 architecture review.
- **Date:** 2026-09-09

## Context

`RAGBackend` was 1265 lines and 21 public methods. The review measured what was actually inside it: of ~525 non-comment code lines, **68% belonged to two clusters with no collaborator behind them** — conversation persistence (204 lines) and evaluation orchestration (154 lines) — written as inline SQLModel queries. **17 of the 19 `select(` calls in `src/` were in that one file.**

The facade was not shallow. Applying the deletion test per cluster: deleting it would make exactly *one* cluster simpler — query, which ADR 0004 had already deepened. Conversation persistence would reappear across nine route handlers, with the message helpers duplicated between the WebSocket handler and the conversation routes; `evaluate_message` alone (112 code lines) would become the largest function in the API layer. Complexity **reappeared** rather than vanishing, which is the signature of a module doing real work — in the wrong place.

The cost was testability. Reaching any conversation behaviour meant constructing the whole RAG facade: a Chroma collection, three LLM handlers, a retriever, a query engine. The evaluation cluster's skip and dedup branches had no direct tests at all, and substituting a judge meant reassigning a module global.

## Decision

**Two packages, each depending on the session factory and nothing else.**

`src/conversations/` — `ConversationStore` (thread lifecycle, search, export, sharing), `ConversationHistory` (message persistence, the sliding window, the auto-title rule), and `shaping.py` (the row→dict wire shapes, which had been written out at five call sites).

`src/evaluation/` — the former `src/evaluation.py` becomes `judges.py`, joined by `message_evaluator.py`. That also resolves a naming collision the review flagged: a module named `src/evaluation.py` sat beside `src/eval/`, `src/api/routes/evaluation.py` and `src/api/routes/eval.py`, shared by production and the harness, and every reader's first guess about it was wrong.

**Injection, not construction.** Both take the *session factory*, not an engine, so the facade's session-per-operation policy keeps one owner rather than being reinvented twice. `MessageEvaluator` also takes a `Judges` struct defaulting to the real functions, so substituting a judge is part of the interface instead of a module patch.

**The facade keeps its whole public surface.** Issue #16 §4 is explicit that routes keep addressing the facade and nothing user-facing changes — including `get_stats()` and `query()`, which have no `src/` callers but are part of the contract.

**One dependency-injection seam.** `BackendDep` moves from `conversations.py` into `src/api/dependencies.py` and every route module uses it. Previously one route file of six used the seam while the other five read `request.app.state.backend` directly, so overriding `get_backend` in a test changed one module's behaviour out of five.

## Consequences

- **`src/backend.py`: 1265 → 783 lines.** Every extracted module is under the project's 250-line ceiling except `message_evaluator.py` (334, including its docstrings).
- **Behaviour was pinned before it moved.** 33 characterization tests were written and committed against the *old* code first, then the extraction was made to pass them with only monkeypatch *paths* changed. Judge injection came as a separate step, once that was green.
- **Testing got dramatically cheaper.** `tests/test_conversations.py` — 33 tests — runs in 0.84s against an in-memory database, with no Chroma collection, no LLM handler and no retriever.
- **A preserved subtlety:** the cached-faithfulness branch carries `details` while the other two metrics do not, because the frontend renders a claim-level breakdown from it. The characterization tests pin this; collapsing the three near-identical blocks would otherwise have quietly normalised it away.
- **Writing the tests surfaced the real judge contracts:** `answer_relevancy` returns a 2-tuple while `faithfulness` and `context_precision` return 3-tuples, and `details` is a JSON string, not a dict.
- **436 tests pass.**
