# ADR 0010 — What the Retriever seam promises, and hardening the wired strategies

- **Status:** Accepted
- **Sequencing:** Follow-up to [ADR 0009](0009-wire-hybrid-and-multi-query.md), from a full audit of `src/retrieval/`, `src/vector_store.py` and `src/domain.py` after both strategies were wired.
- **Date:** 2026-09-10

## Context

ADR 0009 wired `hybrid` and `multi_query` for production. Wiring them turned a
latent disagreement into a live one: **the seam promised an ordering that one of
its own implementations does not provide.**

`Retriever.retrieve` documented "SearchResult list ordered by descending
relevance", and `SearchResult.score` documented "Similarity in `[0, 1]`". Three
things had quietly stopped being true:

1. `BM25HybridRetriever` orders by **fused RRF rank**, and reports `0.0` for a
   sparse-only hit because no cosine similarity exists for one.
2. `CrossEncoderReranker` writes a raw cross-encoder **logit** (roughly
   `-11..+11`) into the same field.
3. Callers had therefore been written against an invariant that held for the
   default strategy and silently failed for the others.

ADR 0009 fixed the first caller it found (`RefusalHandler`, which read
`candidates[0].score`) and justified the fix by claiming the seam "never
promised" score ordering. That justification was wrong — `base.py` did promise
ordering — and the same conflation was still live one layer up.

**`RAGBackend.query_with_telemetry` computed confidence as the mean of
`results[:3]`** — the first three *positions*, not the three best *scores*. With
the default `TOP_K_RESULTS = 5`, a hybrid answer led by two sparse-only hits
scored `[0.0, 0.0, 0.88, 0.85, 0.80]` as `0.293` instead of `0.843`: the slice
kept both zeroes and discarded the two strongest chunks. The frontend renders
that number. At three results or fewer the two forms agree, which is why it
survived ADR 0009's pass.

The audit also found that **`RAGBackend` is a lifespan singleton** whose
`retrieve()` runs on executor threads for the websocket path
(`src/api/routes/query.py`) and on the event loop for `POST /api/query`. Every
cache ADR 0009 introduced was written as if single-threaded.

## Decision

**The seam promises relevance order, not score order.** `Retriever.retrieve`
now says so explicitly, and `SearchResult.score` is documented as
strategy-specific and *not* cross-comparable. This is the honest contract:
every implementation ranks its results, and each scores them in its own space.
A caller asking "how similar is the best match?" reads
`max(r.score for r in results)`.

**Confidence reads the three best scores.** Identical output for `dense`,
`reranked` and `multi_query`, which already return descending scores; correct
for `hybrid`, which does not.

**The caches are made thread-safe, not documented as unsafe.**

- `ChromaVectorStore` holds a write lock across each mutation *and* its counter
  increment. `+= 1` is a read-modify-write, and two racing writes netting one
  increment would leave a derived index recording a revision that already covers
  a write it never saw — permanently stale and convinced it is fresh.
- `ChunkCorpus.snapshot` holds a lock across the read, so a slower thread cannot
  install an older snapshot over a newer one.
- `BM25HybridRetriever._sparse_index` returns the index it just built rather than
  re-reading the attribute, so a caller can never be handed an index built from a
  different snapshot than the one returned beside it.

**A delete that matched nothing does not move the revision.** The counter means
"the corpus differs from what you last saw", not "a write method was called".
Bumping unconditionally forced a full corpus re-read and BM25 rebuild on the
ordinary 404 and retry paths. Holding the write lock across the count and the
delete is what makes a count of zero a real guarantee rather than a hopeful one.

**`top_k` is a floor on the over-fetch, not a value both halves ignore.**
`POST /api/query` accepts `top_k` up to 50 while `RERANK_OVER_FETCH_N`,
`HYBRID_BM25_TOP_K` and `HYBRID_DENSE_TOP_K` all default to 20. `top_k=50`
returned at most 20 chunks under `reranked` and at most 40 under `hybrid`, with
nothing telling the caller their count was not honoured. Both now fetch
`max(configured, top_k)`.

**RRF counts a repeated id once, at its best rank.** No current caller emits
duplicates, which is exactly why unguarded accumulation would fail silently the
first time one did. The docstring's "ties keep first-seen order" claim is also
retired: float addition is not associative, so mathematically tied ids can differ
in the last bit. What *is* documented now is the argument-order bias — disjoint
lists tie at every rank, and the stable sort resolves each tie toward the list
passed first (hybrid passes sparse first).

**The expansion cap is a bound.** `QueryRewriter` checked its limit after the
append, so `max_expansions=0` returned two queries.

**`sparse_index.py` is split out of `hybrid.py`.** Building and scoring the BM25
index is pure — a snapshot in, ids out — so it is testable without a retriever,
an embedder or a vector store. `hybrid.py` is left owning one thing: fusing two
rankings.

## Consequences

- **No default-path behaviour change.** `dense` is untouched. The confidence,
  refusal and over-fetch corrections are no-ops for any strategy that already
  returned descending scores and asked for fewer results than its width.
- **`reranked` confidence is still uninformative, deliberately left so.** The
  reranker writes a logit into a field the backend clamps to `[0, 1]`, so
  confidence saturates at exactly `1.0` or `0.0`. The fix is a sigmoid — it is
  monotonic, so ranking would survive — but it changes what `reranked`
  retrieval *reports*, which moves `RefusalHandler`'s `0.35` threshold and makes
  stored eval runs incomparable to new ones. That is a scoring-semantics
  decision with an eval-baseline blast radius, so it is recorded here rather
  than taken silently. **Open.**
- **Locks cost nothing available anyway.** ChromaDB is single-writer, so writes
  already serialised; the locks make the counter and caches agree with that.
- **A no-op delete is now cheap.** Deleting an unknown doc_id no longer triggers
  a full corpus re-read.
- **`hybrid` and `reranked` honour large `top_k`,** at the cost of a wider
  fetch than the tuned constants when a caller asks for one.
- **The module ceiling moved to 350 lines** (soft mark at 300), with an explicit
  rule that teaching comments are never shaved to fit under it — the ceiling
  exists to force single-responsibility, not to ration prose.
- **New coverage:** confidence over ordered, unordered, >3 and empty result
  lists; hybrid over-fetch above and below the configured widths; reranker
  over-fetch; RRF duplicate handling; and the revision counter's two delete
  cases. 556 tests pass.
