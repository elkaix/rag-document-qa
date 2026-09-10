# ADR 0009 — Wiring the hybrid and multi-query strategies for production

- **Status:** Accepted
- **Supersedes in part:** [ADR 0004](0004-retriever-seam-and-query-engine.md) — its "Deferred, with signal" consequence and the degraded-citation acceptance that came with it.
- **Sequencing:** Closes the last two open items of the RAG architecture deepening spec ([issue #16](https://github.com/elkaix/rag-document-qa/issues/16)).
- **Date:** 2026-09-10

## Context

ADR 0004 built the `Retriever` seam and promoted four eval-proven levers into
`src/retrieval/`, then wired only two of them for production. `hybrid` and
`multi_query` were recognised by name and raised a clear error. Each was
deferred for a specific, stated reason:

**`hybrid` — the corpus was frozen at construction.** `BM25HybridRetriever`
took a `documents: dict[str, str]` snapshot, because a sparse index needs the
whole corpus and a similarity search cannot supply one. That was workable in
the eval harness, whose pipelines ingest once and are then read-only, and
unusable in production, where documents are uploaded and deleted while the
process runs. A production instance would have served BM25 hits from whatever
happened to be indexed at boot and kept serving tombstones for deleted files.

ADR 0004 also accepted, explicitly, that a hybrid result carried empty
`metadata` and `doc_id` — "citations degrade — acceptable while the lever is
off by default". That acceptance was conditional on the lever staying off.

**`multi_query` — an unguarded network call inside retrieval.**
`QueryRewriter.expand` caught `json.JSONDecodeError` and nothing else, so a
provider outage, an expired key, a rate limit, or a timeout raised out of
retrieval and failed the user's whole question. Query expansion is a recall
optimisation; retrieval without it is narrower, not broken. There was also no
way for an operator to see what the extra LLM call was costing.

A third thing was wrong in a way neither ADR noticed: `RETRIEVER_STRATEGY` was
a bare literal in `src/config.py`. ADR 0004's central claim — a validated eval
chain is "promoted by *configuration*, not by a code change" — was false as
written. Switching strategies meant editing a tracked source file.

## Decision

**The store publishes a revision; derived indexes ask.** `ChromaVectorStore`
counts its own writes (`upsert`, `delete_by_doc_id`) and exposes the count as
`revision`. A new `ChunkCorpus` (`src/retrieval/corpus.py`) materialises the
whole collection lazily and re-reads it when the revision moves, returning the
*same snapshot object* while nothing has changed so consumers can cache derived
values on its identity. `BM25HybridRetriever` now takes the store, owns a
`ChunkCorpus`, and rebuilds its index per snapshot.

The rejected alternative was explicit invalidation — have ingest and delete
call `invalidate()`. It is correct only as long as every current *and future*
mutation site remembers to, which is the class of bug that does not show up in
tests. The counter lives with the write instead of with the caller.

**The corpus carries whole chunks, not text.** `all_chunk_texts()` is replaced
by `all_chunks() -> dict[str, Chunk]`. A BM25 hit the dense side did not also
return now renders its filename like any other source, which retires ADR 0004's
conditional acceptance rather than inheriting it.

**Hybrid degrades instead of raising.** `BM25Okapi` divides by the corpus size
and by the term count, so an empty collection and a collection of
untokenizable text both raised `ZeroDivisionError` from inside the library —
and an empty index is the normal state of a fresh deployment. Both now produce
a disabled sparse half and dense-only results. Chunks scoring zero on BM25 are
dropped rather than fused, because on a corpus smaller than `bm25_top_k` every
non-match would otherwise inherit a rank and reach the generator as context.

**The refusal gate reads the best score, not position 0.** `should_refuse` used
`candidates[0].score`, which assumes descending-score order. Dense, reranked
and multi-query retrieval all provide it; hybrid orders by *fused rank*, and a
BM25-only hit carries score `0.0` because no comparable dense similarity
exists. `max(...)` is the question the gate actually means and is
behaviour-identical for every already-shipping strategy.

**Multi-query fuses by rank, not by score.** `MultiQueryRetriever` ranked the
union of its expansions by `score` and truncated. That is correct only while
every inner result's score lives in one comparable space — which stops being
true the moment the inner retriever is `BM25HybridRetriever`. Every eval config
from `phase2e_rewrite.yaml` onward stacks multi-query over hybrid, so the
adapter was sorting exactly the sparse-only results hybrid exists to contribute
to the bottom of the list and cutting them off. It now fuses the per-expansion
rankings with RRF and keeps each chunk's best observed score for consumers that
read the field. `reciprocal_rank_fusion` moved out of `hybrid.py` into
`src/retrieval/fusion.py`: one caller is a hypothetical seam, two make it real.

**Query expansion catches everything and logs its spend.** A failed expansion
returns `([query], 0.0, 0, 0)` and logs with traceback. The cost is logged at
the `MultiQueryRetriever` seam rather than returned: widening
`retrieve(query, top_k)` with a spend channel would make every adapter and
every caller carry a field one lever produces and nothing consumes — the same
verified-zero-readers reasoning that retired eval's `rewriter_cost_usd`.

**`multi_query` fails loudly when unconfigured.** `QueryRewriter(model=None)`
is a legal pass-through, so an unconfigured strategy would compose a retriever
that expands nothing and behaves exactly like `dense` — a deployment believing
it had recall it did not have. `build_retrieval_plan` raises unless
`QUERY_REWRITER_MODEL` is set and an LLM handler was supplied.

**`RETRIEVER_STRATEGY` and `QUERY_REWRITER_MODEL` are read from the
environment.** The hybrid tuning constants (`HYBRID_BM25_TOP_K`,
`HYBRID_DENSE_TOP_K`, `HYBRID_RRF_K`) and `MAX_QUERY_EXPANSIONS` move into
`src/config.py` and the eval schema derives from them, the single-sourcing
already done for the reranker and refusal levers.

## Consequences

- **Production default is unchanged.** `RETRIEVER_STRATEGY` still defaults to
  `dense`. This ADR makes two more strategies *selectable and safe*; choosing
  one is a deployment decision, and `multi_query` in particular adds one LLM
  call per question.
- **Ingestion moves a cost onto the next query.** Under `hybrid`, the first
  query after an upload or a delete pays a full collection read to rebuild the
  BM25 index. It is a latency cliff, not a constant cost, and it is the right
  side of the trade for a corpus read far more often than written.
- **The revision counter is per-process**, which is exactly as strong as
  ChromaDB's own single-writer constraint: production runs one uvicorn worker,
  so there is no second writer whose changes it could miss. A multi-worker
  deployment would need an external sparse index anyway.
- **BM25 goes quiet on tiny corpora.** BM25Okapi's IDF is zero for a term in
  exactly half the corpus and negative above it, so on a two-chunk collection a
  perfectly discriminating term scores zero and is dropped. Term statistics are
  meaningless at that size; going quiet is the correct behaviour.
- **Multi-query retrieval metrics shift again.** ADR 0004 moved eval's dedup
  from first-seen to best-score-and-truncate to match the shipped adapter; this
  moves both to RRF. A chunk that ranks well under several phrasings now beats
  one that ranks highest under a single phrasing, which is the behaviour the
  multi-query literature describes and the only one that survives a hybrid
  inner retriever. Every `phase2e`-and-later config's retrieval numbers move.
- **Sparse-only results score `0.0`.** The two score spaces are not comparable
  and inventing a cosine-space number for a BM25 hit would be worse than
  reporting none. Results are ordered by fused rank; consumers needing "is
  anything here similar enough" must read the best score in the list, which is
  why the refusal gate changed.
- **The eval harness got simpler.** `EvalPipeline` no longer builds its hybrid
  retriever during `ingest()`, and the two divergent corpus-construction sites
  (`dict(zip(ids, documents))` for SQuAD, `all_chunk_texts()` for ML papers)
  are gone. `build_pipeline` wires it up front like every other lever.
- **New coverage:** the corpus freshness contract (cold start on a populated
  collection, identity caching, refresh after write and delete); hybrid
  robustness (empty corpus, untokenizable corpus, zero-score filtering,
  metadata on sparse-only hits, visibility of post-construction ingest and
  delete); rewriter failure fallback at both the rewriter and the seam; and the
  refusal gate's ordering independence.
