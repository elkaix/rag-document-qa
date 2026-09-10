# CONTEXT — domain glossary

The vocabulary of this codebase, so names stay consistent across modules, tests,
and ADRs. New module names enter here as the architecture-deepening spec
([issue #16](https://github.com/elkaix/rag-document-qa/issues/16)) lands them.

Design terms use the `/codebase-design` vocabulary: **module** (a unit with an
interface hiding behaviour), **interface** (the public surface), **depth** (much
behaviour behind a small interface), **seam** (a boundary you can substitute at),
**adapter** (a module presenting one interface over another), **leverage**,
**locality**.

## Modules & types

- **ProviderAdapter** — the interface (Protocol) one LLM provider hides behind:
  `generate(messages) -> GenerationResult` and `stream(messages)` yielding text
  chunks then a terminal `Usage`. Implementations: `OpenAICompatibleAdapter`
  (OpenAI + GLM), `AnthropicAdapter`, `OllamaAdapter`, `DummyAdapter`. Each is
  constructed with an injected `client_factory` so it is testable with a fake.
  See [ADR 0002](docs/adr/0002-provider-adapters.md).
- **Usage** — a value object of `(prompt_tokens, completion_tokens)` for one
  generation call. Provider-reported where the SDK returns it, adapter-counted
  otherwise.
- **GenerationResult** — the `(text, usage)` pair returned by a non-streaming
  generation, so callers never rebuild a prompt to estimate what was billed.
- **LLMHandler** — owns provider *selection* (from the model-name prefix), the
  single-prompt → messages translation, and the unconfigured-provider fallback
  to `DummyAdapter`. Delegates all SDK work to one selected `ProviderAdapter`.
- **telemetry** (`src/telemetry/`) — core token counting (`count_tokens`) and
  cost pricing (`cost_usd`, `MODEL_PRICES`). Owned by the core so both production
  telemetry assembly and the eval harness use one source of truth; the eval
  package imports from here, never the reverse. See
  [ADR 0003](docs/adr/0003-telemetry-ownership.md).
- **domain** (`src/domain.py`) — the leaf module holding the value objects that
  cross module seams: `Document`, `Chunk`, `SearchResult`, and `content_hash`.
  It imports nothing from this package, so naming a type at a seam never drags
  an implementation along. `SearchResult` used to live in `src/vector_store.py`
  (which does `import chromadb`), so the whole `retrieval` and `query_engine`
  packages imported the storage vendor merely to name what a Retriever returns.
  See [ADR 0005](docs/adr/0005-domain-value-types.md).
- **ingestion** (`src/ingestion/`) — `parsers` (one function per format behind a
  `PARSERS` registry, from which `SUPPORTED_EXTENSIONS` is derived, plus the pure
  `normalise_pdf_text`), `loader` (paths, source metadata, batch error policy),
  and `chunking` (the three strategies and the quality filters). Replaces the
  504-line `document_loader` module, whose format dispatch went to private
  methods and whose PDF, DOCX and HTML paths had no tests. See
  [ADR 0008](docs/adr/0008-ingestion-parsing-seam.md).
- **Retriever** — the seam (Protocol) every retrieval strategy hides behind:
  `retrieve(query, top_k) -> list[SearchResult]`. Implementations either conform
  directly (`DenseRetriever`, `BM25HybridRetriever`) or *compose* an inner
  Retriever (`RerankingRetriever` over-fetches then cross-encodes;
  `MultiQueryRetriever` fans rewritten queries out and fuses the per-expansion
  rankings with RRF). Live in
  `src/retrieval/`; composed for both production and eval by
  `compose_retrieval` (`src/retrieval/composition.py`). See
  [ADR 0004](docs/adr/0004-retriever-seam-and-query-engine.md).
- **ChunkCorpus** / **CorpusSnapshot** (`src/retrieval/corpus.py`) — the module
  owning the freshness rule for anything derived from the whole corpus. A
  snapshot is an immutable `{chunk_id: Chunk}` map tagged with the store
  revision it was read at; `ChunkCorpus` re-reads only when
  `ChromaVectorStore.revision` has moved. This is what let the BM25 half of
  `BM25HybridRetriever` ship: its index used to freeze at construction, so
  documents ingested afterwards were invisible and deleted ones still matched.
  See [ADR 0009](docs/adr/0009-wire-hybrid-and-multi-query.md).
- **SparseIndex** (`src/retrieval/sparse_index.py`) — the BM25 half of hybrid
  retrieval, as pure functions: build an index over a `CorpusSnapshot`, score a
  query against it, return ranked chunk ids. It owns the degradation rule —
  BM25 is undefined over an empty or untokenizable corpus (a fresh deployment's
  normal state), so the index reports that instead of raising and hybrid falls
  back to dense-only. Split out of `hybrid.py` so that module owns one thing:
  fusing two rankings. See [ADR 0010](docs/adr/0010-score-contract-and-retrieval-hardening.md).
- **Reciprocal Rank Fusion** (`reciprocal_rank_fusion`, `src/retrieval/fusion.py`)
  — merges several ranked id lists into one by rank rather than by score:
  `score(d) = Σ 1/(rrf_k + rank_r(d))`. Used wherever the lists being merged
  score in incomparable spaces — `BM25HybridRetriever` (sparse vs. dense) and
  `MultiQueryRetriever` (one ranking per expansion). Extracted from `hybrid.py`
  when the second caller arrived. See
  [ADR 0009](docs/adr/0009-wire-hybrid-and-multi-query.md).
- **QueryEngine** (`src/query_engine/`) — the deep module owning retrieve→generate
  for both the sync (`ask`) and streaming (`ask_stream`) paths: one Markdown
  answer prompt, filename-prefixed context, an optional refusal gate, and
  telemetry assembly — all in one place. Both `RAGBackend` and the eval harness
  call it, so eval measures the shipped pipeline. See
  [ADR 0004](docs/adr/0004-retriever-seam-and-query-engine.md).
- **ConversationStore** / **ConversationHistory** (`src/conversations/`) — the
  two modules owning chat-thread persistence. The store handles the thread
  lifecycle, search, export and share tokens; the history handles message
  writes, the completed-pairs sliding window fed to the next prompt, and the
  auto-title rule. Both take the session factory and nothing else. Wire shapes
  live in `shaping.py`. See [ADR 0007](docs/adr/0007-backend-split.md).
- **MessageEvaluator** (`src/evaluation/`) — orchestration around the judges:
  load a persisted message, find the question it answered, score what has not
  been scored yet, persist. The pure scoring functions live beside it in
  `judges.py`. Judges are injected via a `Judges` struct so they can be
  substituted without patching a module. See [ADR 0007](docs/adr/0007-backend-split.md).
- **RefusalHandler** — an answerability gate (not a Retriever): refuses when no
  candidate's similarity reaches a threshold (or nothing was retrieved). It reads
  the best score in the set rather than position 0, because hybrid retrieval
  orders by fused rank — [ADR 0009](docs/adr/0009-wire-hybrid-and-multi-query.md). Applied
  inside the QueryEngine; off by default in production.
