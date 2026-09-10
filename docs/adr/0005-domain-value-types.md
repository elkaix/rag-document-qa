# ADR 0005 — Value types belong to no module

- **Status:** Accepted
- **Sequencing:** Follow-up to the 2026-09-09 architecture review; not part of issue #16's original eight steps.
- **Date:** 2026-09-09

## Context

ADR 0004 cut a `Retriever` seam: one Protocol, `retrieve(query, top_k) -> list[SearchResult]`, with adapters that either conform or compose. The seam works. But `SearchResult` — the type the seam is defined *in terms of* — was declared in `src/vector_store.py`, the module whose first statement is `import chromadb`.

The consequence: every module that names the seam imports the storage vendor. Ten of them did — the whole `src/retrieval/` package (`base`, `dense`, `hybrid`, `reranker`, `query_rewriter`, `refusal_handler`), the whole `src/query_engine/` package (`engine`, `prompt`, `streaming`), and `src/eval/pipeline_factory.py`. `src/retrieval/base.py`, which exists only to declare the Protocol, could not be read or imported without ChromaDB present.

`Document` and `Chunk` had the same shape of problem one step earlier: naming a chunk meant importing the file-parsing module, so `tests/conftest.py` imported `src/document_loader.py` — with its `pypdf`, `python-docx` and `bs4` branches — to construct a fixture.

A second, related problem: `ChromaVectorStore.query()` converts ChromaDB's distance to a similarity with `score = max(0.0, 1.0 - distance)`, which is only correct in cosine space. Nothing enforced that. Instead `metadata={"hnsw:space": "cosine"}` was spelled out at **nine construction sites** (production, the eval pipeline, and seven test fixtures). A site that omitted it got silently wrong similarity scores — no error, just worse answers.

## Decision

**A leaf module, `src/domain.py`,** owning `Document`, `Chunk`, `SearchResult`, and `content_hash`. It imports nothing from this package, so anything may import it without a cycle and without pulling in an implementation. The modules that previously *defined* these types now import them like everyone else.

Plain dataclasses, not Pydantic: these cross internal seams where both sides are trusted. Validation stays at the API boundary, which has its own schemas.

**The store owns its own invariant.** `ChromaVectorStore.open(client, name, embedding_function=...)` creates the collection with `SPACE_METADATA` and wraps it. All nine construction sites now go through it. The bare constructor remains for the case of an existing collection known to be cosine, and its docstring says so.

A `collection` property replaces the two legitimate reads of `_collection` from outside (facade wiring, teardown naming).

## Consequences

- **The seam can be named without the vendor.** `import src.domain` pulls in neither `chromadb` nor `openai`; a test pins this by subprocess. The `retrieval` and `query_engine` *packages* still import the store through their `__init__` re-exports, which is correct — `DenseRetriever` genuinely wraps a store. The point is that the *type at the seam* no longer requires it.
- **Ids are unchanged.** `content_hash` is the same full SHA-256 as the previous `_hash_text`; parity was verified against the old implementation before the move, including the non-ASCII path. These ids are persisted in SQLite and ChromaDB, so a change would have orphaned every stored chunk. An intermediate version of this change truncated the digest to 16 characters and was caught by that check — the truncation had been deliberately fixed earlier and is recorded in `content_hash`'s docstring so it is not reintroduced a third time.
- **The cosine invariant has one owner.** Nine repetitions became one constant. `metadata={"hnsw:space": "cosine"}` no longer appears anywhere outside `src/vector_store.py`.
- **Fixtures got lighter.** `tests/conftest.py` no longer imports the file-parsing module to build a `Chunk`.
- **New coverage:** `tests/test_domain.py` pins id derivation (including that two documents containing the same paragraph keep distinct chunk ids), the full-digest length, and the vendor-free import; `tests/test_vector_store_chroma.py` pins that `open()` produces a cosine collection and honours an explicit embedding function.
