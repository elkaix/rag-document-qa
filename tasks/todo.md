# Fix-all pass — architecture review candidates + defects

Source: architecture review 2026-09-09 (6 candidates) + explorer defect list.
Discipline: leaf-first, behaviour-preserving, tests green after each step.

## Tranche 0 — baseline
- [x] Bug #0: test suite made real billable OpenAI calls. Root fix: removed
      import-time `load_dotenv()` from `src/llm_handler/__init__.py`; added
      explicit `src.config.load_env()` called by the two entry points
      (`src/api/main.py`, `src/eval/cli.py`). Test stub is now default-on
      (opt out with `RAG_QA_LIVE_LLM=1`).  → verify: 327 pass, no env flags.

## Tranche 1 — pure defects (failing test first)
- [x] D1 progress: `update_progress` takes `n_total`; `progress_fraction`
      extracted and tested; route forwards the runner's total.
- [x] D2 `_source_dict` now owns `chunk_index`; both paths share one shape.
- [x] D3 `ChromaVectorStore.all_chunk_texts()` added; reach-around deleted.
- [x] D4 `src/eval/doubles.py`: public `DummyEvalLLM` + one env dispatch.
- [x] D5 empty leftover dirs removed.

## Tranche 2 — C3 value types off the vendor  ✔ (ADR 0005)
- [x] `src/domain.py` leaf module; `content_hash` parity verified before moving
- [x] `ChromaVectorStore.open()` owns cosine; 9 sites → 1

## Tranche 3 — C6 configuration  ✔
- [x] `storage.runs_dir()` resolved per call + `base_dir=` on every function;
      cross-module global mutation and 3 duplicated reload-fixtures deleted
- [x] `resolve_api_key()` in providers; duplicate OTLP read removed;
      `allowed_origins()` moved to config with tests
- [x] Rerank widths + refusal defaults single-sourced from `src/config.py`

## Tranche 4 — C1 retrieval composition  ✔ (ADR 0006)
- [x] `tests/test_retrieval_composition.py` pins order + effective top-k
- [x] `src/retrieval/composition.py` owns the rule; `factory.py` deleted
- [x] Parity test now covers the reranked case + a structural guard

## Tranche 5 — C4 eval run submission  ✔
- [x] `src/eval/submission.py` owns config resolution, run-id reservation,
      doubles, registry lifecycle. `RunProgressSink` Protocol keeps eval free
      of any import from the API layer.
- [x] `run_id_override` → plain `run_id`; `current_git_sha()` single-sourced
- [x] Route is HTTP translation; 478 → 409 lines
- [x] Failure path + progress-total forwarding now tested without FastAPI

## Tranche 6 — C2 backend split  ✔ (ADR 0007)
- [x] 33 characterization tests committed against the old code first
- [x] `src/conversations/` + `src/evaluation/`; backend.py 1265 → 783
- [x] `BackendDep` adopted by all six route modules

## Tranche 7 — C5 parsing seam  ✔ (ADR 0008)
- [x] `src/ingestion/`: parsers registry, loader, chunking
- [x] PDF/DOCX/HTML, `normalise_pdf_text`, ToC filter, min-length floor,
      word overlap, semantic tier, vector-store guards — all covered

## Tranche 8 — tooling  ✔
- [x] `pyproject.toml` added: ruff/black/mypy config with reasons for each
      deliberate exception. Ruff: 231 findings → 0.
- [x] Ruff caught four spend-ceiling tests of mine written without `assert`.

## Tranche 9 — run it, then say what it is  ✔
- [x] **Booted the app.** 508 green tests had never executed the lifespan on a
      real process. `uvicorn src.api.main:app` starts clean; 27 operations
      registered (26 HTTP pairs across 23 paths, plus the WebSocket);
      `/health`, `/api/documents`, `/api/conversations`, `/api/eval/configs`,
      `/api/eval/runs` all 200; a live upload → list → delete round-trip of a
      document with repeated text (the tranche-3 crash) succeeds end to end.
- [x] **`python -m src.api.main` did nothing.** README and CLAUDE.md have
      documented it as *the* local-dev command for months; the module had no
      `__main__` guard, so it imported the app and exited 0. Only Docker ever
      started the server, via uvicorn directly. Added the runner + a test that
      asserts it calls uvicorn on `API_HOST:API_PORT`.
- [x] **`docker-compose.prod.yml` shipped CORS wide open.** `allowed_origins()`
      falls back to `["*"]` when unset (deliberate, for local dev) and the prod
      compose file never set it — while the docstring claimed it did. Pinned to
      the nginx origin, overridable via `ALLOWED_ORIGINS`.
- [x] **`Architecture.md` rewritten.** CLAUDE.md calls it "the source of truth
      for component boundaries, data flows, and design rationale" and it still
      described `src/document_loader.py`, `src/evaluation.py`, CORS allowing all
      origins, `CHUNK_SIZE=500`, `DEFAULT_MODEL=glm-5.1`, a 4-variable env table,
      and no `src/retrieval/`, `src/query_engine/`, `src/domain.py`,
      `src/conversations/` or `src/ingestion/` at all. Overview diagram, both
      pipelines, component sections, endpoint table (27 routes), env table
      (11 vars), testing table and design decisions all now match the code.
- [x] **README file tree + project CLAUDE.md paths** refreshed; the "reranking
      is not yet wired" claim corrected.
- [x] **The suite was writing to `data/`.** Every test entering
      `with TestClient(app)` ran the real lifespan against `data/rag.db` and
      `data/chroma/` — the developer's own store; confirmed by mtime. Tests
      that swapped in a mock backend did so only after startup. A session-wide
      autouse fixture in `conftest.py` now points both at tmp, with a test
      asserting the invariant directly.

## Tranche 10 — formatting and the type checker  ✔
- [x] **black applied**: 83 of 138 files reformatted, in one deliberate commit
      so the diff is legible as formatting and nothing else. `black --check`
      and `ruff check` both clean; 515 tests unchanged.
- [x] **mypy 36 → 0**, four root causes, each fixed in the code rather than
      silenced:
      - SQLModel descriptor gap (9): `Model.field.desc()/.contains()/.in_()`
        made mypy see the *value* type. `sqlmodel.col()` is the documented
        answer and reads no worse. This was the item deferred as "the known
        SQLModel typing gap" — it had a real fix after all.
      - Optional SDK imports (3): three identical try/except blocks where
        `import x as _x` rebound an annotated name. Collapsed into one
        `_optional_module()` helper — DRY, and the redefinition goes away.
      - WebSocket stream bridge (16): an `object()` sentinel widened
        `run_in_executor`'s result to `object`, so `(event_type, data)` could
        not unpack and the whole dispatch lost its types. Replaced with a typed
        `_next_event()` returning `BackendStreamEvent | None`; the three
        string-payload branches then collapse into one.
      - Dataset name literal (1): `DatasetName` alias in `src/eval/config.py`,
        so config and runner key the same closed set.
- [x] **One scoped exception, written down**: `attr-defined` off for
      `src.llm_handler.adapters.*`. Two better fixes were tried first (real SDK
      types under TYPE_CHECKING; structural Protocols for client and responses)
      and both are recorded in `pyproject.toml` with why they fail. Config, not
      `Any` in a signature and not `# type: ignore` at six call sites.

## Tranche 11 — wire the last two strategies  ✔ (ADR 0009)
The two levers ADR 0004 left recognised-but-unbuildable, plus the robustness
each one was deferred *for*. Closes the last open items of issue #16.

- [x] E1 `ChromaVectorStore.revision` — a write counter, bumped by `upsert` and
      `delete_by_doc_id`. The rejected alternative was explicit invalidation
      from every mutation site, which is correct only until someone adds a
      mutation site.
- [x] E2 `ChunkCorpus` (`src/retrieval/corpus.py`) — materialises the whole
      collection lazily, re-reads it when the revision moves, and returns the
      same snapshot object while nothing has changed so the BM25 index can
      cache on identity. Cold start deliberately starts at `None`, not at an
      empty snapshot: a persistent collection restored from disk is at
      revision 0 *and* non-empty.
- [x] E3 `all_chunk_texts()` → `all_chunks() -> dict[str, Chunk]`. The
      text-only shape is what made hybrid drop citations — a sparse-only hit
      had no filename to show. ADR 0004 accepted that "while the lever is off
      by default"; wiring it is when that expires.
- [x] E4 `BM25HybridRetriever` takes the store, not a snapshot. Degrades to
      dense-only instead of raising `ZeroDivisionError` on an empty or
      untokenizable corpus (an empty index is the normal state of a fresh
      deployment), and drops zero-score BM25 candidates so a corpus smaller
      than `bm25_top_k` does not hand every chunk to the generator.
- [x] E5 `RefusalHandler.should_refuse` reads the best score, not
      `candidates[0]`. Hybrid orders by fused rank; a BM25-only hit carries
      0.0, so the gate would have refused answerable questions. Identical
      behaviour for dense / reranked / multi-query, all of which sort.
- [x] E6 `QueryRewriter.expand` catches every provider failure and falls back
      to the original query. Only `JSONDecodeError` was guarded, so a rate
      limit failed the user's whole question over a *recall optimisation*.
- [x] E7 Rewrite spend is logged at the `MultiQueryRetriever` seam rather than
      returned — the seam stays `retrieve(query, top_k)`, on the same
      verified-zero-readers reasoning that retired `rewriter_cost_usd`.
- [x] E8 `build_retrieval_plan` wires all four strategies and raises when
      `multi_query` has no configured model or handler. `QueryRewriter(None)`
      is a legal pass-through, so the alternative is a deployment that thinks
      it has recall it does not have.
- [x] E9 `RETRIEVER_STRATEGY` and `QUERY_REWRITER_MODEL` read from the
      environment; hybrid constants and `MAX_QUERY_EXPANSIONS` single-sourced
      into `src/config.py`. ADR 0004 claimed promotion "by configuration, not
      a code change" while the selector was a tracked literal.
- [x] E10 Eval harness simplified: hybrid is built in `build_pipeline` like
      every other lever, and the two divergent corpus-construction sites in
      `ingest()` are gone.

## Out of scope / deferred
- **CI workflow changes** (shared infrastructure). Verified read-only that the
  new `pyproject.toml` does not affect it: CI installs with
  `uv pip install --system -r requirements.txt`, which ignores the file.
- **`ALLOWED_ORIGINS` in `docker-compose.prod.yml`** defaults to the nginx
  origin. A deployment on any other host must set it.

## Review
Thirteen commits, each with tests green. Four user-visible defects fixed that
were not in the original review: eval-run progress frozen at 0.0; ingestion
crashing on any document containing repeated text; `python -m src.api.main`
starting nothing; and the production compose file leaving CORS at `*`. The
architecture doc CLAUDE.md names as the source of truth now describes the
codebase that exists. Test count 320 → 514.
