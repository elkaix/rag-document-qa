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

## Tranche 2 — C3 value types off the vendor
- [ ] Move `SearchResult` + `Document`/`Chunk` to a leaf value-types module
- [ ] Vector store owns the cosine invariant (stop re-asserting at 9 sites)

## Tranche 3 — C6 configuration
- [ ] Centralise env reads; `EVAL_RUNS_DIR` injectable not monkeypatched
- [ ] Single-source rerank widths + refusal defaults

## Tranche 4 — C1 retrieval composition
- [ ] Characterization test for current composition rule
- [ ] One composition module returning (Retriever, effective top_k)

## Tranche 5 — C4 eval run submission
- [ ] Submission module; route translates HTTP only

## Tranche 6 — C2 backend split
- [ ] Characterization tests for conversation + evaluation clusters
- [ ] Extract ConversationStore, Evaluator
- [ ] Adopt the DI seam in all route modules

## Tranche 7 — C5 parsing seam
- [ ] Characterization tests for PDF/DOCX/HTML (deps confirmed installed)
- [ ] Parsing seam + chunking split; test `_semantic_chunk`

## Out of scope
- Pushing to PR #21 (outward-facing; ask first)
- CI workflow changes (shared infrastructure)
