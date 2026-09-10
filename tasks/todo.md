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

## Out of scope / deferred
- **black**: would reformat 83 of 138 files. The repo was never
  black-formatted; running it now would bury this work's diff. `pyproject.toml`
  records line-length so a future `black .` is one deliberate commit.
- **mypy**: ~19 remaining errors are the known SQLModel/SQLAlchemy typing gap
  (`Model.field.desc()`, `.contains()`, `.in_()` — mypy sees the field's value
  type, not the InstrumentedAttribute). Pre-existing pattern, unchanged by this
  work. The two errors this work introduced (duplicated Protocols in
  `composition.py`) are fixed.
- Pushing to PR #21 (outward-facing; ask first)
- CI workflow changes (shared infrastructure)

## Review
Nine commits, each with tests green. Two user-visible bugs fixed that were not
in the original review: eval-run progress frozen at 0.0, and ingestion crashing
on any document containing repeated text. Test count 320 → 508.
