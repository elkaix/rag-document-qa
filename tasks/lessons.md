# Lessons — rag-qa

Repo-specific rules learned during work. Trigger → mistake → rule.

## Facade attribute name collisions

- **Trigger:** adding a new attribute to a class that already holds several
  (e.g. `RAGBackend`).
- **Mistake:** named the injected QueryEngine `self.engine`, shadowing the
  existing `self.engine` (the SQLAlchemy `Engine`). `Session(self.engine)` then
  got a QueryEngine → `AttributeError: 'QueryEngine' object has no attribute
  'connect'`, failing 17 tests at once.
- **Rule:** before adding `self.<name>` to an existing class, grep the class for
  `self.<name>` and for external `.<name>` reads in tests. Here `test_backend.py`
  read `backend.engine` as the SQLAlchemy engine. Chose `self.query_engine`.

## Removing `Any` surfaces real narrowing needs

- **Trigger:** replacing an `Any`-typed value with a precise union (e.g.
  `Iterator[tuple[str, Any]]` → `Iterator[tuple[str, str | StreamResult]]`).
- **Mistake:** the improvement pushed a union to every consumer; the facade loop
  then failed mypy (`str | StreamResult` where a `str`/`StreamResult` was needed).
- **Rule:** when tightening a boundary type, immediately mypy the *consumers*.
  Narrow with `isinstance` (not a string tag mypy can't follow) and assert an
  internal invariant to bind an "always set" value — no `# type: ignore`.

## Tooling: ruff yes, black no; mypy has a known SDK-seam baseline

- **Rule:** ruff is authoritative on changed files. Do NOT run `black` — the repo
  predates it and untouched files fail it (it would explode the compact
  trailing-comma style). mypy shows pre-existing errors in `llm_handler/adapters`
  (intentional loose `object` SDK seam, ADR 0002) and SQLModel column
  descriptors (`.desc()`/`.contains()`); judge only NEW errors in changed files.
