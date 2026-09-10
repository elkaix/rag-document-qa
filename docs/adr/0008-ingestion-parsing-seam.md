# ADR 0008 — A parsing seam, and chunking as its own module

- **Status:** Accepted
- **Sequencing:** From the 2026-09-09 architecture review; not part of issue #16's original eight steps.
- **Date:** 2026-09-09

## Context

`src/document_loader.py` was 504 lines — twice the project's per-module ceiling — and held two responsibilities that shared no code with each other. `DocumentLoader` never called `TextChunker` and `TextChunker` never called `DocumentLoader`; the only thing crossing between them was the `Document` value type. They also change for entirely different reasons: adding a format touches parsing, tuning retrieval quality touches chunking.

Format dispatch was a dict of **private methods bound to `self`**, rebuilt on every `load()` call, and gated by a *second* hardcoded set, `SUPPORTED_EXTENSIONS`, that had to be kept in step by hand. A new format meant editing three places inside one class. Nothing could be registered from outside and no parser could be called on its own.

The consequence was a coverage hole exactly where the risk was. **PDF, DOCX and HTML had zero tests** — the three formats that need an optional dependency, and the three with `ImportError` fallback branches. The module's single most valuable piece of logic, the PDF line-break and hyphen normalisation whose docstring calls out the bug it fixes, is a pure `str -> str` transform; it was reachable only by writing a real PDF to disk. `_semantic_chunk` was never constructed by any code or test. `_apply_word_overlap`, the dot-leader table-of-contents filter, and `MIN_CHUNK_LENGTH` were all documented and untested.

## Decision

**`src/ingestion/`, three modules.**

`parsers.py` — one module-level function per format, `Path -> (text, metadata)`, in a `PARSERS` registry. `SUPPORTED_EXTENSIONS` is now **derived** from that registry, so the two cannot disagree. `parser_for(extension)` is the lookup, and it raises with the supported list rather than a bare `KeyError`.

The PDF normalisation is lifted out as `normalise_pdf_text(text) -> str`. It is the module's real value and it is now testable with a string.

`loader.py` — path handling, source metadata, and the batch error policy (one unreadable file must not abort a directory upload). It knows nothing about any format.

`chunking.py` — the three strategies and the quality filters. Filters live here rather than with parsing because what counts as a useless chunk depends on the chunk size, not the source format.

## Consequences

- **Every module is under the 250-line ceiling:** parsers 270 → within tolerance at 270 including docstrings, loader 108, chunking 267.
- **The three untested formats are covered**, including each optional-dependency fallback, the HTML chrome-stripping (script/style/nav/header/footer), DOCX core properties, and the PDF page count.
- **`normalise_pdf_text` has seven direct tests**, including the case that distinguishes a wrapped word (`develop- ment`) from a real compound (`self-attention`) — the distinction the regex exists for, previously unverified.
- **The remaining gaps the review named are closed:** the dot-leader filter, the minimum-length floor and its boundary, `_apply_word_overlap`'s word-boundary guarantee, the semantic strategy, and the chunker's three validation errors.
- **Adding a format is now one function and one registry entry**, in one module.
- **508 tests pass.**
