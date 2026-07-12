"""QueryEngine package — the shared retrieve->generate module (issue #16, step 4).

`QueryEngine` owns retrieval, prompt construction, generation, and telemetry
assembly behind a small interface (`ask`, `ask_stream`). Both the production
`RAGBackend` facade and the eval harness call it, so eval measures the shipped
pipeline. See [ADR 0004](../../docs/adr/0004-retriever-seam-and-query-engine.md).
"""

from __future__ import annotations

from src.query_engine.engine import QueryEngine, StreamResult

__all__ = ["QueryEngine", "StreamResult"]
