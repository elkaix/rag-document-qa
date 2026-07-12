"""Production retriever selection — turn a config strategy into a Retriever.

RAG Pipeline Position:
    config strategy -> [build_retriever] -> Retriever -> QueryEngine

This is the seam through which a validated eval chain is promoted to production
by *configuration* (ADR 0004). `RAGBackend` reads `RETRIEVER_STRATEGY` from
config and calls this factory; the QueryEngine never knows which strategy it got.

Wiring status (step 4):
    - ``dense``    — wired (the default; the behaviour-preserving anchor).
    - ``reranked`` — wired (composes dense + a cross-encoder; cost-clean).
    - ``multi_query`` — deferred to step 4c (its rewriter-cost surfacing lands
      with its production wiring, so it never ships a path that silently drops
      cost).
    - ``hybrid``   — deferred: BM25 needs a live corpus kept in sync with
      ingestion/deletion, a genuinely new feature beyond this refactor.
"""

from __future__ import annotations

from src.retrieval.base import Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.reranker import CrossEncoderReranker, RerankingRetriever
from src.vector_store import ChromaVectorStore

_DEFERRED = {
    "hybrid": "needs a live BM25 corpus synced with ingestion (a new feature)",
    "multi_query": "lands in step 4c with its rewriter-cost surfacing",
}


def build_retriever(
    strategy: str,
    vector_store: ChromaVectorStore,
    rerank_over_fetch_n: int = 20,
) -> Retriever:
    """Construct the production Retriever for a config strategy.

    Args:
        strategy: One of ``dense`` or ``reranked`` (wired), or ``hybrid`` /
            ``multi_query`` (recognised but deferred — see module docstring).
        vector_store: The dense index every strategy is built over.
        rerank_over_fetch_n: Candidate width the reranked strategy over-fetches.

    Returns:
        A Retriever conforming to the seam.

    Raises:
        ValueError: If the strategy is unknown, or recognised but not yet wired
            for production.
    """
    dense = DenseRetriever(vector_store)
    if strategy == "dense":
        return dense
    if strategy == "reranked":
        return RerankingRetriever(
            inner=dense, reranker=CrossEncoderReranker(), over_fetch_n=rerank_over_fetch_n,
        )
    if strategy in _DEFERRED:
        raise ValueError(
            f"Retriever strategy {strategy!r} is validated in the eval harness but "
            f"not yet wired for production ({_DEFERRED[strategy]}); see ADR 0004."
        )
    raise ValueError(f"Unknown retriever strategy: {strategy!r}")
