"""Evaluation metrics for RAG systems."""
import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def compute_relevance_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze retrieval quality from result scores."""
    if not results:
        return {"avg_score": 0, "max_score": 0, "min_score": 0}
    scores = [r["score"] for r in results]
    return {"avg_score": round(np.mean(scores), 4), "max_score": round(max(scores), 4), "min_score": round(min(scores), 4)}


def generate_report(pipeline_stats: Dict[str, Any]) -> str:
    lines = ["# RAG Pipeline Report", f"- Documents: {pipeline_stats.get('documents', 0)}",
             f"- Chunks: {pipeline_stats.get('chunks', 0)}"]
    return "\n".join(lines)
