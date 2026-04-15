"""Response generator — assembles context and produces answers.

The generator takes retrieved chunks and the user query to produce
a coherent answer. In a real RAG system, this would send the context
to an LLM (Claude, GPT, etc.).

This implementation uses a template-based approach for educational purposes:
it shows the CONTEXT ASSEMBLY pattern that all RAG systems use.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

RAG_PROMPT_TEMPLATE = """Based on the following document excerpts, answer the question.

--- DOCUMENT CONTEXT ---
{context}
--- END CONTEXT ---

Question: {query}

Answer:"""


class ResponseGenerator:
    """Generate answers from retrieved document chunks."""

    def __init__(self, template: str = RAG_PROMPT_TEMPLATE) -> None:
        self.template = template

    def assemble_context(self, results: List[Dict[str, Any]]) -> str:
        """Join retrieved chunks into a single context block."""
        parts = []
        for i, r in enumerate(results):
            parts.append(f"[Chunk {i+1} (score: {r['score']})]\n{r['chunk']['text']}")
        return "\n\n".join(parts)

    def generate(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a response using retrieved context."""
        context = self.assemble_context(results)
        prompt = self.template.format(context=context, query=query)

        # Educational: show the assembled prompt (in production, send to LLM)
        return {
            "query": query,
            "context": context,
            "prompt": prompt,
            "answer": f"Based on {len(results)} retrieved chunks, here is a summary of relevant information for: '{query}'. In a production system, this prompt would be sent to an LLM for a detailed answer.",
            "sources": [{"chunk_id": r["chunk"]["id"], "score": r["score"]} for r in results],
        }
