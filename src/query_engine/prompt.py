"""Prompt and context assembly — the single source of answer instructions.

RAG Pipeline Position:
    retrieved chunks + question -> [PROMPT] -> (system, user) -> LLM

Design Decision:
    Before step 4 the answer system prompt existed in three diverged copies (a
    plain one on the sync query path, a Markdown one on the streaming path, and a
    third in the eval harness). This module makes the **Markdown** prompt the one
    answer prompt for every path — it is what the frontend's renderer expects,
    and the eval harness must measure the shipped prompt. Context is
    filename-prefixed everywhere so citations survive. See ADR 0004.
"""

from __future__ import annotations

from src.vector_store import SearchResult

# The single answer system prompt for sync, streaming, and eval paths.
ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question based solely on the "
    "provided context. If the context does not contain enough information, say so.\n\n"
    "Format your response using Markdown for readability:\n"
    "- Use ## for main sections and ### for sub-sections (max 3 levels)\n"
    "- Use **bold** for key terms and important concepts\n"
    "- Use bullet points (-) for lists of related items\n"
    "- Use numbered lists (1.) for sequential steps\n"
    "- Use `inline code` for technical terms, parameters, or commands\n"
    "- Use fenced code blocks (```language) for code snippets\n"
    "- Use > blockquotes for notable quotes from the context\n"
    "- Keep paragraphs short (2-3 sentences max)\n"
    "- Add blank lines between sections for visual breathing room\n"
    "Do NOT use # (h1) headings. Start directly with content or ## sections."
)

# The planning-pass prompt (streaming only). It asks for an outcome-oriented
# reasoning summary, not raw chain-of-thought, so users are not shown uncertain
# intermediate beliefs they might mistake for the answer.
REASONING_SYSTEM_PROMPT = (
    "You are the planning step of a retrieval-augmented Q&A system. "
    "In 3-5 concise sentences, summarise how you will construct the "
    "answer using the retrieved excerpts. Cover:\n"
    "1) What the user is asking, resolving any ambiguity explicitly.\n"
    "2) Which excerpts are most relevant and the gist of their support.\n"
    "3) Any gaps or conflicts the reader should be aware of.\n"
    "4) The shape of the answer you will give next.\n"
    "Stay factual and outcome-oriented — describe the plan, do not "
    "verbalise stream-of-consciousness reasoning. No markdown headings, "
    "no bullet lists, no preamble. Do NOT produce the final answer."
)

# Shown (and returned) when the index has no documents to retrieve from.
NO_DOCUMENTS_ANSWER = "No documents indexed yet. Please upload documents first."


def build_context(results: list[SearchResult]) -> str:
    """Join retrieved chunks into a filename-prefixed context block.

    Args:
        results: Retrieved chunks, best first.

    Returns:
        A ``"[filename] content"`` block per chunk, separated by blank lines —
        the prefix is what lets the model (and downstream citations) attribute
        each passage to its source document.
    """
    return "\n\n".join(
        f"[{r.metadata.get('filename', 'unknown')}] {r.content}" for r in results
    )


def build_answer_user_prompt(context: str, question: str) -> str:
    """Assemble the answer-pass user message from context and question."""
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def build_reasoning_user_prompt(context: str, question: str) -> str:
    """Assemble the planning-pass user message (summary only, no answer)."""
    return (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Reasoning plan (summary only, do not answer):"
    )
