"""Row-to-dict shapes for conversation data.

RAG Pipeline Position:
    SQLite rows -> [SHAPING] -> dicts the API serialises

What concept it teaches:
    One definition per wire shape. The conversation summary was written out at
    five call sites and the message shape at two; a field added to one and
    forgotten at another is a silent inconsistency, which is exactly the bug
    class that produced differing source-citation shapes on the two query paths.

Design Decision:
    Free functions taking ORM rows, not methods. They touch no session and hold
    no state, so they are testable with constructed rows alone.
"""

from __future__ import annotations

from typing import Any

from src.models.conversation import Conversation
from src.models.message import Message, MessageSource


def conversation_summary(conv: Conversation) -> dict[str, Any]:
    """Shape one conversation without its messages, for list and detail views.

    Args:
        conv: The conversation row.

    Returns:
        Dict with id, title, pinned, created_at, updated_at.
    """
    return {
        "id": conv.id,
        "title": conv.title,
        "pinned": conv.pinned,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
    }


def source_dict(source: MessageSource) -> dict[str, Any]:
    """Shape one cited chunk as the frontend renders it.

    Args:
        source: The message-source row.

    Returns:
        Dict with doc_id, chunk_id, filename, score, excerpt.
    """
    return {
        "doc_id": source.doc_id,
        "chunk_id": source.chunk_id,
        "filename": source.filename,
        "score": source.score,
        "excerpt": source.excerpt,
    }


def message_dict(msg: Message, sources: list[MessageSource]) -> dict[str, Any]:
    """Shape one message together with the chunks it cited.

    Args:
        msg: The message row.
        sources: The message's cited chunks, already loaded.

    Returns:
        Dict with id, role, content, model, created_at, sources.
    """
    return {
        "id": msg.id,
        "role": msg.role,
        "content": msg.content,
        "model": msg.model,
        "created_at": msg.created_at.isoformat(),
        "sources": [source_dict(s) for s in sources],
    }


def conversation_detail(
    conv: Conversation, messages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Shape a conversation with its messages attached.

    Args:
        conv: The conversation row.
        messages: Already-shaped message dicts, in chronological order.

    Returns:
        The summary shape plus a ``messages`` list.
    """
    return {**conversation_summary(conv), "messages": messages}
