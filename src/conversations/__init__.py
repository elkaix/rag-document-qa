"""Conversation persistence — chat threads, their messages, and their sources.

RAG Pipeline Position:
    Answer -> [CONVERSATIONS] -> SQLite rows -> sidebar, history, sharing

What this package holds:
    ``ConversationStore`` owns every read and write of conversations, messages
    and message sources: creation, listing, the three-level load, updates,
    deletion, search, Markdown export, share tokens, the sliding-history window
    used to give the LLM context, and the auto-title rule.

Why this is its own module:
    These were 204 code lines inside the 1265-line RAGBackend facade, written as
    inline SQLModel queries with no module between them and the database — 17 of
    the 19 ``select(`` calls in ``src/`` were in that one file. Deleting the
    facade would not have removed the complexity: it would have reappeared
    across nine route handlers, with the message helpers duplicated between the
    WebSocket handler and the conversation routes.
"""

from src.conversations.history import ConversationHistory
from src.conversations.store import ConversationStore

__all__ = ["ConversationHistory", "ConversationStore"]
