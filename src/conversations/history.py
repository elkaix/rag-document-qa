"""Message persistence and the history window fed back to the LLM.

RAG Pipeline Position:
    Answer -> [HISTORY] -> SQLite -> sliding window -> next turn's prompt
                  ^^^
    Writing a turn and reading back the prior turns are the same concern: both
    depend on how a thread is laid out in the message table.

What concept it teaches:
    Why "the last N messages" is the wrong window. Only *completed* exchanges
    belong in an LLM's context; a dangling question would ask the model to
    continue from a turn that never got an answer.

Design Decision:
    Placeholder title replacement lives here rather than in the store because it
    is driven by the first user message, not by a thread-lifecycle event.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlmodel import Session, col, select

from src.config import MAX_TITLE_LENGTH, SLIDING_WINDOW_SIZE
from src.models.conversation import Conversation
from src.models.message import Message, MessageSource

logger = logging.getLogger(__name__)

PLACEHOLDER_TITLE = "New Chat"


class ConversationHistory:
    """Writes turns into a thread and reads back the context for the next one."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        """Wire the history to its database.

        Args:
            session_factory: Returns a fresh short-lived Session.
        """
        self._session = session_factory

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model: str | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> str:
        """Persist one message, its cited sources, and touch the parent thread.

        Args:
            conversation_id: UUID of the parent conversation.
            role: ``user`` or ``assistant``.
            content: Message text.
            model: Generating model name, for assistant messages.
            sources: Source-citation dicts, for assistant messages.

        Returns:
            The new message's UUID.

        WHY the id is captured before commit: ``Message.id`` is assigned at
            construction by a uuid4 default factory, and SQLAlchemy expires every
            attribute after commit — reading ``msg.id`` afterwards would raise
            DetachedInstanceError.

        WHY sources are separate rows rather than a JSON column: it keeps the
            schema normalised and lets a source be queried on its own.
        """
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            model=model,
        )
        msg_id = msg.id

        with self._session() as session:
            session.add(msg)

            for src in sources or []:
                session.add(
                    MessageSource(
                        message_id=msg_id,
                        doc_id=src.get("doc_id", ""),
                        chunk_id=src.get("chunk_id", ""),
                        filename=src.get("filename"),
                        score=src.get("score", 0.0),
                        excerpt=src.get("excerpt", ""),
                    )
                )

            conv = session.get(Conversation, conversation_id)
            if conv:
                conv.updated_at = datetime.now(UTC)
                session.add(conv)

            session.commit()

        return msg_id

    def sliding_window(
        self,
        conversation_id: str,
        max_pairs: int = SLIDING_WINDOW_SIZE,
    ) -> list[dict[str, str]]:
        """Return the last N *completed* exchanges, oldest first.

        Args:
            conversation_id: UUID of the conversation.
            max_pairs: Maximum user/assistant pairs to include.

        Returns:
            ``{"role", "content"}`` dicts in chronological order.

        WHY only completed pairs: the window is fed to the answer prompt, which
            appends the current question as its own user turn. A dangling user
            message — a prior turn whose generation failed before the reply was
            persisted — would show the model a question with no answer, and it
            may repeat it or get confused. Any other adjacency (user→user,
            assistant→assistant, a lone message) is skipped for the same reason.
        """
        with self._session() as session:
            messages = session.exec(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(col(Message.created_at))
            ).all()

            # WHY inside the session scope: building the window touches message
            #     attributes, which would be expired outside it.
            paired: list[Message] = []
            i = 0
            while i < len(messages) - 1:
                if messages[i].role == "user" and messages[i + 1].role == "assistant":
                    paired.extend((messages[i], messages[i + 1]))
                    i += 2
                else:
                    i += 1

            window = paired[-(max_pairs * 2) :]
            return [{"role": m.role, "content": m.content} for m in window]

    def auto_title(self, conversation_id: str, first_query: str) -> None:
        """Name an untitled thread after its first question.

        Args:
            conversation_id: UUID of the conversation.
            first_query: The user's first question.

        WHY the placeholder check: a title the user can see must never be
            overwritten by a later turn. Only the untouched default is replaced.
        """
        with self._session() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None or conv.title != PLACEHOLDER_TITLE:
                return

            conv.title = _truncate_on_word_boundary(first_query.strip())
            conv.updated_at = datetime.now(UTC)
            session.add(conv)
            session.commit()


def _truncate_on_word_boundary(title: str) -> str:
    """Shorten a title to MAX_TITLE_LENGTH without cutting mid-word.

    Args:
        title: The candidate title.

    Returns:
        The title unchanged when short enough, otherwise a truncation ending in
        an ellipsis. A single over-long word is cut hard, since there is no
        boundary to fall back to.
    """
    if len(title) <= MAX_TITLE_LENGTH:
        return title

    truncated = title[:MAX_TITLE_LENGTH]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        return truncated[:last_space] + "..."
    return truncated + "..."
