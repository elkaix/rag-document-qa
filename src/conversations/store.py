"""Conversation storage — chat threads and their lifecycle.

RAG Pipeline Position:
    Answer -> [CONVERSATION STORE] -> SQLite -> sidebar / history / sharing

What concept it teaches:
    A store with one dependency. Every method here needs a database session and
    nothing else — no vector store, no LLM, no retrieval. That is what makes
    these behaviours testable on their own, which they were not while they lived
    inside the RAG facade alongside query orchestration.

Design Decision:
    The session factory is injected rather than an engine, so this module shares
    the facade's session-per-operation policy — short-lived sessions, tight
    transaction scope, none shared across requests — instead of inventing a
    second one.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from sqlmodel import Session, select

from src.conversations.shaping import (
    conversation_detail,
    conversation_summary,
    message_dict,
)
from src.models.conversation import Conversation
from src.models.message import Message, MessageSource

logger = logging.getLogger(__name__)


class ConversationStore:
    """Reads and writes chat threads, their messages, and their cited sources."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        """Wire the store to its database.

        Args:
            session_factory: Returns a fresh short-lived Session.
        """
        self._session = session_factory

    # ------------------------------------------------------------------ #
    # Thread lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def create(self, title: str = "New Chat") -> dict[str, Any]:
        """Create a conversation.

        Args:
            title: Human-readable title. The default is the placeholder that
                :meth:`auto_title` is allowed to replace.

        Returns:
            The conversation summary.
        """
        conv = Conversation(title=title)
        with self._session() as session:
            session.add(conv)
            session.commit()
            session.refresh(conv)
            return conversation_summary(conv)

    def list_all(self) -> list[dict[str, Any]]:
        """Return every conversation, pinned first, then most recently updated.

        Returns:
            Conversation summaries in sidebar order.

        WHY pinned first: users pin threads so they stay at the top of the
            sidebar regardless of when they were last touched.
        """
        with self._session() as session:
            convs = session.exec(
                select(Conversation).order_by(
                    Conversation.pinned.desc(), Conversation.updated_at.desc()
                )
            ).all()
            return [conversation_summary(c) for c in convs]

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        """Return a conversation with its messages and each message's sources.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            The detail shape, or None when no such conversation exists.
        """
        with self._session() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                return None

            messages = session.exec(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at)
            ).all()

            shaped = []
            for msg in messages:
                sources = session.exec(
                    select(MessageSource).where(MessageSource.message_id == msg.id)
                ).all()
                shaped.append(message_dict(msg, list(sources)))

            return conversation_detail(conv, shaped)

    def update(
        self,
        conversation_id: str,
        title: str | None = None,
        pinned: bool | None = None,
    ) -> dict[str, Any] | None:
        """Change a conversation's title and/or pinned state.

        Args:
            conversation_id: UUID of the conversation.
            title: New title, when supplied.
            pinned: New pinned state, when supplied.

        Returns:
            The updated summary, or None when no such conversation exists.
        """
        with self._session() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                return None

            if title is not None:
                conv.title = title
            if pinned is not None:
                conv.pinned = pinned

            conv.updated_at = datetime.now(timezone.utc)
            session.add(conv)
            session.commit()
            session.refresh(conv)
            return conversation_summary(conv)

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation with its messages and sources.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            True when a conversation was deleted, False when none matched.

        WHY no explicit child deletes: the foreign keys declare ON DELETE
            CASCADE, and database.py enables ``PRAGMA foreign_keys=ON`` so
            SQLite honours them.
        """
        with self._session() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                return False
            session.delete(conv)
            session.commit()
        return True

    def search(self, query: str) -> list[dict[str, Any]]:
        """Find conversations whose title or any message contains a substring.

        Args:
            query: Search string.

        Returns:
            Matching conversation summaries, most recently updated first. A
            conversation matching on both title and body appears once.

        TRADE-OFF: SQL LIKE rather than full-text search. Adequate at this
            scale; FTS5 would be the production answer.
        """
        with self._session() as session:
            # WHY two queries and a set union rather than a JOIN: a JOIN over
            #     messages returns one row per matching message, so a thread
            #     with three hits would appear three times.
            by_title = session.exec(
                select(Conversation.id).where(Conversation.title.contains(query))
            ).all()
            by_message = session.exec(
                select(Message.conversation_id).where(Message.content.contains(query))
            ).all()

            matching_ids = set(by_title) | set(by_message)
            if not matching_ids:
                return []

            convs = session.exec(
                select(Conversation)
                .where(Conversation.id.in_(matching_ids))
                .order_by(Conversation.updated_at.desc())
            ).all()
            return [conversation_summary(c) for c in convs]

    # ------------------------------------------------------------------ #
    # Export and sharing                                                  #
    # ------------------------------------------------------------------ #

    def export_markdown(self, conversation_id: str) -> str | None:
        """Render a conversation as Markdown.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            A Markdown transcript, or None when no such conversation exists.
        """
        data = self.get(conversation_id)
        if data is None:
            return None

        lines = [f"# {data['title']}", "---", ""]
        for msg in data["messages"]:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"**{role_label}:** {msg['content']}")
            lines.append("")
        return "\n".join(lines)

    def create_share_token(self, conversation_id: str) -> str | None:
        """Mint a token granting read-only access to a conversation.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            The token, or None when no such conversation exists.

        SECURITY: UUID4 — opaque and unguessable. Anyone holding the token can
            read the thread, so it must not be sequential or derivable.
        """
        token = str(uuid.uuid4())
        with self._session() as session:
            conv = session.get(Conversation, conversation_id)
            if conv is None:
                return None
            conv.share_token = token
            session.add(conv)
            session.commit()
        return token

    def get_by_share_token(self, token: str) -> dict[str, Any] | None:
        """Return the conversation a share token points at.

        Args:
            token: The share token.

        Returns:
            The detail shape, or None when the token matches nothing.
        """
        with self._session() as session:
            conv = session.exec(
                select(Conversation).where(Conversation.share_token == token)
            ).first()
            if conv is None:
                return None
            # WHY capture the id inside the session: attributes are expired on
            #     exit, and get() opens a session of its own.
            conv_id = conv.id

        return self.get(conv_id)
