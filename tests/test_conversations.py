"""Tests for src.conversations — the store and the history window.

These exercise the modules directly, with only a database. While this code lived
inside RAGBackend, reaching it meant constructing the whole RAG facade: a Chroma
collection, three LLM handlers, a retriever and a query engine.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session

from src.conversations import ConversationHistory, ConversationStore
from src.conversations.history import PLACEHOLDER_TITLE, _truncate_on_word_boundary
from src.conversations.shaping import conversation_summary, message_dict, source_dict
from src.database import create_db_and_tables, get_engine
from src.models.conversation import Conversation
from src.models.message import Message, MessageSource


@pytest.fixture
def session_factory():
    engine = get_engine("sqlite://")
    create_db_and_tables(engine)
    return lambda: Session(engine)


@pytest.fixture
def store(session_factory) -> ConversationStore:
    return ConversationStore(session_factory)


@pytest.fixture
def history(session_factory) -> ConversationHistory:
    return ConversationHistory(session_factory)


class TestThreadLifecycle:
    def test_create_returns_a_summary(self, store):
        summary = store.create("My thread")
        assert summary["title"] == "My thread"
        assert summary["pinned"] is False
        assert summary["id"]

    def test_get_unknown_returns_none(self, store):
        assert store.get("nope") is None

    def test_update_unknown_returns_none(self, store):
        assert store.update("nope", title="x") is None

    def test_delete_unknown_returns_false(self, store):
        assert store.delete("nope") is False

    def test_delete_cascades_to_messages(self, store, history, session_factory):
        conv_id = store.create()["id"]
        history.save_message(conv_id, "user", "hi")

        assert store.delete(conv_id) is True

        with session_factory() as session:
            assert session.get(Conversation, conv_id) is None

    def test_update_touches_only_supplied_fields(self, store):
        conv_id = store.create("original")["id"]
        store.update(conv_id, pinned=True)
        summary = store.get(conv_id)
        assert summary["title"] == "original"
        assert summary["pinned"] is True

    def test_list_puts_pinned_first(self, store):
        first = store.create("older")["id"]
        store.create("newer")
        store.update(first, pinned=True)
        assert store.list_all()[0]["id"] == first


class TestSearch:
    def test_no_match_returns_empty(self, store):
        store.create("unrelated")
        assert store.search("kangaroo") == []

    def test_matches_on_title(self, store):
        conv_id = store.create("kangaroo notes")["id"]
        assert [c["id"] for c in store.search("kangaroo")] == [conv_id]

    def test_matches_on_message_body(self, store, history):
        conv_id = store.create("untitled")["id"]
        history.save_message(conv_id, "user", "about kangaroo biology")
        assert [c["id"] for c in store.search("kangaroo")] == [conv_id]

    def test_a_thread_matching_twice_appears_once(self, store, history):
        conv_id = store.create("kangaroo notes")["id"]
        history.save_message(conv_id, "user", "kangaroo one")
        history.save_message(conv_id, "user", "kangaroo two")
        assert [c["id"] for c in store.search("kangaroo")].count(conv_id) == 1


class TestExportAndSharing:
    def test_export_unknown_returns_none(self, store):
        assert store.export_markdown("nope") is None

    def test_export_renders_roles(self, store, history):
        conv_id = store.create("Thread")["id"]
        history.save_message(conv_id, "user", "question?")
        history.save_message(conv_id, "assistant", "answer.")

        markdown = store.export_markdown(conv_id)

        assert markdown.startswith("# Thread")
        assert "**User:** question?" in markdown
        assert "**Assistant:** answer." in markdown

    def test_share_token_for_unknown_returns_none(self, store):
        assert store.create_share_token("nope") is None

    def test_a_minted_token_resolves_back_to_the_thread(self, store):
        conv_id = store.create("Shared")["id"]
        token = store.create_share_token(conv_id)
        assert store.get_by_share_token(token)["id"] == conv_id

    def test_an_unknown_token_resolves_to_none(self, store):
        assert store.get_by_share_token("not-a-token") is None

    def test_tokens_are_not_predictable(self, store):
        a = store.create_share_token(store.create()["id"])
        b = store.create_share_token(store.create()["id"])
        assert a != b and len(a) == 36


class TestSaveMessage:
    def test_returns_an_id_usable_after_commit(self, store, history):
        conv_id = store.create()["id"]
        msg_id = history.save_message(conv_id, "user", "hello")
        assert msg_id
        assert store.get(conv_id)["messages"][0]["id"] == msg_id

    def test_persists_sources(self, store, history):
        conv_id = store.create()["id"]
        history.save_message(
            conv_id, "assistant", "answer", model="m",
            sources=[{
                "doc_id": "d", "chunk_id": "c", "filename": "f.txt",
                "score": 0.5, "excerpt": "e",
            }],
        )
        sources = store.get(conv_id)["messages"][0]["sources"]
        assert sources == [{
            "doc_id": "d", "chunk_id": "c", "filename": "f.txt",
            "score": 0.5, "excerpt": "e",
        }]

    def test_missing_source_fields_fall_back(self, store, history):
        conv_id = store.create()["id"]
        history.save_message(conv_id, "assistant", "a", sources=[{}])
        source = store.get(conv_id)["messages"][0]["sources"][0]
        assert source["doc_id"] == "" and source["score"] == 0.0

    def test_bumps_the_parent_thread(self, store, history):
        conv_id = store.create()["id"]
        before = store.get(conv_id)["updated_at"]
        history.save_message(conv_id, "user", "hi")
        assert store.get(conv_id)["updated_at"] >= before


class TestSlidingWindow:
    def _turn(self, session_factory, conv_id, role, content, offset):
        with session_factory() as session:
            session.add(Message(
                conversation_id=conv_id, role=role, content=content,
                created_at=datetime.now(timezone.utc) + timedelta(seconds=offset),
            ))
            session.commit()

    def test_empty_thread_yields_nothing(self, store, history):
        assert history.sliding_window(store.create()["id"]) == []

    def test_a_dangling_question_is_excluded(self, store, history, session_factory):
        """A turn whose generation failed must not reach the next prompt."""
        conv_id = store.create()["id"]
        self._turn(session_factory, conv_id, "user", "answered", 0)
        self._turn(session_factory, conv_id, "assistant", "reply", 1)
        self._turn(session_factory, conv_id, "user", "never answered", 2)

        window = history.sliding_window(conv_id)

        assert [m["content"] for m in window] == ["answered", "reply"]

    def test_consecutive_same_role_messages_are_skipped(
        self, store, history, session_factory
    ):
        conv_id = store.create()["id"]
        self._turn(session_factory, conv_id, "user", "first", 0)
        self._turn(session_factory, conv_id, "user", "second", 1)
        self._turn(session_factory, conv_id, "assistant", "reply", 2)

        window = history.sliding_window(conv_id)

        assert [m["content"] for m in window] == ["second", "reply"]

    def test_respects_the_pair_limit(self, store, history, session_factory):
        conv_id = store.create()["id"]
        for i in range(4):
            self._turn(session_factory, conv_id, "user", f"q{i}", i * 2)
            self._turn(session_factory, conv_id, "assistant", f"a{i}", i * 2 + 1)

        window = history.sliding_window(conv_id, max_pairs=2)

        assert [m["content"] for m in window] == ["q2", "a2", "q3", "a3"]


class TestAutoTitle:
    def test_replaces_only_the_placeholder(self, store, history):
        conv_id = store.create(PLACEHOLDER_TITLE)["id"]
        history.auto_title(conv_id, "What is RAG?")
        titled = store.get(conv_id)["title"]
        assert titled == "What is RAG?"

        history.auto_title(conv_id, "A different question")
        assert store.get(conv_id)["title"] == titled

    def test_leaves_a_user_chosen_title_alone(self, store, history):
        conv_id = store.create("My own title")["id"]
        history.auto_title(conv_id, "What is RAG?")
        assert store.get(conv_id)["title"] == "My own title"

    def test_unknown_conversation_is_a_no_op(self, history):
        history.auto_title("nope", "anything")


class TestTruncateOnWordBoundary:
    def test_short_titles_are_unchanged(self):
        assert _truncate_on_word_boundary("short") == "short"

    def test_cuts_at_a_space(self):
        title = _truncate_on_word_boundary("word " * 40)
        assert title.endswith("...")
        assert "  " not in title

    def test_a_single_long_word_is_cut_hard(self):
        title = _truncate_on_word_boundary("x" * 200)
        assert title == "x" * 60 + "..."


class TestShaping:
    def test_summary_has_no_messages_key(self):
        conv = Conversation(title="t")
        assert "messages" not in conversation_summary(conv)

    def test_message_dict_embeds_shaped_sources(self):
        msg = Message(conversation_id="c", role="user", content="hi")
        src = MessageSource(
            message_id=msg.id, doc_id="d", chunk_id="ch",
            filename="f", score=0.5, excerpt="e",
        )
        shaped = message_dict(msg, [src])
        assert shaped["sources"] == [source_dict(src)]
