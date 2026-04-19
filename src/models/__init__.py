"""
SQLModel table models for the RAG chat-history persistence layer.

RAG Pipeline Position:
  This package is the DATA MODEL layer for SQLite persistence.

  Document -> Chunks -> Embeddings -> Vector Store -> Retrieval -> Generator
                                          |
                              [MODELS] <-> SQLite (via SQLModel + SQLAlchemy)

What concept it teaches:
  Centralising all model imports in __init__.py guarantees that every table
  class is registered with SQLModel.metadata before create_all() runs.
  SQLModel.metadata only knows about tables whose Python classes have been
  imported into the process.

Why this approach over alternatives:
  If create_db_and_tables() imported models inline, a developer could add a
  new model file and forget to add it here — silently missing its table.
  Having __init__.py as the single import manifest makes omissions obvious.

Where it fits in the RAG pipeline:
  src/database.py imports from this package before calling
  SQLModel.metadata.create_all(engine), ensuring all four tables are created.
"""

from __future__ import annotations

# WHY: These imports do two things:
#   1. Make the classes available via `from src.models import Conversation`
#   2. Register each SQLModel table class with SQLModel.metadata so that
#      create_db_and_tables() can call SQLModel.metadata.create_all() and
#      find all four tables.
#
# CRITICAL: Every new model file MUST be added here or its table will never
#           be created and tests will fail with "no such table" errors.
from src.models.conversation import Conversation
from src.models.document import DocumentRecord
from src.models.evaluation import MessageEvaluation
from src.models.message import Message, MessageSource

__all__ = [
    "Conversation",
    "Message",
    "MessageSource",
    "DocumentRecord",
    "MessageEvaluation",
]
