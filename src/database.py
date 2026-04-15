"""
Database engine factory, table initialisation, and session dependency.

RAG Pipeline Position:
  This module sits between the model definitions and every API route that
  needs to read or write persistent data.

  [MODELS] -> [DATABASE] -> API Routes -> Response

What concept it teaches:
  Three distinct responsibilities kept in three functions:
    1. get_engine()          — create + configure the SQLAlchemy engine once
    2. create_db_and_tables() — run DDL (CREATE TABLE IF NOT EXISTS) at startup
    3. get_session()         — yield a short-lived Session per HTTP request

  This separation means tests can call get_engine("sqlite://") for an
  in-memory DB without touching any HTTP layer.

Why this approach over alternatives:
  A global module-level engine would be created at import time, which makes
  testing (with a different DB URL) require monkey-patching. Passing the engine
  explicitly (dependency injection) keeps tests clean.

Where it fits in the RAG pipeline:
  src/api/main.py calls get_engine(SQLITE_URL) + create_db_and_tables() in the
  lifespan startup hook, then injects get_session via FastAPI Depends() into
  every route that needs DB access.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, event, text
from sqlmodel import Session, SQLModel, create_engine

logger = logging.getLogger(__name__)


def get_engine(url: str) -> Engine:
    """
    Create a SQLAlchemy engine configured for the given database URL.

    For SQLite URLs the engine is configured with:
      - check_same_thread=False  so FastAPI's threadpool workers can share it
      - PRAGMA foreign_keys=ON   fired on every new connection via an event
                                 listener so ON DELETE CASCADE works in SQLite

    Args:
        url: SQLAlchemy database URL. Use "sqlite://" for an in-memory DB
             (tests), or "sqlite:///path/to/file.db" for a persistent DB.

    Returns:
        Configured SQLAlchemy Engine instance.

    Example:
        >>> engine = get_engine("sqlite://")           # in-memory (tests)
        >>> engine = get_engine(config.SQLITE_URL)     # file-backed (production)
    """
    # PATTERN: Only pass SQLite-specific connect_args when the URL is a SQLite
    #          URL. Other backends (Postgres, MySQL) don't accept check_same_thread.
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        # WHY: FastAPI runs route handlers in a thread pool. Without
        #      check_same_thread=False, SQLite raises ProgrammingError if a
        #      connection created in one thread is used by another.
        connect_args["check_same_thread"] = False

    engine = create_engine(url, connect_args=connect_args)

    if url.startswith("sqlite"):
        _attach_foreign_key_pragma(engine)

    logger.info("Database engine created: %s", url)
    return engine


def _attach_foreign_key_pragma(engine: Engine) -> None:
    """
    Attach an event listener that enables foreign key enforcement on SQLite.

    WHY: SQLite disables foreign-key constraints (including ON DELETE CASCADE)
         by default. The setting must be turned ON per connection, not once
         globally. SQLAlchemy's 'connect' event fires every time a new raw
         DBAPI connection is opened, guaranteeing the PRAGMA is set before any
         queries run — including after pool recycling.

    CRITICAL: Without this, deleting a Conversation leaves orphan Messages
              and MessageSources in the database. There is no error — the
              DELETE just silently ignores the FK relationships.

    Args:
        engine: The SQLAlchemy engine to attach the listener to.
    """

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_db_and_tables(engine: Engine) -> None:
    """
    Create all SQLModel tables in the database if they do not already exist.

    This function imports every model class from src.models to ensure they are
    all registered with SQLModel.metadata before calling create_all(). If a
    model class has not been imported, its table will not be created.

    Args:
        engine: The SQLAlchemy engine pointing to the target database.

    Note:
        This is idempotent — safe to call on every application startup.
        Existing tables and data are left unchanged.

    Example:
        >>> engine = get_engine(SQLITE_URL)
        >>> create_db_and_tables(engine)   # called once in the lifespan hook
    """
    # WHY: Importing here (rather than at module top level) avoids a circular
    #      import if any model file imports from database.py in the future.
    #      It also makes the dependency explicit: "to create tables, I need
    #      the model classes loaded."
    #
    # CRITICAL: src.models.__init__ imports all four model classes. If you add
    #           a new model, add its import to src/models/__init__.py or its
    #           table will never be created.
    import src.models  # noqa: F401 — side-effect import registers all tables

    SQLModel.metadata.create_all(engine)
    logger.info("Database tables created (or already exist).")


def get_session(engine: Engine) -> Generator[Session, None, None]:
    """
    Yield a SQLModel Session for use as a FastAPI dependency.

    FastAPI calls next() on this generator to get the Session before the
    route handler runs, then resumes it (running the finally block) after
    the response is sent. This guarantees the session is always closed, even
    if the handler raises an exception.

    Args:
        engine: The SQLAlchemy engine to open the session against.

    Yields:
        An open SQLModel Session bound to the given engine.

    Example (FastAPI route)::

        from typing import Annotated
        from fastapi import Depends
        from sqlmodel import Session

        SessionDep = Annotated[Session, Depends(lambda: get_session(engine))]

        @router.get("/conversations")
        def list_conversations(session: SessionDep) -> list[Conversation]:
            return session.exec(select(Conversation)).all()

    PATTERN: We do NOT commit inside get_session. Route handlers are
             responsible for calling session.commit() when they mutate data.
             get_session only handles Session lifecycle (open / close).
    """
    with Session(engine) as session:
        yield session
