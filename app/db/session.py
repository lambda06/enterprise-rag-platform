"""
Async SQLAlchemy engine, session factory, and FastAPI dependency for the
Enterprise RAG Platform.

Architecture
------------
SQLAlchemy 2.0 introduced first-class async support via ``create_async_engine``
and ``AsyncSession``.  We use these throughout so that all database I/O can be
awaited inside FastAPI endpoint handlers and LangGraph nodes without blocking
the event loop.

Connection string
-----------------
``asyncpg`` is the async PostgreSQL driver.  SQLAlchemy requires the dialect
prefix ``postgresql+asyncpg://`` — the ``DATABASE_URL`` in ``.env`` is
expected to include this prefix (or we detect and rewrite it below so a plain
``postgresql://`` URL also works).

Usage in FastAPI
----------------
Declare ``db: AsyncSession = Depends(get_db)`` in any endpoint function.
The session is committed on success and rolled back + closed on any exception,
with no extra boilerplate in the handler.

Usage in LangGraph nodes
------------------------
Because nodes are async functions, they can ``async with async_session() as db``
directly without going through the FastAPI dependency injection mechanism.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _make_async_url(url: str) -> str:
    """
    Normalise the database URL for asyncpg.

    Two transformations are applied:

    1. Scheme normalisation — any ``postgres://`` / ``postgresql://`` /
       ``postgresql+psycopg2://`` variant is rewritten to
       ``postgresql+asyncpg://``.

    2. Full query-string removal — asyncpg does NOT accept ANY URL query
       parameters (``sslmode``, ``channel_binding``, ``connect_timeout``…).
       Passing them causes errors like:
           connect() got an unexpected keyword argument 'sslmode'
           database "neondb&channel_binding=require" does not exist
       The entire query string is stripped here; SSL is re-applied via
       ``connect_args`` inside ``_create_engine``.

    Args:
        url: Raw DATABASE_URL string from settings / environment.

    Returns:
        A URL string starting with ``postgresql+asyncpg://`` and with
        the query string removed entirely.
    """
    url = url.strip()

    # Step 1 — fix the scheme prefix
    url = re.sub(
        r"^postgres(?:ql)?(?:\+\w+)?://",
        "postgresql+asyncpg://",
        url,
        flags=re.IGNORECASE,
    )

    # Step 2 — strip the ENTIRE query string.
    # asyncpg accepts zero URL query parameters. Stripping only ?sslmode
    # leaves other params (e.g. &channel_binding=require) as orphaned '&'
    # fragments that corrupt the database name in the parsed URL.
    url = re.sub(r"\?.*$", "", url)

    return url


def _extract_sslmode(raw_url: str) -> str | None:
    """
    Extract the ``sslmode`` value from a raw DATABASE_URL before it is
    normalised, so we can translate it into asyncpg's ``connect_args``.

    Returns:
        The sslmode value string (e.g. ``'require'``, ``'disable'``),
        or ``None`` if not present.
    """
    match = re.search(r"[?&]sslmode=([^&]+)", raw_url, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _create_engine():
    """
    Build the async SQLAlchemy engine from application settings.

    Called once at module import time — the resulting engine object is a
    connection-pool manager and is safe to share across all coroutines.

    Pool settings
    -------------
    ``pool_pre_ping=True``
        Emits a lightweight ``SELECT 1`` before handing a connection from the
        pool to a session.  Prevents "connection was closed by the server"
        errors after idle periods (common with Neon's serverless PostgreSQL
        which closes idle connections aggressively).

    ``pool_size / max_overflow``
        Defaults (5 + 10) are appropriate for most deployments.  Tune via
        environment variables if needed.
    """
    settings = get_settings()
    raw_url: str = settings.postgres.database_url

    if not raw_url:
        raise RuntimeError(
            "DATABASE_URL is not set.  Add it to your .env file.\n"
            "Example: DATABASE_URL=postgresql+asyncpg://user:pass@host/dbname"
        )

    # Extract sslmode BEFORE stripping it from the URL.
    sslmode = _extract_sslmode(raw_url)
    async_url = _make_async_url(raw_url)

    # Translate sslmode → asyncpg connect_args.
    # asyncpg accepts ssl=True (require) or ssl=False (disable).
    # Any sslmode other than 'disable' is treated as requiring SSL.
    connect_args: dict = {}
    if sslmode and sslmode != "disable":
        connect_args["ssl"] = True
        logger.info("asyncpg SSL enabled (sslmode=%r → ssl=True).", sslmode)
    elif sslmode == "disable":
        connect_args["ssl"] = False
        logger.info("asyncpg SSL explicitly disabled (sslmode=disable).")

    logger.info("Creating async database engine (driver: asyncpg).")

    return create_async_engine(
        async_url,
        echo=settings.app.debug,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


# Module-level singletons — created once, shared for the lifetime of the process.
engine = _create_engine()

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

# ``async_sessionmaker`` is the async equivalent of ``sessionmaker``.
# ``expire_on_commit=False`` prevents SQLAlchemy from expiring ORM objects
# after a commit, which would trigger lazy-load I/O (not allowed in async
# mode without explicit ``await``).
async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,   # Explicit flush gives cleaner error surfaces
    autocommit=False,  # Always use explicit transactions
)

# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields a database session per request.

    Lifecycle
    ---------
    1. A new ``AsyncSession`` is created at the start of the request.
    2. The session is yielded to the endpoint handler.
    3. On success (no exception): the session is committed and closed.
    4. On exception: the session is rolled back and closed, then the
       exception is re-raised so FastAPI returns the appropriate error.

    Usage
    -----
    ::

        from app.db.session import get_db
        from sqlalchemy.ext.asyncio import AsyncSession

        @router.post("/turns")
        async def create_turn(
            payload: TurnCreate,
            db: AsyncSession = Depends(get_db),
        ) -> TurnResponse:
            turn = ConversationTurn(**payload.model_dump())
            db.add(turn)
            # commit happens automatically when the dependency exits cleanly
            return TurnResponse.model_validate(turn)

    Why commit inside the dependency rather than in the handler?
    ------------------------------------------------------------
    Committing in the dependency keeps each handler free of explicit
    ``await db.commit()`` calls and ensures consistency: if the handler
    raises after writing data, the rollback fires automatically.  The
    trade-off is that late errors (e.g. response serialisation) can cause
    unexpected rollbacks — for those edge cases, call ``await db.commit()``
    explicitly inside the handler before returning.
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# Type alias for cleaner handler signatures
# ---------------------------------------------------------------------------

# Use this Annotated alias as the type annotation in endpoint handlers:
#
#   async def handler(db: DbSession) -> ...:
#
# instead of the more verbose:
#
#   async def handler(db: AsyncSession = Depends(get_db)) -> ...:
#
DbSession = Annotated[AsyncSession, Depends(get_db)]
