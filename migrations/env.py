"""
Alembic migration environment — configured for async PostgreSQL (asyncpg).

Key customisations vs the default async template
-------------------------------------------------
1. DATABASE_URL is read from app.core.config (honours the .env file) rather
   than from alembic.ini, so there is a single source of truth for the URL.
2. The URL is normalised through _make_async_url / _extract_sslmode so that
   ?sslmode=require and ?channel_binding=require are stripped and SSL is
   re-applied via connect_args — exactly the same logic used at runtime.
3. target_metadata points to Base.metadata from app.models.conversation so
   that `alembic revision --autogenerate` discovers the ConversationTurn table.
"""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so `app.*` imports resolve when
# alembic is invoked from the repo root via `alembic upgrade head`.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Alembic Config object
# ---------------------------------------------------------------------------
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Target metadata — must point to our declarative Base so autogenerate works.
# ---------------------------------------------------------------------------
# Import Base and all models so SQLAlchemy knows the full table schema.
from app.models.conversation import Base  # noqa: E402  (after sys.path fix)
import app.models.conversation  # noqa: E402, F401 — ensures model is registered

target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Database URL — read from app settings, not alembic.ini.
# We reuse the same URL normalisation helpers from app.db.session so that
# the URL that Alembic connects to is identical to what SQLAlchemy uses at
# runtime.
# ---------------------------------------------------------------------------
from app.db.session import _extract_sslmode, _make_async_url  # noqa: E402
from app.core.config import get_settings  # noqa: E402

def _get_url_and_connect_args() -> tuple[str, dict]:
    """Return (async_url, connect_args) honoring the .env DATABASE_URL."""
    settings = get_settings()
    raw = settings.postgres.database_url
    sslmode = _extract_sslmode(raw)
    url = _make_async_url(raw)
    connect_args: dict = {}
    if sslmode and sslmode != "disable":
        connect_args["ssl"] = True
    elif sslmode == "disable":
        connect_args["ssl"] = False
    return url, connect_args


# ---------------------------------------------------------------------------
# Offline mode (generates SQL without a live DB connection)
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Emit raw SQL to stdout — useful for DBA review without a live DB."""
    url, _ = _get_url_and_connect_args()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode (connects to the live DB and applies migrations)
# ---------------------------------------------------------------------------

def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create a fresh asyncpg engine and run migrations against the live DB."""
    url, connect_args = _get_url_and_connect_args()

    # Build the engine config dict that async_engine_from_config expects.
    engine_config = {
        "sqlalchemy.url": url,
    }

    connectable = async_engine_from_config(
        engine_config,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migrations — called by `alembic upgrade head`."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
