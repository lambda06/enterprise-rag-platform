#!/usr/bin/env python3
"""
Verify connectivity to external services used by the Enterprise RAG Platform.

Requires: pip install python-dotenv httpx qdrant-client asyncpg redis

Usage: python scripts/verify_connections.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `app` package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings

# Load .env from project root
env_path = PROJECT_ROOT / ".env"
if not env_path.exists():
    print(f"Error: .env file not found at {env_path}")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(env_path)
except ImportError:
    print("Error: python-dotenv required. Run: pip install python-dotenv")
    sys.exit(1)


# Load typed settings (also exercises app/core/config.py)
SETTINGS = get_settings()


async def check_groq() -> tuple[bool, str]:
    """Test Groq API connectivity."""
    try:
        import httpx
    except ImportError:
        return False, "httpx not installed (pip install httpx)"

    api_key = SETTINGS.groq.api_key
    if not api_key or api_key.startswith("gsk_xxx"):
        return False, "GROQ_API_KEY not set or still placeholder"

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if r.status_code == 200:
                return True, "Connected"
            return False, f"HTTP {r.status_code}: {r.text[:100]}"
        except Exception as e:
            return False, str(e)


def _check_qdrant_sync() -> tuple[bool, str]:
    """Synchronous Qdrant check (client is sync-only)."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return False, "qdrant-client not installed (pip install qdrant-client)"

    url = SETTINGS.qdrant.url
    api_key = SETTINGS.qdrant.api_key or None

    if not url:
        return False, "QDRANT_URL not set"

    try:
        client = QdrantClient(url=url, api_key=api_key or None)
        client.get_collections()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


async def check_qdrant() -> tuple[bool, str]:
    """Test Qdrant Cloud connectivity."""
    return await asyncio.to_thread(_check_qdrant_sync)


async def check_postgres() -> tuple[bool, str]:
    """Test Neon PostgreSQL connectivity."""
    try:
        import asyncpg
    except ImportError:
        return False, "asyncpg not installed (pip install asyncpg)"

    url = SETTINGS.postgres.database_url
    if not url:
        return False, "DATABASE_URL not set"

        from redis.asyncio.connection import Connection
        from redis.asyncio.connection import Connection
    # asyncpg expects postgresql://, not postgresql+asyncpg:// (SQLAlchemy format)
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)

    try:
        conn = await asyncpg.connect(url, timeout=5.0)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


async def check_redis() -> tuple[bool, str]:
    """Test Upstash Redis connectivity (REST API or standard Redis)."""
    # Prefer Upstash REST API (UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN)
    rest_url = SETTINGS.redis.rest_url
    rest_token = SETTINGS.redis.rest_token
    if rest_url and rest_token:
        try:
            import httpx
        except ImportError:
            return False, "httpx not installed (pip install httpx)"
        try:
            ping_url = rest_url.rstrip("/") + "/ping"
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    ping_url,
                    headers={"Authorization": f"Bearer {rest_token}"},
                )
                if r.status_code == 200:
                    return True, "Connected (REST API)"
                return False, f"HTTP {r.status_code}: {r.text[:100]}"
        except Exception as e:
            return False, str(e)

    # Fallback to standard Redis (REDIS_URL)
    try:
        from redis.asyncio import Redis
    except ImportError:
        return False, "redis not installed (pip install redis)"

    url = os.getenv("REDIS_URL")
    if not url or "xxxxx" in url:
        return False, "REDIS_URL or UPSTASH_REDIS_REST_URL not set"

    try:
        client = Redis.from_url(url, decode_responses=True, socket_timeout=5.0)
        await client.ping()
        await client.aclose()
        return True, "Connected"
    except Exception as e:
        return False, str(e)


async def main() -> None:
    print("=" * 60)
    print("Enterprise RAG Platform — Connection Verification")
    print("=" * 60)

    checks = [
        ("Groq API", check_groq),
        ("Qdrant Cloud", check_qdrant),
        ("Neon PostgreSQL", check_postgres),
        ("Upstash Redis", check_redis),
    ]

    results = await asyncio.gather(
        *[fn() for _, fn in checks],
        return_exceptions=False,
    )

    for (name, _), (ok, msg) in zip(checks, results):
        status = "SUCCESS" if ok else "FAILED"
        print(f"\n{name}: {status}")
        print(f"  {msg}")

    print("\n" + "=" * 60)
    failed = sum(1 for ok, _ in results if not ok)
    if failed == 0:
        print("All services connected successfully.")
    else:
        print(f"{failed} service(s) failed. Check your .env configuration.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
