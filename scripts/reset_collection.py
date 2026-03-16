"""
reset_collection.py — Development utility: drop and recreate the Qdrant collection.

WARNING: THIS IS A DEVELOPMENT UTILITY ONLY.
It must NEVER be imported or called from application code (API routes, services,
startup hooks, background tasks, etc.).  It is intentionally kept outside the
app/ package for that reason.  Run it manually from the command line when you
need to migrate the vector store schema (e.g., after changing the embedding
model and therefore the vector dimension).

Background
──────────
The Qdrant collection schema is tied to the dense-vector dimension.  When the
embedding model changed from BAAI/bge-small-en-v1.5 (384-dim, text-only) to
gemini-embedding-2-preview (768-dim, multimodal) the existing collection became
dimensionally incompatible.  This script performs the one-time (or per-env)
migration:

  1. Prompts for explicit confirmation so the operator cannot run it by accident.
  2. Logs a prominent warning describing exactly what will be deleted and why.
  3. Drops the existing collection (ALL vectors and payloads are lost).
  4. Recreates it with 768-dim dense (cosine) + sparse BM25 indexes.

After running this script, re-ingest all documents via the normal upload API.

Usage
─────
  python scripts/reset_collection.py

The script reads QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME, and
QDRANT_VECTOR_SIZE from the project .env file (via app.core.config).
"""

# ── DEVELOPMENT UTILITY — DO NOT IMPORT FROM APPLICATION CODE ──────────────
# This script is intentionally a top-level __main__ module, not part of the
# app package.  Importing it from application code is an error.
# ───────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import sys
import os

# Ensure the application package is importable when this script is run directly.
# The script lives outside the `app/` package, so Python doesn't automatically
# include the project root on sys.path.  Add it explicitly before importing
# anything from `app`.
#
# This mirrors what you'd get if you ran the project via `python -m`, and it
# keeps the utility self‑contained without requiring callers to set
# PYTHONPATH manually.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ---------------------------------------------------------------------------
# Logging setup
# Must be configured before any app imports so that the app's own log calls
# are captured from the very first line they emit.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _confirm(collection_name: str) -> bool:
    """Ask the operator to type 'yes' before any destructive action.

    Returns True only when the operator types the exact string 'yes'.
    Any other input (including 'y', 'YES', blank) is treated as a cancellation.
    """
    print()
    print("━" * 62)
    print("  QDRANT COLLECTION RESET — PERMANENT DATA DELETION")
    print("━" * 62)
    print(f"  Collection  : {collection_name}")
    print("  This will   : DROP the collection and ALL its vectors/payloads.")
    print("  Why         : Embedding model changed from BAAI/bge-small-en-v1.5")
    print("                (384-dim, text-only) to gemini-embedding-2-preview")
    print("                (768-dim, multimodal).  Old vectors are dimensionally")
    print("                incompatible with the new schema.")
    print("  After this  : Re-ingest ALL documents via the upload API.")
    print("━" * 62)
    print()
    answer = input("  Type 'yes' to proceed, anything else to cancel: ").strip()
    print()
    return answer == "yes"


def reset_collection() -> None:
    """Drop and recreate the configured Qdrant collection.

    Reads connection parameters and target vector_size from app settings
    (QDRANT_* env vars / .env file).  Logs extensively so the operation is
    auditable in any log aggregator.
    """
    # Import here (not at module top) so that misconfigured environments fail
    # loudly at runtime rather than at import time when other scripts import
    # helpers from this file (though they shouldn't — see module docstring).
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        SparseIndexParams,
        SparseVectorParams,
        VectorParams,
    )

    from app.core.config import get_settings
    from app.vectorstore.qdrant_client import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

    settings = get_settings()
    qdrant_cfg = settings.qdrant
    vector_size: int = qdrant_cfg.vector_size
    collection_name: str = qdrant_cfg.collection_name

    client = QdrantClient(
        url=qdrant_cfg.url,
        api_key=qdrant_cfg.api_key,
        timeout=qdrant_cfg.timeout,
    )

    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        logger.warning(
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  QDRANT COLLECTION RESET — PERMANENT DATA DELETION IN PROGRESS  \n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Collection : %s\n"
            "  Reason     : Embedding model migrated from BAAI/bge-small-en-v1.5\n"
            "               (384-dim, text-only) to gemini-embedding-2-preview\n"
            "               (768-dim, multimodal).  Old vectors are dimensionally\n"
            "               incompatible with the new schema — the collection\n"
            "               must be dropped and recreated before re-ingestion.\n"
            "  Action     : Dropping collection now.  ALL STORED VECTORS AND\n"
            "               PAYLOADS WILL BE LOST.  Re-ingest all documents\n"
            "               after this operation completes.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            collection_name,
        )
        client.delete_collection(collection_name)
        logger.info("Collection '%s' deleted successfully.", collection_name)
    else:
        logger.info(
            "Collection '%s' does not exist — nothing to drop.", collection_name
        )

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    logger.info(
        "Collection '%s' recreated with %d-dim dense vectors (cosine) "
        "and sparse BM25 index.  Re-ingest all documents now.",
        collection_name,
        vector_size,
    )


if __name__ == "__main__":
    # ── Safety gate ─────────────────────────────────────────────────────────
    # Import settings early so we can show the collection name in the prompt
    # before establishing a live Qdrant connection.
    from app.core.config import get_settings

    _collection = get_settings().qdrant.collection_name

    if not _confirm(_collection):
        print("  Cancelled — no changes were made.")
        sys.exit(0)

    reset_collection()
    print("  Done.  Collection has been reset.  Re-ingest your documents now.")
