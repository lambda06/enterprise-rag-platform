"""
FastAPI application entry point for the Enterprise RAG Platform.
"""

from contextlib import asynccontextmanager

import logging
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router

# Explicitly configure the 'app' logger so it is visible in the terminal even
# after uvicorn has already claimed the root logger (which makes basicConfig a
# no-op and silently drops all app.* log records).
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(levelname)-9s %(name)s - %(message)s"))

_app_logger = logging.getLogger("app")
_app_logger.setLevel(logging.INFO)
_app_logger.addHandler(_handler)
_app_logger.propagate = False  # Don't double-log through uvicorn's root handler

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown.

    On shutdown, flushes any buffered Langfuse trace events so no telemetry
    is lost when uvicorn stops (e.g. on --reload or CTRL-C).
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    try:
        from app.observability.langfuse_tracer import tracer
        tracer.flush()
    except Exception:
        pass


app = FastAPI(
    title="Enterprise RAG Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: explicit allowed origins.
# Wildcard + allow_credentials=True is rejected by all browsers for credentialed
# requests (CORS spec). We read from CORS_ORIGINS env var (comma-separated) so
# deployments can configure without a code change.
_cors_origins_env = os.getenv("CORS_ORIGINS", "")
if _cors_origins_env:
    _allowed_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
else:
    # Sensible defaults for local development and Streamlit frontend.
    _allowed_origins = [
        "http://localhost:8501",   # Streamlit default port
        "http://localhost:3000",   # Next.js dev
        "http://localhost:8000",   # FastAPI self (for Swagger UI)
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API v1 routes
app.include_router(documents_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "ok"}
