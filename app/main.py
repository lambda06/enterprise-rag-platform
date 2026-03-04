"""
FastAPI application entry point for the Enterprise RAG Platform.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router


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

# CORS: allow all origins for development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
