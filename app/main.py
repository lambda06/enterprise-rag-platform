"""
FastAPI application entry point for the Enterprise RAG Platform.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router

app = FastAPI(
    title="Enterprise RAG Platform",
    version="0.1.0",
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


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "ok"}
