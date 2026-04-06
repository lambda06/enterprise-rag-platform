#!/bin/bash
# =============================================================================
# entrypoint.sh — Container startup script
#
# Steps:
#   1. Run Alembic database migrations (idempotent — safe to run every start)
#   2. Start Uvicorn serving the FastAPI application
#
# Environment variables (from .env / Cloud Run secrets / docker-compose):
#   DATABASE_URL   — PostgreSQL connection string (required for migrations)
#   PORT           — Override listen port (default: 8000, used by Cloud Run)
# =============================================================================

set -e  # Exit immediately on any error

echo "==> Running Alembic migrations..."
alembic upgrade head
echo "==> Migrations complete."

# Cloud Run injects a PORT env var; default to 8000 for local docker-compose
PORT="${PORT:-8000}"

echo "==> Starting Uvicorn on port ${PORT}..."
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers 1 \
    --log-level info
