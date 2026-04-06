# =============================================================================
# Enterprise RAG Platform — Dockerfile
# Multi-stage build:
#   Stage 1 (builder) — installs Python dependencies into /opt/venv
#   Stage 2 (runtime) — copies only the venv + source; runs as non-root
#
# Target: python:3.12-slim  (~150 MB base)
# Expected final image size: ~450–550 MB  (after removing torch/transformers)
# =============================================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System packages needed only at build time (gcc for some C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment so Stage 2 can copy it cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first — Docker layer cache means this only re-runs
# when requirements.txt changes, not on every source code edit.
COPY requirements.txt .

# Install all Python dependencies into the venv
# --no-cache-dir keeps the layer small (no pip wheel cache baked in)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Minimal runtime system packages:
#   curl — used by the HEALTHCHECK below
# PyMuPDF, pdfplumber, pdfminer, pypdfium2 all ship their own binaries;
# no system-level poppler / tesseract / libxml2 needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual environment from Stage 1
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Run as a non-root user (security best practice)
RUN useradd --no-create-home --shell /bin/bash appuser

# Set the working directory
WORKDIR /app

# Copy application source
# .dockerignore excludes venv_312/, .env, __pycache__, logs/, data/, etc.
COPY . .

# Copy and set up the entrypoint script
RUN mv /app/docker/entrypoint.sh /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Switch to non-root user
USER appuser

# FastAPI listens on 8000
EXPOSE 8000

# Lightweight health check so Cloud Run / docker compose know when ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command — entrypoint.sh runs alembic migrations then uvicorn
CMD ["/app/entrypoint.sh"]
