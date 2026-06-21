# Dockerfile — Celery Worker for Production RAG System
#
# PREREQUISITES (run on Windows before building):
#   1. Update pyproject.toml with the full dependencies list
#   2. uv lock           ← generates uv.lock
#   3. docker compose up -d --build

FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv --break-system-packages --quiet

# Use the system Python — prevents uv downloading Python 3.14
ENV UV_PYTHON_PREFERENCE=only-system
ENV UV_PYTHON=python3.12

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install ALL dependencies from lockfile
# --no-install-project = install dependencies only, do NOT build/install
#   the project itself as a Python package (avoids hatchling wheel errors)
# --no-dev = skip pytest and other dev tools
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code AFTER installing deps (better layer caching)
COPY src/ ./src/

ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Run as non-root user
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser