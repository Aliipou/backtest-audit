# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── install dependencies ──────────────────────────────────────────────────────
FROM base AS deps
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --upgrade pip && \
    pip install ".[api]"

# ── production image ──────────────────────────────────────────────────────────
FROM deps AS production

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "backtest_audit.api:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--access-log", "--log-level", "info"]
