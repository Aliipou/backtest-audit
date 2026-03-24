"""
FastAPI REST API for backtest-audit.

Endpoints
---------
POST /audit             – full audit (DSR + Monte Carlo)
POST /audit/dsr         – DSR test only
POST /audit/mc          – Monte Carlo test only
POST /audit/pbo         – PBO test (requires returns matrix)
POST /audit/sensitivity – parameter sensitivity test
GET  /health            – liveness probe
GET  /metrics           – request counters
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .auditor import BacktestAuditor
from .deflated_sharpe import deflated_sharpe_ratio
from .logging_config import configure_logging, get_logger
from .monte_carlo import monte_carlo_permutation_test
from .pbo import probability_of_backtest_overfitting
from .sensitivity import parameter_sensitivity

configure_logging()
logger = get_logger("api")

# ---------------------------------------------------------------------------
# Rate limiter: 60 req / 60 s per IP
# ---------------------------------------------------------------------------
_RATE_LIMIT = 60
_RATE_WINDOW = 60.0
_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(ip: str) -> None:
    now = time.monotonic()
    _rate_buckets[ip] = [t for t in _rate_buckets[ip] if now - t < _RATE_WINDOW]
    if len(_rate_buckets[ip]) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Max 60 requests/minute.",
        )
    _rate_buckets[ip].append(now)


_metrics: dict[str, int] = defaultdict(int)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ReturnsPayload(BaseModel):
    returns: list[float] = Field(..., min_length=2, max_length=50_000)
    n_trials: int = Field(default=1, ge=1, le=100_000)
    n_permutations: int = Field(default=1000, ge=10, le=100_000)
    skewness: float = Field(default=0.0, ge=-10.0, le=10.0)
    kurtosis: float = Field(default=3.0, ge=1.0, le=50.0)
    random_state: int | None = None

    @field_validator("returns")
    @classmethod
    def at_least_two_finite(cls, v: list[float]) -> list[float]:
        import math
        finite = [x for x in v if not math.isnan(x) and not math.isinf(x)]
        if len(finite) < 2:
            raise ValueError("Need at least 2 finite return values.")
        return v


class PBOPayload(BaseModel):
    returns_matrix: dict[str, list[float]] = Field(
        ..., description="Column name → returns list"
    )
    n_splits: int = Field(default=16, ge=2, le=64)

    @model_validator(mode="after")
    def check_matrix(self) -> "PBOPayload":
        if len(self.returns_matrix) < 2:
            raise ValueError("returns_matrix must have at least 2 strategy columns.")
        lengths = {len(v) for v in self.returns_matrix.values()}
        if len(lengths) != 1:
            raise ValueError("All strategy return series must have equal length.")
        if lengths.pop() < 4:
            raise ValueError("Each return series needs at least 4 observations.")
        return self


class SensitivityPayload(BaseModel):
    results: dict[str, float] = Field(..., description="param_combo_str → Sharpe")
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)

    @field_validator("results")
    @classmethod
    def non_empty(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("results must not be empty.")
        return v


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    logger.info("backtest-audit API starting")
    yield
    logger.info("backtest-audit API stopping")


app = FastAPI(
    title="backtest-audit",
    version="0.1.0",
    description="Statistical overfitting audit tools for algorithmic trading backtests.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-Id"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next: Any) -> Response:
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "unknown"

    try:
        _check_rate_limit(client_ip)
    except HTTPException as exc:
        _metrics["rate_limited"] += 1
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers={"X-Request-Id": request_id},
        )

    t0 = time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

    _metrics["requests_total"] += 1
    _metrics[f"status_{response.status_code}"] += 1

    logger.info(
        "request",
        extra={
            "extra": {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_ms": elapsed_ms,
                "ip": client_ip,
            }
        },
    )
    response.headers["X-Request-Id"] = request_id
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/metrics", tags=["ops"])
def metrics() -> dict[str, int]:
    return dict(_metrics)


@app.post("/audit", tags=["audit"])
def run_full_audit(payload: ReturnsPayload) -> dict:
    """Run DSR + Monte Carlo and return AuditReport as JSON."""
    try:
        auditor = BacktestAuditor(pd.Series(payload.returns), n_trials=payload.n_trials)
        report = auditor.run_all(
            n_permutations=payload.n_permutations,
            skewness=payload.skewness,
            kurtosis=payload.kurtosis,
        )
        return report.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/audit/dsr", tags=["audit"])
def run_dsr(payload: ReturnsPayload) -> dict:
    """Run Deflated Sharpe Ratio test."""
    try:
        return deflated_sharpe_ratio(
            pd.Series(payload.returns),
            n_trials=payload.n_trials,
            skewness=payload.skewness,
            kurtosis=payload.kurtosis,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/audit/mc", tags=["audit"])
def run_mc(payload: ReturnsPayload) -> dict:
    """Run Monte Carlo permutation test."""
    try:
        return monte_carlo_permutation_test(
            pd.Series(payload.returns),
            n_permutations=payload.n_permutations,
            random_state=payload.random_state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/audit/pbo", tags=["audit"])
def run_pbo(payload: PBOPayload) -> dict:
    """Run Probability of Backtest Overfitting test."""
    try:
        df = pd.DataFrame(payload.returns_matrix)
        return probability_of_backtest_overfitting(df, n_splits=payload.n_splits)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/audit/sensitivity", tags=["audit"])
def run_sensitivity(payload: SensitivityPayload) -> dict:
    """Run parameter sensitivity analysis."""
    try:
        return parameter_sensitivity(payload.results, payload.param_grid)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
