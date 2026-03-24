"""
FastAPI REST API for backtest-audit.

Endpoints
---------
POST /audit                    – full audit (DSR + MC + economic + walk-forward + regime + robustness)
POST /audit/dsr                – DSR test only
POST /audit/mc                 – Monte Carlo test only
POST /audit/pbo                – PBO test (requires returns matrix)
POST /audit/sensitivity        – parameter sensitivity test
POST /audit/economic           – economic significance (effect size, MDE, R²)
POST /audit/walk-forward       – walk-forward OOS validation
POST /audit/regime             – regime-conditional audit
POST /audit/robustness         – robustness stress test
GET  /health                   – liveness probe
GET  /metrics                  – request counters
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
from .economic_significance import economic_significance
from .logging_config import configure_logging, get_logger
from .monte_carlo import monte_carlo_permutation_test
from .pbo import probability_of_backtest_overfitting
from .regime import regime_audit
from .robustness import RobustnessTester
from .sensitivity import parameter_sensitivity
from .walk_forward import walk_forward_validation

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
    def check_matrix(self) -> PBOPayload:
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
    version="0.2.0",
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
    return {"status": "ok", "version": "0.2.0"}


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


@app.post("/audit/economic", tags=["audit"])
def run_economic(payload: ReturnsPayload) -> dict:
    """Run economic significance analysis (effect size, MDE, R²)."""
    try:
        result = economic_significance(pd.Series(payload.returns))
        return {
            "cohens_d": result.cohens_d,
            "effect_size_label": result.effect_size_label,
            "mde_sharpe": result.mde_sharpe,
            "n_obs": result.n_obs,
            "annualised_return": result.annualised_return,
            "annualised_vol": result.annualised_vol,
            "sharpe_ratio": result.sharpe_ratio,
            "break_even_cost_bps": result.break_even_cost_bps,
            "r_squared": result.r_squared,
            "verdict": result.verdict,
            "notes": result.notes,
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class WalkForwardPayload(BaseModel):
    returns: list[float] = Field(..., min_length=20, max_length=50_000)
    n_splits: int = Field(default=5, ge=2, le=20)
    periods_per_year: int = Field(default=252, ge=1)

    @field_validator("returns")
    @classmethod
    def at_least_two_finite(cls, v: list[float]) -> list[float]:
        import math
        finite = [x for x in v if not math.isnan(x) and not math.isinf(x)]
        if len(finite) < 20:
            raise ValueError("Need at least 20 finite return values for walk-forward.")
        return v


@app.post("/audit/walk-forward", tags=["audit"])
def run_walk_forward(payload: WalkForwardPayload) -> dict:
    """Run walk-forward out-of-sample validation."""
    try:
        result = walk_forward_validation(
            pd.Series(payload.returns),
            n_splits=payload.n_splits,
            periods_per_year=payload.periods_per_year,
        )
        return result.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/audit/regime", tags=["audit"])
def run_regime(payload: ReturnsPayload) -> dict:
    """Run regime-conditional audit (per vol/trend regime)."""
    try:
        result = regime_audit(
            pd.Series(payload.returns),
            n_permutations=min(payload.n_permutations, 200),
        )
        return result.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/audit/robustness", tags=["audit"])
def run_robustness(payload: ReturnsPayload) -> dict:
    """Run robustness stress test — 7 failure scenarios."""
    try:
        tester = RobustnessTester(pd.Series(payload.returns))
        report = tester.run_all()
        return report.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
