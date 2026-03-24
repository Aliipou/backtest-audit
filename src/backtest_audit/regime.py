"""
Regime-Conditional Audit — does the edge survive across market regimes?

A backtest that looks good in aggregate can hide regime dependency: strong
in low-vol trending markets, terrible in high-vol or mean-reverting ones.

This module:
  1. Classifies each return observation into a regime using volatility
     (low / normal / high) and optionally trend (up / down / flat).
  2. Runs DSR + Monte Carlo on each regime slice.
  3. Reports per-regime verdicts and an overall regime consistency score.

Regime classification
---------------------
  - Compute a rolling 20-period volatility (EWMA-based).
  - Normalise by the full-sample std.
  - LOW_VOL  : rolling_vol < 0.75 × global_std
  - HIGH_VOL : rolling_vol > 1.5  × global_std
  - NORMAL   : in between
  - Trend determined by sign of 10-period trailing return.

Verdict
-------
  ROBUST   : all regimes individually PASS or WARN
  FRAGILE  : at least one regime FAILs but not the majority
  BROKEN   : majority of regimes FAIL
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .deflated_sharpe import deflated_sharpe_ratio
from .monte_carlo import monte_carlo_permutation_test

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOW_VOL_FACTOR = 0.75
HIGH_VOL_FACTOR = 1.50
TREND_WINDOW = 10
VOL_WINDOW = 20
MIN_REGIME_OBS = 10      # skip regime slice if fewer than this


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RegimeSliceResult:
    regime: str
    n_obs: int
    sharpe: float
    dsr: float
    mc_pvalue: float
    verdict: str          # "PASS" | "WARN" | "FAIL" | "SKIP"
    notes: str


@dataclass
class RegimeAuditResult:
    slices: list[RegimeSliceResult] = field(default_factory=list)
    n_pass: int = 0
    n_warn: int = 0
    n_fail: int = 0
    n_skip: int = 0
    overall_verdict: str = "FAIL"
    consistency_score: float = 0.0    # fraction of non-skipped slices that pass/warn
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "overall_verdict": self.overall_verdict,
            "consistency_score": round(self.consistency_score, 4),
            "n_pass": self.n_pass,
            "n_warn": self.n_warn,
            "n_fail": self.n_fail,
            "n_skip": self.n_skip,
            "notes": self.notes,
            "slices": [
                {
                    "regime": s.regime,
                    "n_obs": s.n_obs,
                    "sharpe": round(s.sharpe, 4),
                    "dsr": round(s.dsr, 4),
                    "mc_pvalue": round(s.mc_pvalue, 4),
                    "verdict": s.verdict,
                    "notes": s.notes,
                }
                for s in self.slices
            ],
        }


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def regime_audit(
    returns: pd.Series,
    periods_per_year: int = 252,
    n_permutations: int = 500,
    classify_trend: bool = True,
    n_trials: int = 1,
) -> RegimeAuditResult:
    """
    Run DSR + Monte Carlo on each regime slice.

    Parameters
    ----------
    returns : pd.Series
        Return series (daily or per-period).
    periods_per_year : int
        252 daily / 52 weekly.
    n_permutations : int
        Monte Carlo permutations per regime (fewer for speed).
    classify_trend : bool
        Whether to add trend-based regime splits (up/down) in addition to vol.
    n_trials : int
        Passed to DSR; set higher if many parameter combos were tried.

    Returns
    -------
    RegimeAuditResult
    """
    ret = returns.dropna().reset_index(drop=True)
    arr = np.asarray(ret, dtype=float)
    n = len(arr)

    if n < 2 * MIN_REGIME_OBS:
        raise ValueError(
            f"Need at least {2 * MIN_REGIME_OBS} observations for regime analysis; got {n}."
        )

    # ── Compute rolling vol ────────────────────────────────────────────────
    global_std = float(np.std(arr, ddof=1))
    if global_std < 1e-12:
        raise ValueError("Return series has near-zero variance — cannot classify regimes.")

    rolling_vol = _ewma_vol(arr, span=VOL_WINDOW)

    # ── Classify vol regimes ───────────────────────────────────────────────
    vol_regime = np.where(
        rolling_vol < LOW_VOL_FACTOR * global_std, "low_vol",
        np.where(rolling_vol > HIGH_VOL_FACTOR * global_std, "high_vol", "normal_vol"),
    )

    regimes: dict[str, np.ndarray] = {
        "low_vol":    arr[vol_regime == "low_vol"],
        "normal_vol": arr[vol_regime == "normal_vol"],
        "high_vol":   arr[vol_regime == "high_vol"],
    }

    # ── Optionally add trend regimes ───────────────────────────────────────
    if classify_trend and n >= TREND_WINDOW + 1:
        trailing = np.array([
            float(np.sum(arr[max(0, i - TREND_WINDOW):i]))
            for i in range(1, n + 1)
        ])
        regimes["trend_up"]   = arr[trailing > 0]
        regimes["trend_down"] = arr[trailing < 0]

    # ── Run DSR + MC per regime ────────────────────────────────────────────
    slices: list[RegimeSliceResult] = []
    for regime_name, regime_ret in regimes.items():
        slice_result = _audit_slice(
            regime_name, regime_ret, n_trials, n_permutations, periods_per_year
        )
        slices.append(slice_result)

    n_pass = sum(1 for s in slices if s.verdict == "PASS")
    n_warn = sum(1 for s in slices if s.verdict == "WARN")
    n_fail = sum(1 for s in slices if s.verdict == "FAIL")
    n_skip = sum(1 for s in slices if s.verdict == "SKIP")

    n_tested = n_pass + n_warn + n_fail
    consistency = (n_pass + n_warn) / n_tested if n_tested > 0 else 0.0

    if n_tested == 0:
        overall = "FAIL"
        notes = "All regime slices were too small to test."
    elif n_fail == 0:
        overall = "ROBUST"
        notes = f"All {n_tested} tested regimes pass/warn (consistency={consistency:.0%})"
    elif n_fail <= n_tested // 2:
        overall = "FRAGILE"
        notes = f"{n_fail}/{n_tested} regimes fail (consistency={consistency:.0%})"
    else:
        overall = "BROKEN"
        notes = f"Majority of regimes fail: {n_fail}/{n_tested} (consistency={consistency:.0%})"

    return RegimeAuditResult(
        slices=slices,
        n_pass=n_pass,
        n_warn=n_warn,
        n_fail=n_fail,
        n_skip=n_skip,
        overall_verdict=overall,
        consistency_score=round(consistency, 4),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _audit_slice(
    name: str,
    arr: np.ndarray,
    n_trials: int,
    n_permutations: int,
    periods_per_year: int,
) -> RegimeSliceResult:
    n = len(arr)
    if n < MIN_REGIME_OBS:
        return RegimeSliceResult(
            regime=name, n_obs=n, sharpe=0.0, dsr=0.0, mc_pvalue=1.0,
            verdict="SKIP", notes=f"Only {n} obs — need {MIN_REGIME_OBS}",
        )

    ret_series = pd.Series(arr)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    sharpe = mu / sigma * math.sqrt(periods_per_year) if sigma > 1e-12 else 0.0

    try:
        dsr_res = deflated_sharpe_ratio(ret_series, n_trials=n_trials)
        dsr = float(dsr_res.get("dsr", 0.0))
        dsr_verdict = dsr_res.get("verdict", "FAIL")
    except Exception:
        dsr = 0.0
        dsr_verdict = "FAIL"

    try:
        mc_res = monte_carlo_permutation_test(ret_series, n_permutations=n_permutations)
        mc_pvalue = float(mc_res.get("pvalue", 1.0))
        mc_verdict = mc_res.get("verdict", "FAIL")
    except Exception:
        mc_pvalue = 1.0
        mc_verdict = "FAIL"

    # Combined verdict: need both DSR and MC to agree
    if dsr_verdict == "PASS" and mc_verdict == "PASS":
        verdict = "PASS"
    elif dsr_verdict == "FAIL" or mc_verdict == "FAIL":
        verdict = "FAIL" if sharpe < 0 else "WARN"
    else:
        verdict = "WARN"

    return RegimeSliceResult(
        regime=name,
        n_obs=n,
        sharpe=round(sharpe, 6),
        dsr=round(dsr, 6),
        mc_pvalue=round(mc_pvalue, 6),
        verdict=verdict,
        notes=f"DSR={dsr:.3f} [{dsr_verdict}], MC_p={mc_pvalue:.3f} [{mc_verdict}]",
    )


def _ewma_vol(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponentially weighted moving average of abs(return) as vol proxy."""
    alpha = 2.0 / (span + 1)
    vol = np.empty(len(arr))
    vol[0] = abs(arr[0])
    for i in range(1, len(arr)):
        vol[i] = alpha * abs(arr[i]) + (1 - alpha) * vol[i - 1]
    return vol
