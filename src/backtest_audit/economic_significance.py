"""
Economic Significance Analysis — goes beyond p-values.

Even a statistically significant backtest edge (low p-value) can be economically
useless if:
  - The effect size is tiny (R² = 0.01 explains almost nothing)
  - The minimum detectable effect is larger than realistic transaction costs
  - Annualised return does not justify the operational risk

This module adds:
  1. Cohen's d — standardised effect size (small / medium / large)
  2. Minimum Detectable Effect (MDE) at a given power
  3. Annualised return estimate and break-even cost
  4. Economic verdict: STRONG / MARGINAL / WEAK

Reference: Cohen (1988), "Statistical Power Analysis for the Behavioral Sciences"
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EconomicSignificanceResult:
    # Effect size
    cohens_d: float            # standardised effect size
    effect_size_label: str     # "negligible" | "small" | "medium" | "large"

    # MDE — smallest true Sharpe detectable at given power
    mde_sharpe: float          # minimum detectable Sharpe ratio
    n_obs: int                 # number of observations used

    # Returns / costs
    annualised_return: float   # mean_daily * 252
    annualised_vol: float      # std_daily * sqrt(252)
    sharpe_ratio: float        # annualised SR (no risk-free adjustment)
    break_even_cost_bps: float # max round-trip cost at which edge disappears

    # R-squared (variance explained by positive-return "signal")
    r_squared: float

    # Verdict
    verdict: str               # "STRONG" | "MARGINAL" | "WEAK"
    notes: str


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def economic_significance(
    returns: pd.Series,
    periods_per_year: int = 252,
    mde_alpha: float = 0.05,
    mde_power: float = 0.80,
    min_sharpe_threshold: float = 0.5,
    min_annual_return_threshold: float = 0.02,  # 2% minimum annual return
) -> EconomicSignificanceResult:
    """
    Compute economic significance of a return stream.

    Parameters
    ----------
    returns : pd.Series
        Daily (or per-period) P&L returns — not cumulative.
    periods_per_year : int
        Trading periods per year; 252 for daily, 52 for weekly.
    mde_alpha : float
        Type-I error rate for MDE calculation.
    mde_power : float
        Statistical power for MDE calculation (1 - Type-II error).
    min_sharpe_threshold : float
        Minimum Sharpe ratio considered "economically meaningful".
    min_annual_return_threshold : float
        Minimum annualised return (fraction) to call edge strong.

    Returns
    -------
    EconomicSignificanceResult
    """
    ret = np.asarray(returns, dtype=float)
    ret = ret[np.isfinite(ret)]
    n = len(ret)
    if n < 2:
        raise ValueError("Need at least 2 finite return observations.")

    mu = float(np.mean(ret))
    sigma = float(np.std(ret, ddof=1))

    # ── Cohen's d ────────────────────────────────────────────────────────────
    # Compare return distribution against the zero-return null
    cohens_d = mu / sigma if sigma > 1e-12 else 0.0
    effect_size_label = _cohens_d_label(abs(cohens_d))

    # ── Annualised metrics ────────────────────────────────────────────────────
    ann_return = mu * periods_per_year
    ann_vol = sigma * math.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 1e-12 else 0.0

    # ── Break-even cost ───────────────────────────────────────────────────────
    # How many bps of round-trip cost wipes out the edge?
    # Assumes every period has one round-trip (conservative upper bound).
    break_even_bps = ann_return * 1e4 / periods_per_year if ann_return > 0 else 0.0

    # ── Minimum Detectable Effect (Sharpe) ────────────────────────────────────
    # Based on a one-sided t-test for mu > 0.
    # n * mu² / sigma² ~ chi²(1) under H1.  Solving for smallest detectable mu:
    z_alpha = stats.norm.ppf(1 - mde_alpha)
    z_beta = stats.norm.ppf(mde_power)
    # MDE for daily mean; then convert to Sharpe
    mde_daily_mean = (z_alpha + z_beta) * sigma / math.sqrt(n)
    mde_sharpe = mde_daily_mean * math.sqrt(periods_per_year) / (ann_vol / math.sqrt(periods_per_year) if ann_vol > 1e-12 else 1.0)

    # ── R² analog — fraction of variance "explained" by positive signal ───────
    # Regress returns on their own sign (1/0): how much variance is systematic?
    signal = (ret > 0).astype(float)
    if np.std(signal) > 1e-12:
        corr = float(np.corrcoef(signal, ret)[0, 1])
        r_squared = corr ** 2
    else:
        r_squared = 0.0

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict, notes = _verdict(
        sharpe, ann_return, cohens_d, r_squared,
        min_sharpe_threshold, min_annual_return_threshold,
    )

    return EconomicSignificanceResult(
        cohens_d=round(cohens_d, 6),
        effect_size_label=effect_size_label,
        mde_sharpe=round(mde_sharpe, 6),
        n_obs=n,
        annualised_return=round(ann_return, 6),
        annualised_vol=round(ann_vol, 6),
        sharpe_ratio=round(sharpe, 6),
        break_even_cost_bps=round(break_even_bps, 4),
        r_squared=round(r_squared, 6),
        verdict=verdict,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cohens_d_label(d: float) -> str:
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def _verdict(
    sharpe: float,
    ann_return: float,
    cohens_d: float,
    r_squared: float,
    min_sharpe: float,
    min_return: float,
) -> tuple[str, str]:
    reasons: list[str] = []

    if sharpe >= min_sharpe and ann_return >= min_return and abs(cohens_d) >= 0.2:
        if r_squared >= 0.05:
            return "STRONG", f"Sharpe={sharpe:.2f}, R²={r_squared:.3f}, d={cohens_d:.3f}"
        return "STRONG", f"Sharpe={sharpe:.2f}, d={cohens_d:.3f} (low R²={r_squared:.3f} but acceptable)"

    if sharpe < 0 or ann_return < 0:
        return "WEAK", f"Negative edge: ann_return={ann_return:.2%}, Sharpe={sharpe:.2f}"

    if sharpe < min_sharpe:
        reasons.append(f"Sharpe {sharpe:.2f} < threshold {min_sharpe:.2f}")
    if ann_return < min_return:
        reasons.append(f"ann_return {ann_return:.2%} < threshold {min_return:.2%}")
    if abs(cohens_d) < 0.2:
        reasons.append(f"negligible effect size d={cohens_d:.3f}")
    if r_squared < 0.01:
        reasons.append(f"R²={r_squared:.4f} (< 1% variance explained)")

    if len(reasons) >= 2:
        return "WEAK", "; ".join(reasons)
    return "MARGINAL", "; ".join(reasons) if reasons else f"Sharpe={sharpe:.2f}"
