"""
Walk-Forward Out-of-Sample Validation.

In-sample statistical significance (DSR, MC) doesn't prove the edge is real —
it might just mean the model was overfit to the backtest window.  Walk-forward
OOS validation splits the return history into non-overlapping in-sample /
out-of-sample windows and verifies that in-sample winners also win OOS.

Method
------
1. Divide total return history into `n_splits` windows.
2. For each window i:  train on window i, validate on window i+1.
3. Compute Sharpe ratio and DSR on each in-sample and OOS slice.
4. Report: fraction of windows where OOS SR > 0, IS→OOS correlation, OOS hit rate.

Verdict thresholds
------------------
  PASS  :  OOS hit rate >= 0.6  and  IS/OOS Sharpe correlation > 0
  WARN  :  OOS hit rate >= 0.4
  FAIL  :  OOS hit rate < 0.4  or  all OOS windows lose money
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .deflated_sharpe import deflated_sharpe_ratio

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    window_idx: int
    is_sharpe: float
    oos_sharpe: float
    is_dsr: float
    oos_positive: bool          # OOS SR > 0


@dataclass
class WalkForwardResult:
    n_windows: int
    windows: list[WindowResult] = field(default_factory=list)
    oos_hit_rate: float = 0.0          # fraction of OOS windows with SR > 0
    is_oos_correlation: float = 0.0    # Pearson r between IS and OOS Sharpes
    mean_oos_sharpe: float = 0.0
    mean_is_sharpe: float = 0.0
    verdict: str = "FAIL"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "n_windows": self.n_windows,
            "oos_hit_rate": round(self.oos_hit_rate, 4),
            "is_oos_correlation": round(self.is_oos_correlation, 4),
            "mean_oos_sharpe": round(self.mean_oos_sharpe, 4),
            "mean_is_sharpe": round(self.mean_is_sharpe, 4),
            "verdict": self.verdict,
            "notes": self.notes,
            "windows": [
                {
                    "idx": w.window_idx,
                    "is_sharpe": round(w.is_sharpe, 4),
                    "oos_sharpe": round(w.oos_sharpe, 4),
                    "is_dsr": round(w.is_dsr, 4),
                    "oos_positive": w.oos_positive,
                }
                for w in self.windows
            ],
        }


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def walk_forward_validation(
    returns: pd.Series,
    n_splits: int = 5,
    periods_per_year: int = 252,
    min_window_size: int = 20,
    oos_hit_rate_pass: float = 0.6,
    oos_hit_rate_warn: float = 0.4,
) -> WalkForwardResult:
    """
    Perform walk-forward out-of-sample validation.

    Parameters
    ----------
    returns : pd.Series
        Complete return series (daily or per-period).
    n_splits : int
        Number of IS/OOS window pairs.  Total splits = n_splits + 1 windows.
    periods_per_year : int
        252 for daily returns, 52 for weekly.
    min_window_size : int
        Minimum observations per window; raises if violated.
    oos_hit_rate_pass : float
        Fraction of OOS-positive windows required to PASS.
    oos_hit_rate_warn : float
        Fraction required to WARN (vs FAIL).

    Returns
    -------
    WalkForwardResult
    """
    ret = np.asarray(returns, dtype=float)
    ret = ret[np.isfinite(ret)]
    n = len(ret)

    total_windows = n_splits + 1
    window_size = n // total_windows

    if window_size < min_window_size:
        raise ValueError(
            f"Window size ({window_size}) < min_window_size ({min_window_size}). "
            f"Need at least {total_windows * min_window_size} observations for {n_splits} splits."
        )

    window_results: list[WindowResult] = []
    is_sharpes: list[float] = []
    oos_sharpes: list[float] = []

    for i in range(n_splits):
        is_start = i * window_size
        is_end = (i + 1) * window_size
        oos_start = is_end
        oos_end = min((i + 2) * window_size, n)

        is_ret = pd.Series(ret[is_start:is_end])
        oos_ret = pd.Series(ret[oos_start:oos_end])

        is_sharpe = _sharpe(is_ret, periods_per_year)
        oos_sharpe = _sharpe(oos_ret, periods_per_year)

        try:
            dsr_result = deflated_sharpe_ratio(is_ret, n_trials=1)
            is_dsr = float(dsr_result.get("dsr", 0.0))
        except Exception:
            is_dsr = 0.0

        wr = WindowResult(
            window_idx=i,
            is_sharpe=round(is_sharpe, 6),
            oos_sharpe=round(oos_sharpe, 6),
            is_dsr=round(is_dsr, 6),
            oos_positive=oos_sharpe > 0,
        )
        window_results.append(wr)
        is_sharpes.append(is_sharpe)
        oos_sharpes.append(oos_sharpe)

    oos_hit_rate = sum(1 for w in window_results if w.oos_positive) / n_splits

    # IS/OOS Sharpe correlation
    if n_splits >= 2 and np.std(is_sharpes) > 1e-12 and np.std(oos_sharpes) > 1e-12:
        is_oos_corr = float(np.corrcoef(is_sharpes, oos_sharpes)[0, 1])
    else:
        is_oos_corr = 0.0

    mean_oos = float(np.mean(oos_sharpes))
    mean_is = float(np.mean(is_sharpes))

    verdict, notes = _verdict(oos_hit_rate, is_oos_corr, mean_oos, oos_hit_rate_pass, oos_hit_rate_warn)

    return WalkForwardResult(
        n_windows=n_splits,
        windows=window_results,
        oos_hit_rate=round(oos_hit_rate, 4),
        is_oos_correlation=round(is_oos_corr, 4),
        mean_oos_sharpe=round(mean_oos, 6),
        mean_is_sharpe=round(mean_is, 6),
        verdict=verdict,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sharpe(ret: pd.Series, periods_per_year: int) -> float:
    arr = np.asarray(ret, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma < 1e-12:
        return 0.0
    return mu / sigma * math.sqrt(periods_per_year)


def _verdict(
    oos_hit_rate: float,
    is_oos_corr: float,
    mean_oos_sharpe: float,
    pass_threshold: float,
    warn_threshold: float,
) -> tuple[str, str]:
    if mean_oos_sharpe < 0:
        return "FAIL", f"Mean OOS Sharpe negative ({mean_oos_sharpe:.3f})"
    if oos_hit_rate >= pass_threshold and is_oos_corr > 0:
        return "PASS", (
            f"OOS hit rate {oos_hit_rate:.0%} >= {pass_threshold:.0%}, "
            f"IS/OOS corr={is_oos_corr:.3f}"
        )
    if oos_hit_rate >= pass_threshold:
        return "PASS", (
            f"OOS hit rate {oos_hit_rate:.0%} >= {pass_threshold:.0%} "
            f"(IS/OOS corr={is_oos_corr:.3f} non-positive)"
        )
    if oos_hit_rate >= warn_threshold:
        return "WARN", (
            f"OOS hit rate {oos_hit_rate:.0%} between {warn_threshold:.0%}"
            f" and {pass_threshold:.0%}"
        )
    return "FAIL", f"OOS hit rate {oos_hit_rate:.0%} < {warn_threshold:.0%}"
