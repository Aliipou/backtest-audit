"""
Monte Carlo Permutation Test for Backtest Overfitting.

The null hypothesis is that the order of returns does not matter —
i.e., the strategy has no genuine timing skill.  We test this by
shuffling the return series N times and computing the Sharpe Ratio
for each permutation.  The p-value is the fraction of permuted
Sharpe Ratios that are *at least as large* as the observed one.

A high percentile rank (close to 1.0) means the real strategy performs
better than almost all random orderings of the same returns — evidence
of genuine edge.

Reference
---------
White, H. (2000). "A Reality Check for Data Snooping."
Econometrica, 68(5), 1097–1126.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sharpe(returns: np.ndarray) -> float:
    """Per-period Sharpe Ratio (mean / std, ddof=1)."""
    std = returns.std(ddof=1)
    if std < 1e-14:
        return 0.0
    return float(returns.mean() / std)


def monte_carlo_permutation_test(
    returns: pd.Series,
    n_permutations: int = 1000,
    random_state: int | None = None,
) -> dict:
    """
    Run a Monte Carlo permutation test on a strategy return series.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns (one value per period).
    n_permutations : int
        Number of random shuffles to generate the null distribution.
    random_state : int or None
        Seed for the NumPy random number generator (for reproducibility).

    Returns
    -------
    dict with keys:
        sharpe      – observed per-period Sharpe Ratio
        percentile  – fraction of permuted Sharpes <= observed Sharpe (0..1)
        pvalue      – fraction of permuted Sharpes >= observed Sharpe (0..1)
        verdict     – "PASS" | "WARN" | "FAIL"
    """
    arr = pd.Series(returns).dropna().to_numpy(dtype=float)
    if len(arr) < 2:
        raise ValueError("Need at least 2 non-NaN return observations.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1.")

    rng = np.random.default_rng(random_state)
    obs_sharpe = _sharpe(arr)

    # Build the null distribution
    null_sharpes = np.empty(n_permutations)
    perm = arr.copy()
    for i in range(n_permutations):
        rng.shuffle(perm)
        null_sharpes[i] = _sharpe(perm)

    # Percentile rank: fraction of null Sharpes <= observed
    percentile = float(np.mean(null_sharpes <= obs_sharpe))

    # Two-sided-ish p-value: fraction of null Sharpes >= observed
    pvalue = float(np.mean(null_sharpes >= obs_sharpe))

    if percentile > 0.95:
        verdict = "PASS"
    elif percentile > 0.85:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "sharpe": round(obs_sharpe, 6),
        "percentile": round(percentile, 6),
        "pvalue": round(pvalue, 6),
        "verdict": verdict,
    }
