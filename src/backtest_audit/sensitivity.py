"""
Parameter Sensitivity Analysis.

A robust backtest should not be sensitive to small changes in its
parameters.  If the Sharpe Ratio swings wildly as you move a
single parameter (e.g. a look-back window) by one step, the strategy
is probably fitting noise rather than signal.

This module quantifies that fragility via the *sensitivity ratio*:

    sensitivity_ratio = std(sharpes) / |mean(sharpes)|

A ratio close to 0 indicates a stable region (all parameter combinations
produce similar performance).  A high ratio indicates fragility.

We also report the size of the "stable region" — the number of parameter
combinations whose Sharpe Ratio is within one standard deviation of the
best observed Sharpe.
"""

from __future__ import annotations

import numpy as np


def parameter_sensitivity(
    results: dict[str, float],
    param_grid: dict[str, list],
) -> dict:
    """
    Measure the sensitivity of strategy performance to parameter choices.

    Parameters
    ----------
    results : dict[str, float]
        Maps each parameter-combination string to its Sharpe Ratio.
        Example: {"fast=5,slow=20": 1.2, "fast=5,slow=30": 0.8, ...}
    param_grid : dict[str, list]
        The grid of parameter values that was searched.
        Example: {"fast": [5, 10, 15], "slow": [20, 30, 40]}
        Used only to count the total grid size; need not exactly match
        the keys in *results*.

    Returns
    -------
    dict with keys:
        sensitivity_ratio  – std(sharpes) / |mean(sharpes)|; higher = more fragile
        best_params        – key from *results* with the highest Sharpe
        stable_region_size – number of combos within 1 std-dev of best Sharpe
        verdict            – "PASS" | "WARN" | "FAIL"
    """
    if not results:
        raise ValueError("results dict must not be empty.")

    sharpes = np.array(list(results.values()), dtype=float)
    keys = list(results.keys())

    mean_sharpe = float(np.mean(sharpes))
    std_sharpe = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0

    if abs(mean_sharpe) < 1e-10:
        # Avoid division by near-zero; treat as maximally sensitive
        sensitivity_ratio = float("inf") if std_sharpe > 1e-10 else 0.0
    else:
        sensitivity_ratio = std_sharpe / abs(mean_sharpe)

    # Best param combo
    best_idx = int(np.argmax(sharpes))
    best_params = keys[best_idx]
    best_sharpe = float(sharpes[best_idx])

    # Stable region: combos within 1 std-dev of the best Sharpe
    threshold = best_sharpe - std_sharpe
    stable_region_size = int(np.sum(sharpes >= threshold))

    # Verdict
    if sensitivity_ratio < 0.3:
        verdict = "PASS"
    elif sensitivity_ratio < 0.6:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "sensitivity_ratio": round(sensitivity_ratio, 6),
        "best_params": best_params,
        "stable_region_size": stable_region_size,
        "verdict": verdict,
    }
