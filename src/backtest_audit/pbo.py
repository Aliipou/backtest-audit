"""
Probability of Backtest Overfitting (PBO) — Lopez de Prado & Bailey (2014).

Uses Combinatorial Purged Cross-Validation (CPCV): all C(S, S/2) ways
to split S time-blocks into train and test halves are evaluated.  For
each split we ask: was the strategy that ranked best in-sample also the
best out-of-sample?  PBO is the fraction of splits where a *different*
strategy won out-of-sample.

Reference
---------
Bailey, D. H., & Lopez de Prado, M. (2014).
"The Probability of Backtest Overfitting."
Journal of Computational Finance (risk.net).
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd


def _sharpe(returns: np.ndarray) -> float:
    """Per-period Sharpe (mean / std).  Returns 0 if std == 0."""
    std = returns.std(ddof=1)
    if std < 1e-14:
        return 0.0
    return float(returns.mean() / std)


def probability_of_backtest_overfitting(
    returns_matrix: pd.DataFrame,
    n_splits: int = 16,
) -> dict:
    """
    Estimate the Probability of Backtest Overfitting (PBO).

    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Rows  = time periods (e.g. daily returns).
        Columns = strategy variants (different parameter sets).
        All strategies must share the same time index.
    n_splits : int
        Number of equal-length time blocks to divide the data into.
        Must be even and >= 2.  C(n_splits, n_splits//2) CV trials are run.
        Capped automatically when the combination count would exceed 2 000.

    Returns
    -------
    dict with keys:
        pbo             – Probability of Backtest Overfitting in [0, 1]
        n_combinations  – number of train/test splits evaluated
        verdict         – "PASS" | "WARN" | "FAIL"
    """
    df = pd.DataFrame(returns_matrix).dropna()
    n_obs, n_strategies = df.shape
    if n_strategies < 2:
        raise ValueError("returns_matrix must have at least 2 strategy columns.")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even.")

    # Cap n_splits to keep combination count manageable
    s = n_splits
    while s >= 2 and math.comb(s, s // 2) > 2000:
        s -= 2
    n_splits = max(s, 2)

    # Divide rows into n_splits blocks of (approximately) equal size
    block_indices = _make_blocks(n_obs, n_splits)

    all_block_ids = list(range(n_splits))
    half = n_splits // 2
    combos = list(combinations(all_block_ids, half))
    n_combinations = len(combos)

    mat = df.values  # shape (n_obs, n_strategies)

    overfit_count = 0

    for train_blocks in combos:
        test_blocks = tuple(b for b in all_block_ids if b not in train_blocks)

        train_idx = _concat_blocks(block_indices, train_blocks)
        test_idx = _concat_blocks(block_indices, test_blocks)

        train_mat = mat[train_idx, :]
        test_mat = mat[test_idx, :]

        # Best strategy in-sample (highest Sharpe)
        is_sharpes = np.array([_sharpe(train_mat[:, j]) for j in range(n_strategies)])
        best_is = int(np.argmax(is_sharpes))

        # Best strategy out-of-sample
        oos_sharpes = np.array([_sharpe(test_mat[:, j]) for j in range(n_strategies)])
        best_oos = int(np.argmax(oos_sharpes))

        if best_is != best_oos:
            overfit_count += 1

    pbo = overfit_count / n_combinations if n_combinations > 0 else 1.0

    if pbo < 0.3:
        verdict = "PASS"
    elif pbo < 0.5:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "pbo": round(pbo, 6),
        "n_combinations": n_combinations,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_blocks(n_obs: int, n_splits: int) -> list[list[int]]:
    """Divide range(n_obs) into n_splits consecutive blocks."""
    base = n_obs // n_splits
    remainder = n_obs % n_splits
    blocks: list[list[int]] = []
    start = 0
    for i in range(n_splits):
        size = base + (1 if i < remainder else 0)
        blocks.append(list(range(start, start + size)))
        start += size
    return blocks


def _concat_blocks(block_indices: list[list[int]], block_ids: tuple[int, ...]) -> list[int]:
    """Concatenate index lists for the given block ids."""
    idx: list[int] = []
    for bid in block_ids:
        idx.extend(block_indices[bid])
    return idx
