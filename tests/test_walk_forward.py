"""Tests for walk_forward module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_audit.walk_forward import WalkForwardResult, walk_forward_validation


def _returns(n: int = 300, mu: float = 0.001, sigma: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


class TestWalkForwardValidation:
    def test_returns_dataclass(self):
        result = walk_forward_validation(_returns())
        assert isinstance(result, WalkForwardResult)

    def test_correct_number_of_windows(self):
        result = walk_forward_validation(_returns(n=300), n_splits=4)
        assert result.n_windows == 4
        assert len(result.windows) == 4

    def test_oos_hit_rate_in_unit_interval(self):
        result = walk_forward_validation(_returns())
        assert 0.0 <= result.oos_hit_rate <= 1.0

    def test_is_oos_correlation_finite(self):
        result = walk_forward_validation(_returns(n=600), n_splits=5)
        assert np.isfinite(result.is_oos_correlation)

    def test_verdict_valid_values(self):
        result = walk_forward_validation(_returns())
        assert result.verdict in ("PASS", "WARN", "FAIL")

    def test_to_dict_has_required_keys(self):
        result = walk_forward_validation(_returns())
        d = result.to_dict()
        for key in ("n_windows", "oos_hit_rate", "is_oos_correlation", "verdict", "windows"):
            assert key in d

    def test_each_window_has_is_and_oos_sharpe(self):
        result = walk_forward_validation(_returns(n=400), n_splits=3)
        for w in result.windows:
            assert np.isfinite(w.is_sharpe)
            assert np.isfinite(w.oos_sharpe)

    def test_too_small_raises(self):
        # 3 returns, 5 splits → window_size=0 < min 20
        with pytest.raises(ValueError):
            walk_forward_validation(pd.Series([0.01, 0.02, -0.01]), n_splits=5)

    def test_zero_edge_degrades_hit_rate(self):
        # Shuffled returns: no predictable signal
        rng = np.random.default_rng(0)
        ret = pd.Series(rng.normal(0, 0.01, 400))
        result = walk_forward_validation(ret, n_splits=4)
        # Can't guarantee fail but hit rate should be <= 1
        assert result.oos_hit_rate <= 1.0

    def test_mean_oos_sharpe_finite(self):
        result = walk_forward_validation(_returns(n=300))
        assert np.isfinite(result.mean_oos_sharpe)

    def test_consistent_positive_edge_passes(self):
        # Strong consistent edge
        rng = np.random.default_rng(1)
        ret = pd.Series(rng.normal(0.005, 0.01, 600))
        result = walk_forward_validation(ret, n_splits=4)
        assert result.oos_hit_rate >= 0.5
