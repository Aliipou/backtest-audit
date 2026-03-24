"""Tests for economic_significance module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_audit.economic_significance import (
    EconomicSignificanceResult,
    _cohens_d_label,
    economic_significance,
)


def _make_returns(n: int = 300, mu: float = 0.001, sigma: float = 0.01, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


class TestEconomicSignificance:
    def test_returns_dataclass(self):
        ret = _make_returns()
        result = economic_significance(ret)
        assert isinstance(result, EconomicSignificanceResult)

    def test_strong_edge_gets_strong_verdict(self):
        # Sharpe ~1.6, clear positive edge
        ret = _make_returns(n=500, mu=0.001, sigma=0.01)
        result = economic_significance(ret, min_sharpe_threshold=0.5)
        assert result.verdict in ("STRONG", "MARGINAL")

    def test_negative_edge_gets_weak_verdict(self):
        ret = _make_returns(n=300, mu=-0.002, sigma=0.01)
        result = economic_significance(ret)
        assert result.verdict == "WEAK"

    def test_annualised_return_positive(self):
        ret = _make_returns(mu=0.001)
        result = economic_significance(ret)
        assert result.annualised_return > 0

    def test_annualised_vol_positive(self):
        ret = _make_returns()
        result = economic_significance(ret)
        assert result.annualised_vol > 0

    def test_sharpe_ratio_finite(self):
        ret = _make_returns()
        result = economic_significance(ret)
        assert np.isfinite(result.sharpe_ratio)

    def test_cohens_d_positive_for_positive_mu(self):
        ret = _make_returns(mu=0.005)
        result = economic_significance(ret)
        assert result.cohens_d > 0

    def test_r_squared_in_unit_interval(self):
        ret = _make_returns()
        result = economic_significance(ret)
        assert 0.0 <= result.r_squared <= 1.0

    def test_mde_sharpe_positive(self):
        ret = _make_returns()
        result = economic_significance(ret)
        assert result.mde_sharpe > 0

    def test_break_even_cost_nonneg(self):
        ret = _make_returns(mu=0.001)
        result = economic_significance(ret)
        assert result.break_even_cost_bps >= 0

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            economic_significance(pd.Series([0.01]))

    def test_constant_series_zero_cohens_d(self):
        # constant returns: std=0 → cohen's d=0, verdict WEAK
        ret = pd.Series([0.001] * 100)
        result = economic_significance(ret)
        assert result.cohens_d == 0.0
        assert result.verdict == "WEAK"


class TestCohensD:
    @pytest.mark.parametrize("d,expected", [
        (0.1, "negligible"),
        (0.3, "small"),
        (0.6, "medium"),
        (1.0, "large"),
    ])
    def test_labels(self, d, expected):
        assert _cohens_d_label(d) == expected
