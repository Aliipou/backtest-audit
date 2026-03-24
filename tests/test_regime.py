"""Tests for regime module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_audit.regime import (
    RegimeAuditResult,
    _ewma_vol,
    regime_audit,
)


def _returns(n: int = 300, mu: float = 0.001, sigma: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


class TestRegimeAudit:
    def test_returns_dataclass(self):
        result = regime_audit(_returns(n=200))
        assert isinstance(result, RegimeAuditResult)

    def test_slices_have_regime_names(self):
        result = regime_audit(_returns(n=300), classify_trend=False)
        names = {s.regime for s in result.slices}
        assert {"low_vol", "normal_vol", "high_vol"}.issubset(names | {"SKIP"})

    def test_trend_regimes_present_when_enabled(self):
        result = regime_audit(_returns(n=300), classify_trend=True)
        names = {s.regime for s in result.slices}
        assert "trend_up" in names or "trend_down" in names

    def test_overall_verdict_valid(self):
        result = regime_audit(_returns(n=300))
        assert result.overall_verdict in ("ROBUST", "FRAGILE", "BROKEN", "FAIL")

    def test_consistency_score_in_unit_interval(self):
        result = regime_audit(_returns(n=300))
        assert 0.0 <= result.consistency_score <= 1.0

    def test_n_counts_sum(self):
        result = regime_audit(_returns(n=300), classify_trend=False)
        non_skip = result.n_pass + result.n_warn + result.n_fail
        assert non_skip + result.n_skip == len(result.slices)

    def test_slice_verdict_valid_values(self):
        result = regime_audit(_returns(n=300))
        for s in result.slices:
            assert s.verdict in ("PASS", "WARN", "FAIL", "SKIP")

    def test_to_dict_has_required_keys(self):
        result = regime_audit(_returns(n=200), classify_trend=False)
        d = result.to_dict()
        assert "overall_verdict" in d
        assert "slices" in d
        assert "consistency_score" in d

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            regime_audit(pd.Series([0.01, 0.02]))

    def test_constant_series_raises(self):
        with pytest.raises(ValueError):
            regime_audit(pd.Series([0.0] * 100))

    def test_n_permutations_respected(self):
        # Should run without error with minimal permutations
        result = regime_audit(_returns(n=200), n_permutations=10)
        assert isinstance(result, RegimeAuditResult)


class TestEwmaVol:
    def test_output_length_matches_input(self):
        arr = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
        vol = _ewma_vol(arr, span=3)
        assert len(vol) == len(arr)

    def test_all_positive(self):
        arr = np.random.default_rng(0).normal(0, 0.01, 100)
        vol = _ewma_vol(arr, span=10)
        assert np.all(vol >= 0)
