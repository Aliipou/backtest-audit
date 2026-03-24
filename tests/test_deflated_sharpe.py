"""Tests for deflated_sharpe module."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from backtest_audit.deflated_sharpe import (
    _benchmark_sharpe,
    _sharpe_ratio,
    _variance_ratio_correction,
    deflated_sharpe_ratio,
)


class TestSharpeRatio:
    def test_positive_mean(self):
        assert _sharpe_ratio(pd.Series([0.01] * 10)) > 0

    def test_zero_std_returns_zero(self):
        assert _sharpe_ratio(pd.Series([0.0] * 5)) == 0.0

    def test_annualize(self):
        s = pd.Series([0.01] * 252)
        raw = _sharpe_ratio(s, annualize=False)
        ann = _sharpe_ratio(s, annualize=True)
        assert abs(ann - raw * math.sqrt(252)) < 1e-9


class TestBenchmarkSharpe:
    def test_n1_returns_zero(self):
        assert _benchmark_sharpe(1) == 0.0

    def test_n0_returns_zero(self):
        assert _benchmark_sharpe(0) == 0.0

    def test_increases_with_trials(self):
        assert _benchmark_sharpe(10) < _benchmark_sharpe(100)


class TestVarianceRatio:
    def test_iid_near_one(self, good_returns):
        vr = _variance_ratio_correction(good_returns)
        assert 0.5 < vr < 3.0

    def test_short_series_returns_one(self):
        assert _variance_ratio_correction(pd.Series([0.01, -0.01])) == 1.0


class TestDeflatedSharpeRatio:
    def test_good_strategy_passes(self, good_returns):
        r = deflated_sharpe_ratio(good_returns, n_trials=1)
        assert r["verdict"] == "PASS"

    def test_result_keys(self, short_returns):
        r = deflated_sharpe_ratio(short_returns, n_trials=1)
        assert {"dsr", "pvalue", "obs_sharpe", "benchmark_sharpe", "verdict"} == set(r)

    def test_pvalue_in_range(self, good_returns):
        r = deflated_sharpe_ratio(good_returns, n_trials=5)
        assert 0.0 <= r["pvalue"] <= 1.0

    def test_raises_too_short(self):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(pd.Series([0.01]), n_trials=1)

    def test_raises_bad_n_trials(self, short_returns):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(short_returns, n_trials=0)

    def test_list_input(self):
        r = deflated_sharpe_ratio([0.01, -0.005, 0.003, 0.002], n_trials=1)
        assert "verdict" in r

    def test_more_trials_lowers_dsr(self, good_returns):
        r1 = deflated_sharpe_ratio(good_returns, n_trials=1)
        r1000 = deflated_sharpe_ratio(good_returns, n_trials=1000)
        assert r1["dsr"] >= r1000["dsr"]
