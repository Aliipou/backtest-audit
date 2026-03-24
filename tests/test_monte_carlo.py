"""Tests for monte_carlo module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_audit.monte_carlo import _sharpe, monte_carlo_permutation_test


class TestSharpe:
    def test_basic(self):
        assert _sharpe(np.array([0.01, 0.02, 0.015])) > 0

    def test_zero_std(self):
        assert _sharpe(np.zeros(5)) == 0.0


class TestMonteCarlo:
    def test_result_keys(self, good_returns):
        r = monte_carlo_permutation_test(good_returns, n_permutations=100)
        assert {"sharpe", "percentile", "pvalue", "verdict"} == set(r)

    def test_percentile_in_range(self, good_returns):
        r = monte_carlo_permutation_test(good_returns, n_permutations=200)
        assert 0.0 <= r["percentile"] <= 1.0

    def test_reproducible_with_seed(self, good_returns):
        r1 = monte_carlo_permutation_test(good_returns, n_permutations=100, random_state=7)
        r2 = monte_carlo_permutation_test(good_returns, n_permutations=100, random_state=7)
        assert r1["percentile"] == r2["percentile"]

    def test_raises_too_short(self):
        with pytest.raises(ValueError):
            monte_carlo_permutation_test(pd.Series([0.01]))

    def test_raises_bad_n_permutations(self, short_returns):
        with pytest.raises(ValueError):
            monte_carlo_permutation_test(short_returns, n_permutations=0)

    def test_list_input(self):
        r = monte_carlo_permutation_test([0.01, -0.005, 0.003, 0.002], n_permutations=50)
        assert "verdict" in r

    def test_verdict_values(self, good_returns):
        r = monte_carlo_permutation_test(good_returns, n_permutations=200)
        assert r["verdict"] in ("PASS", "WARN", "FAIL")
