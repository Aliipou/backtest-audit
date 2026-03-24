"""Tests for pbo module."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest_audit.pbo import (
    _concat_blocks,
    _make_blocks,
    probability_of_backtest_overfitting,
)


class TestMakeBlocks:
    def test_block_count(self):
        assert len(_make_blocks(100, 4)) == 4

    def test_covers_all_indices(self):
        blocks = _make_blocks(100, 4)
        assert sorted(sum(blocks, [])) == list(range(100))

    def test_uneven_split(self):
        blocks = _make_blocks(10, 3)
        assert sum(len(b) for b in blocks) == 10


class TestConcatBlocks:
    def test_basic(self):
        blocks = [[0, 1], [2, 3], [4, 5]]
        assert _concat_blocks(blocks, (0, 2)) == [0, 1, 4, 5]


class TestPBO:
    def test_result_keys(self, returns_matrix):
        r = probability_of_backtest_overfitting(returns_matrix, n_splits=4)
        assert {"pbo", "n_combinations", "verdict"} == set(r)

    def test_pbo_in_range(self, returns_matrix):
        r = probability_of_backtest_overfitting(returns_matrix, n_splits=4)
        assert 0.0 <= r["pbo"] <= 1.0

    def test_n_combinations(self, returns_matrix):
        r = probability_of_backtest_overfitting(returns_matrix, n_splits=4)
        assert r["n_combinations"] == math.comb(4, 2)

    def test_raises_single_strategy(self, good_returns):
        df = pd.DataFrame({"s0": good_returns})
        with pytest.raises(ValueError, match="2"):
            probability_of_backtest_overfitting(df, n_splits=4)

    def test_raises_odd_splits(self, returns_matrix):
        with pytest.raises(ValueError, match="even"):
            probability_of_backtest_overfitting(returns_matrix, n_splits=3)

    def test_verdict_values(self, returns_matrix):
        r = probability_of_backtest_overfitting(returns_matrix, n_splits=4)
        assert r["verdict"] in ("PASS", "WARN", "FAIL")

    def test_identical_strategies_low_pbo(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0.001, 0.01, 200)
        df = pd.DataFrame({f"s{i}": base + rng.normal(0, 1e-9, 200) for i in range(3)})
        r = probability_of_backtest_overfitting(df, n_splits=4)
        assert r["pbo"] <= 0.5
