"""Tests for sensitivity module."""
from __future__ import annotations

import math

import pytest

from backtest_audit.sensitivity import parameter_sensitivity


class TestParameterSensitivity:
    def test_result_keys(self):
        r = parameter_sensitivity({"a": 1.2, "b": 0.8, "c": 1.0}, {})
        assert {"sensitivity_ratio", "best_params", "stable_region_size", "verdict"} == set(r)

    def test_stable_passes(self):
        results = {f"p={i}": 1.0 + i * 0.001 for i in range(10)}
        r = parameter_sensitivity(results, {})
        assert r["verdict"] == "PASS"

    def test_fragile_fails(self):
        results = {"p=1": 2.0, "p=2": -1.5, "p=3": 3.0, "p=4": -2.0}
        r = parameter_sensitivity(results, {})
        assert r["verdict"] == "FAIL"

    def test_best_params(self):
        r = parameter_sensitivity({"a": 0.5, "b": 1.5, "c": 1.0}, {})
        assert r["best_params"] == "b"

    def test_single_result_sensitivity_zero(self):
        r = parameter_sensitivity({"p=1": 1.0}, {})
        assert r["sensitivity_ratio"] == 0.0

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="empty"):
            parameter_sensitivity({}, {})

    def test_near_zero_mean(self):
        r = parameter_sensitivity({"a": 0.0001, "b": -0.0001}, {})
        assert r["sensitivity_ratio"] == math.inf or r["sensitivity_ratio"] >= 0
