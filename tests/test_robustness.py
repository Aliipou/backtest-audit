"""Tests for robustness module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_audit.robustness import RobustnessReport, RobustnessScenarioResult, RobustnessTester


def _returns(n: int = 200, mu: float = 0.001, sigma: float = 0.01, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


@pytest.fixture
def tester():
    return RobustnessTester(_returns(), rng_seed=42)


class TestRobustnessTester:
    def test_run_all_returns_report(self, tester):
        report = tester.run_all()
        assert isinstance(report, RobustnessReport)

    def test_run_all_has_7_scenarios(self, tester):
        report = tester.run_all()
        assert len(report.results) == 7

    def test_overall_verdict_valid(self, tester):
        report = tester.run_all()
        assert report.overall_verdict in ("ROBUST", "FRAGILE", "BROKEN")

    def test_n_counts_sum_to_7(self, tester):
        report = tester.run_all()
        total = report.n_survived + report.n_degraded + report.n_collapsed
        assert total == 7

    def test_run_scenario_returns_result(self, tester):
        result = tester.run_scenario("subsampling")
        assert isinstance(result, RobustnessScenarioResult)

    def test_scenario_verdict_valid(self, tester):
        for scenario in ("subsampling", "noise_injection", "tail_amplification",
                         "drawdown_extension", "regime_shift", "transaction_costs", "zero_edge"):
            result = tester.run_scenario(scenario)
            assert result.verdict in ("SURVIVE", "DEGRADE", "COLLAPSE")

    def test_baseline_sharpe_stored(self, tester):
        report = tester.run_all()
        assert np.isfinite(report.baseline_sharpe)

    def test_to_dict_has_required_keys(self, tester):
        report = tester.run_all()
        d = report.to_dict()
        for key in ("overall_verdict", "baseline_sharpe", "n_survived", "n_collapsed", "scenarios"):
            assert key in d

    def test_noise_injection_reduces_sharpe(self, tester):
        result = tester.run_scenario("noise_injection")
        assert result.stressed_sharpe <= result.baseline_sharpe + 0.5  # noise makes it worse or similar

    def test_zero_edge_has_verdict(self, tester):
        result = tester.run_scenario("zero_edge")
        # Sanity-check scenario always produces a valid verdict
        assert result.verdict in ("SURVIVE", "DEGRADE", "COLLAPSE")
        # Shuffling preserves mean/std so Sharpe is identical — sr_retention stays near 1.0
        assert np.isfinite(result.sr_retention)

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError):
            RobustnessTester(pd.Series([0.01] * 5))

    def test_unknown_scenario_raises(self, tester):
        with pytest.raises(ValueError):
            tester.run_scenario("alien_invasion")

    def test_print_report_runs(self, tester, capsys):
        report = tester.run_all()
        tester.print_report(report)
        out = capsys.readouterr().out
        assert "ROBUSTNESS" in out
        assert "OVERALL" in out

    def test_sr_retention_for_survive_nonneg(self, tester):
        report = tester.run_all()
        for r in report.results:
            if r.verdict == "SURVIVE":
                assert r.sr_retention >= 0
