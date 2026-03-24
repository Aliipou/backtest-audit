"""Tests for BacktestAuditor and AuditReport."""
from __future__ import annotations

import pandas as pd
import pytest

from backtest_audit import AuditReport, BacktestAuditor


def _report(dsr_v: str = "PASS", mc_v: str = "PASS") -> AuditReport:
    return AuditReport(
        dsr_result={
            "dsr": 1.0, "pvalue": 0.1, "obs_sharpe": 0.5,
            "benchmark_sharpe": 0.0, "verdict": dsr_v,
        },
        monte_carlo_result={
            "sharpe": 0.5, "percentile": 0.97, "pvalue": 0.03, "verdict": mc_v,
        },
    )


class TestAuditReport:
    def test_both_pass(self):
        assert _report("PASS", "PASS").overall_verdict == "PASS"

    def test_one_fail(self):
        assert _report("FAIL", "PASS").overall_verdict == "FAIL"

    def test_one_warn(self):
        assert _report("WARN", "PASS").overall_verdict == "WARN"

    def test_fail_beats_warn(self):
        assert _report("FAIL", "WARN").overall_verdict == "FAIL"

    def test_empty_fails(self):
        assert AuditReport().overall_verdict == "FAIL"

    def test_to_dict_keys(self):
        d = _report().to_dict()
        assert {"dsr", "monte_carlo", "pbo", "sensitivity", "overall_verdict"} == set(d)

    def test_summary_contains_verdict(self):
        assert "PASS" in _report().summary()

    def test_print_report(self, capsys):
        _report().print_report()
        assert "OVERALL VERDICT" in capsys.readouterr().out


class TestBacktestAuditor:
    def test_run_all(self, good_returns):
        report = BacktestAuditor(good_returns, n_trials=1).run_all(n_permutations=100)
        assert isinstance(report, AuditReport)
        assert report.overall_verdict in ("PASS", "WARN", "FAIL")

    def test_accepts_list(self):
        BacktestAuditor([0.01, -0.005, 0.003] * 20, n_trials=1)

    def test_raises_too_short(self):
        with pytest.raises(ValueError):
            BacktestAuditor(pd.Series([0.01]))

    def test_raises_bad_n_trials(self, good_returns):
        with pytest.raises(ValueError):
            BacktestAuditor(good_returns, n_trials=0)

    def test_run_dsr(self, good_returns):
        assert "dsr" in BacktestAuditor(good_returns, n_trials=5).run_dsr()

    def test_run_monte_carlo(self, good_returns):
        assert "percentile" in BacktestAuditor(good_returns).run_monte_carlo(n_permutations=100)
