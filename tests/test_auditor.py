"""Tests for BacktestAuditor and AuditReport."""
from __future__ import annotations

import pandas as pd
import pytest

from backtest_audit import AuditReport, BacktestAuditor
from backtest_audit.economic_significance import EconomicSignificanceResult
from backtest_audit.regime import RegimeAuditResult
from backtest_audit.robustness import RobustnessReport
from backtest_audit.walk_forward import WalkForwardResult


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


def _eco(verdict: str) -> EconomicSignificanceResult:
    return EconomicSignificanceResult(
        cohens_d=0.5, effect_size_label="medium",
        mde_sharpe=0.3, n_obs=200,
        annualised_return=0.1, annualised_vol=0.15,
        sharpe_ratio=0.67, break_even_cost_bps=5.0,
        r_squared=0.05, verdict=verdict, notes="",
    )


def _wf(verdict: str) -> WalkForwardResult:
    return WalkForwardResult(
        n_windows=3, windows=[],
        oos_hit_rate=0.67, is_oos_correlation=0.3,
        mean_oos_sharpe=0.5, mean_is_sharpe=0.8,
        verdict=verdict, notes="",
    )


def _regime(verdict: str) -> RegimeAuditResult:
    return RegimeAuditResult(
        slices={}, n_pass=1, n_warn=0, n_fail=0, n_skip=0,
        overall_verdict=verdict, consistency_score=1.0, notes="",
    )


def _robustness(verdict: str) -> RobustnessReport:
    return RobustnessReport(
        results={}, n_survived=1, n_degraded=0, n_collapsed=0,
        overall_verdict=verdict, baseline_sharpe=1.0, notes="",
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


class TestOverallRiskScore:
    def test_no_data_returns_one(self):
        assert AuditReport().overall_risk_score() == 1.0

    def test_all_pass_near_zero(self):
        r = _report("PASS", "PASS")
        assert r.overall_risk_score() < 0.2

    def test_all_fail_near_one(self):
        r = _report("FAIL", "FAIL")
        assert r.overall_risk_score() >= 0.7

    def test_warn_is_between(self):
        score_pass = _report("PASS", "PASS").overall_risk_score()
        score_warn = _report("WARN", "WARN").overall_risk_score()
        score_fail = _report("FAIL", "FAIL").overall_risk_score()
        assert score_pass < score_warn < score_fail

    def test_in_unit_interval(self):
        for dsr in ("PASS", "WARN", "FAIL"):
            for mc in ("PASS", "WARN", "FAIL"):
                s = _report(dsr, mc).overall_risk_score()
                assert 0.0 <= s <= 1.0, f"out of range for dsr={dsr} mc={mc}: {s}"

    def test_economic_strong_lowers_score(self):
        r_base = _report("PASS", "PASS")
        r_eco = AuditReport(
            dsr_result=r_base.dsr_result,
            monte_carlo_result=r_base.monte_carlo_result,
            economic_result=_eco("STRONG"),
        )
        assert r_eco.overall_risk_score() <= r_base.overall_risk_score() + 0.05

    def test_economic_weak_raises_score(self):
        r_base = _report("PASS", "PASS")
        r_eco = AuditReport(
            dsr_result=r_base.dsr_result,
            monte_carlo_result=r_base.monte_carlo_result,
            economic_result=_eco("WEAK"),
        )
        assert r_eco.overall_risk_score() > r_base.overall_risk_score()

    def test_walk_forward_fail_raises_score(self):
        r_base = _report("PASS", "PASS")
        r_wf = AuditReport(
            dsr_result=r_base.dsr_result,
            monte_carlo_result=r_base.monte_carlo_result,
            walk_forward_result=_wf("FAIL"),
        )
        assert r_wf.overall_risk_score() > r_base.overall_risk_score()

    def test_regime_broken_raises_score(self):
        r_base = _report("PASS", "PASS")
        r_regime = AuditReport(
            dsr_result=r_base.dsr_result,
            monte_carlo_result=r_base.monte_carlo_result,
            regime_result=_regime("BROKEN"),
        )
        assert r_regime.overall_risk_score() > r_base.overall_risk_score()

    def test_robustness_broken_raises_score(self):
        r_base = _report("PASS", "PASS")
        r_rob = AuditReport(
            dsr_result=r_base.dsr_result,
            monte_carlo_result=r_base.monte_carlo_result,
            robustness_report=_robustness("BROKEN"),
        )
        assert r_rob.overall_risk_score() > r_base.overall_risk_score()

    def test_position_scale_formula(self):
        """overall_risk_score should be directly usable as 1 - score position multiplier."""
        r = _report("PASS", "PASS")
        scale = 1.0 - r.overall_risk_score()
        assert 0.0 <= scale <= 1.0


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
