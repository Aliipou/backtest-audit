"""
BacktestAuditor — orchestrates all statistical overfitting tests.

Usage
-----
>>> import pandas as pd
>>> from backtest_audit import BacktestAuditor
>>>
>>> returns = pd.Series([...])          # daily P&L returns
>>> auditor = BacktestAuditor(returns, n_trials=50)
>>> report  = auditor.run_all()
>>> report.print_report()
>>> print(report.overall_verdict)       # "PASS" | "WARN" | "FAIL"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .deflated_sharpe import deflated_sharpe_ratio
from .economic_significance import EconomicSignificanceResult, economic_significance
from .monte_carlo import monte_carlo_permutation_test
from .regime import RegimeAuditResult, regime_audit
from .robustness import RobustnessReport, RobustnessTester
from .walk_forward import WalkForwardResult, walk_forward_validation

# ---------------------------------------------------------------------------
# AuditReport
# ---------------------------------------------------------------------------

@dataclass
class AuditReport:
    """
    Container for the results of all overfitting tests run by BacktestAuditor.

    Attributes
    ----------
    dsr_result              : Result dict from deflated_sharpe_ratio().
    monte_carlo_result      : Result dict from monte_carlo_permutation_test().
    pbo_result              : Result dict from probability_of_backtest_overfitting(),
                              or None if not run (requires returns_matrix input).
    sensitivity_result      : Result dict from parameter_sensitivity(),
                              or None if not run (requires param grid input).
    economic_result         : EconomicSignificanceResult — effect size, MDE, R².
    walk_forward_result     : WalkForwardResult — OOS consistency.
    regime_result           : RegimeAuditResult — per-regime verdicts.
    robustness_report       : RobustnessReport — stress-test survival.
    """

    dsr_result: dict = field(default_factory=dict)
    monte_carlo_result: dict = field(default_factory=dict)
    pbo_result: dict | None = None
    sensitivity_result: dict | None = None

    # New: production-grade additions
    economic_result: EconomicSignificanceResult | None = None
    walk_forward_result: WalkForwardResult | None = None
    regime_result: RegimeAuditResult | None = None
    robustness_report: RobustnessReport | None = None

    # ------------------------------------------------------------------
    # overall_verdict property
    # ------------------------------------------------------------------

    @property
    def overall_verdict(self) -> str:
        """
        Aggregate verdict across all tests that were run.

        Rules:
          - "FAIL"  if *any* core test returned FAIL
          - "WARN"  if *any* test returned WARN (and none failed)
          - "PASS"  if *all* tests returned PASS
        """
        verdicts = []

        # Core statistical tests
        for result in (self.dsr_result, self.monte_carlo_result, self.pbo_result, self.sensitivity_result):
            if result:
                verdicts.append(result.get("verdict", "FAIL"))

        # Economic significance
        if self.economic_result:
            es = self.economic_result
            if es.verdict == "WEAK":
                verdicts.append("FAIL")
            elif es.verdict == "MARGINAL":
                verdicts.append("WARN")
            else:
                verdicts.append("PASS")

        # Walk-forward OOS
        if self.walk_forward_result:
            verdicts.append(self.walk_forward_result.verdict)

        # Regime consistency  (ROBUST→PASS, FRAGILE→WARN, BROKEN→FAIL)
        if self.regime_result:
            regime_map = {"ROBUST": "PASS", "FRAGILE": "WARN", "BROKEN": "FAIL"}
            verdicts.append(regime_map.get(self.regime_result.overall_verdict, "FAIL"))

        # Robustness stress test  (ROBUST→PASS, FRAGILE→WARN, BROKEN→FAIL)
        if self.robustness_report:
            rob_map = {"ROBUST": "PASS", "FRAGILE": "WARN", "BROKEN": "FAIL"}
            verdicts.append(rob_map.get(self.robustness_report.overall_verdict, "FAIL"))

        if not verdicts:
            return "FAIL"
        if "FAIL" in verdicts:
            return "FAIL"
        if "WARN" in verdicts:
            return "WARN"
        return "PASS"

    # ------------------------------------------------------------------
    # summary / serialisation helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a one-line human-readable summary string."""
        parts: list[str] = []
        if self.dsr_result:
            parts.append(f"DSR={self.dsr_result['dsr']:.3f} [{self.dsr_result['verdict']}]")
        if self.monte_carlo_result:
            parts.append(
                f"MC%ile={self.monte_carlo_result['percentile']:.3f} "
                f"[{self.monte_carlo_result['verdict']}]"
            )
        if self.pbo_result:
            parts.append(f"PBO={self.pbo_result['pbo']:.3f} [{self.pbo_result['verdict']}]")
        if self.sensitivity_result:
            parts.append(
                f"Sensitivity={self.sensitivity_result['sensitivity_ratio']:.3f} "
                f"[{self.sensitivity_result['verdict']}]"
            )
        if self.economic_result:
            e = self.economic_result
            parts.append(f"Eco:Sharpe={e.sharpe_ratio:.2f},d={e.cohens_d:.2f} [{e.verdict}]")
        if self.walk_forward_result:
            wf = self.walk_forward_result
            parts.append(f"WF:OOS_hit={wf.oos_hit_rate:.0%} [{wf.verdict}]")
        if self.regime_result:
            rg = self.regime_result
            parts.append(f"Regime:consistency={rg.consistency_score:.0%} [{rg.overall_verdict}]")
        if self.robustness_report:
            rb = self.robustness_report
            parts.append(f"Robust:survived={rb.n_survived}/7 [{rb.overall_verdict}]")
        verdict_str = f"Overall: {self.overall_verdict}"
        return " | ".join(parts + [verdict_str])

    def to_dict(self) -> dict:
        """Serialise to a plain dict (suitable for JSON export)."""
        result: dict = {
            "dsr": self.dsr_result,
            "monte_carlo": self.monte_carlo_result,
            "pbo": self.pbo_result,
            "sensitivity": self.sensitivity_result,
            "overall_verdict": self.overall_verdict,
        }
        if self.economic_result:
            e = self.economic_result
            result["economic_significance"] = {
                "cohens_d": e.cohens_d,
                "effect_size_label": e.effect_size_label,
                "mde_sharpe": e.mde_sharpe,
                "annualised_return": e.annualised_return,
                "annualised_vol": e.annualised_vol,
                "sharpe_ratio": e.sharpe_ratio,
                "break_even_cost_bps": e.break_even_cost_bps,
                "r_squared": e.r_squared,
                "verdict": e.verdict,
                "notes": e.notes,
            }
        if self.walk_forward_result:
            result["walk_forward"] = self.walk_forward_result.to_dict()
        if self.regime_result:
            result["regime_audit"] = self.regime_result.to_dict()
        if self.robustness_report:
            result["robustness"] = self.robustness_report.to_dict()
        return result

    # ------------------------------------------------------------------
    # Terminal pretty-print
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a formatted table of all test results to stdout."""
        _PASS_ICON = "[PASS]"
        _WARN_ICON = "[WARN]"
        _FAIL_ICON = "[FAIL]"
        _SKIP_ICON = "[SKIP]"

        def _icon(v: str) -> str:
            # normalize module-specific verdicts to PASS/WARN/FAIL
            _norm = {
                "STRONG": "PASS", "MARGINAL": "WARN", "WEAK": "FAIL",
                "ROBUST": "PASS", "FRAGILE": "WARN", "BROKEN": "FAIL",
                "SURVIVE": "PASS", "DEGRADE": "WARN", "COLLAPSE": "FAIL",
            }
            v = _norm.get(v, v)
            return {"PASS": _PASS_ICON, "WARN": _WARN_ICON, "FAIL": _FAIL_ICON}.get(v, _SKIP_ICON)

        width = 72
        border = "=" * width
        thin = "-" * width

        print(border)
        print("  BACKTEST OVERFITTING AUDIT REPORT".center(width))
        print(border)

        def _row(label: str, value: str, verdict: str) -> None:
            icon = _icon(verdict)
            print(f"  {label:<34} {value:<20} {icon}")

        if self.dsr_result:
            r = self.dsr_result
            _row("Deflated Sharpe Ratio (DSR)", f"{r['dsr']:.4f}  (SR={r['obs_sharpe']:.4f})", r["verdict"])
            _row("  Benchmark SR (n_trials)", f"{r['benchmark_sharpe']:.4f}", r["verdict"])
            _row("  p-value", f"{r['pvalue']:.4f}", r["verdict"])

        if self.monte_carlo_result:
            print(thin)
            r = self.monte_carlo_result
            _row("Monte Carlo Permutation Test", f"pctile={r['percentile']:.4f}", r["verdict"])
            _row("  Observed Sharpe", f"{r['sharpe']:.4f}", r["verdict"])
            _row("  p-value", f"{r['pvalue']:.4f}", r["verdict"])

        if self.pbo_result:
            print(thin)
            r = self.pbo_result
            _row(
                "Probability of Backtest Overfitting",
                f"{r['pbo']:.4f}  (N={r['n_combinations']})",
                r["verdict"],
            )

        if self.sensitivity_result:
            print(thin)
            r = self.sensitivity_result
            _row("Parameter Sensitivity Ratio", f"{r['sensitivity_ratio']:.4f}", r["verdict"])
            _row("  Stable region size", str(r["stable_region_size"]), r["verdict"])
            _row("  Best params", r["best_params"][:20], r["verdict"])

        if self.economic_result:
            print(thin)
            e = self.economic_result
            _row("Economic Significance", f"Sharpe={e.sharpe_ratio:.3f}", e.verdict)
            _row("  Cohen's d", f"{e.cohens_d:.4f} ({e.effect_size_label})", e.verdict)
            _row("  R^2 (signal variance)", f"{e.r_squared:.4f}", e.verdict)
            _row("  Ann. Return", f"{e.annualised_return:.2%}", e.verdict)
            _row("  Break-even cost (bps)", f"{e.break_even_cost_bps:.2f}", e.verdict)
            _row("  MDE Sharpe", f"{e.mde_sharpe:.4f}", e.verdict)

        if self.walk_forward_result:
            print(thin)
            wf = self.walk_forward_result
            _row("Walk-Forward OOS Validation", f"hit={wf.oos_hit_rate:.0%}", wf.verdict)
            _row("  IS/OOS Correlation", f"{wf.is_oos_correlation:.4f}", wf.verdict)
            _row("  Mean OOS Sharpe", f"{wf.mean_oos_sharpe:.4f}", wf.verdict)
            _row("  Windows tested", str(wf.n_windows), wf.verdict)

        if self.regime_result:
            print(thin)
            rg = self.regime_result
            _row(
                "Regime Consistency",
                f"{rg.consistency_score:.0%} consistent",
                rg.overall_verdict,
            )
            for s in rg.slices:
                _row(f"  [{s.regime}]", f"n={s.n_obs} SR={s.sharpe:.3f}", s.verdict)

        if self.robustness_report:
            print(thin)
            rb = self.robustness_report
            _row(
                "Robustness (stress test)",
                f"{rb.n_survived}/7 survived",
                rb.overall_verdict,
            )
            for r in rb.results:
                _row(f"  [{r.scenario}]", f"ret={r.sr_retention:.0%}", r.verdict)

        print(border)
        ov = self.overall_verdict
        print(f"  OVERALL VERDICT: {ov}  {_icon(ov)}".center(width))
        print(border)


# ---------------------------------------------------------------------------
# BacktestAuditor
# ---------------------------------------------------------------------------

class BacktestAuditor:
    """
    Orchestrates all statistical overfitting tests for a single strategy.

    Parameters
    ----------
    returns : pd.Series
        Strategy return series (one value per period, e.g. daily).
    n_trials : int
        Number of strategy variants / parameter combinations that were
        evaluated before selecting this strategy.  Used by the DSR test.
        Defaults to 1 (no multiple-testing correction).
    """

    def __init__(self, returns: pd.Series, n_trials: int = 1) -> None:
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        self._returns = returns.dropna()
        if len(self._returns) < 2:
            raise ValueError("At least 2 non-NaN return observations are required.")
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1.")
        self._n_trials = n_trials

    # ------------------------------------------------------------------
    # Individual test runners
    # ------------------------------------------------------------------

    def run_dsr(self, skewness: float = 0.0, kurtosis: float = 3.0) -> dict:
        """Run the Deflated Sharpe Ratio test."""
        return deflated_sharpe_ratio(
            self._returns,
            n_trials=self._n_trials,
            skewness=skewness,
            kurtosis=kurtosis,
        )

    def run_monte_carlo(self, n_permutations: int = 1000) -> dict:
        """Run the Monte Carlo permutation test."""
        return monte_carlo_permutation_test(self._returns, n_permutations=n_permutations)

    def run_economic_significance(self, periods_per_year: int = 252) -> EconomicSignificanceResult:
        """Run economic significance analysis — effect size, MDE, R²."""
        return economic_significance(self._returns, periods_per_year=periods_per_year)

    def run_walk_forward(self, n_splits: int = 5, periods_per_year: int = 252) -> WalkForwardResult:
        """Run walk-forward OOS validation."""
        return walk_forward_validation(
            self._returns, n_splits=n_splits, periods_per_year=periods_per_year
        )

    def run_regime_audit(self, periods_per_year: int = 252, n_permutations: int = 200) -> RegimeAuditResult:
        """Run regime-conditional audit (DSR + MC per vol/trend regime)."""
        return regime_audit(
            self._returns,
            periods_per_year=periods_per_year,
            n_permutations=n_permutations,
            n_trials=self._n_trials,
        )

    def run_robustness(self, periods_per_year: int = 252) -> RobustnessReport:
        """Run backtest robustness stress test."""
        tester = RobustnessTester(self._returns, periods_per_year=periods_per_year)
        return tester.run_all()

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------

    def run_all(
        self,
        n_permutations: int = 1000,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        periods_per_year: int = 252,
        include_walk_forward: bool = True,
        include_regime: bool = True,
        include_robustness: bool = True,
        include_economic: bool = True,
        wf_n_splits: int = 5,
    ) -> AuditReport:
        """
        Run all available audit tests and return a complete AuditReport.

        To include PBO or sensitivity results, attach them manually:
            report = auditor.run_all()
            from backtest_audit.pbo import probability_of_backtest_overfitting
            report.pbo_result = probability_of_backtest_overfitting(matrix)

        Parameters
        ----------
        n_permutations : int      Passed to MC and regime tests.
        skewness / kurtosis       Passed to DSR.
        periods_per_year : int    252 daily / 52 weekly.
        include_walk_forward      Run OOS walk-forward validation.
        include_regime            Run per-regime audit.
        include_robustness        Run stress-test robustness battery.
        include_economic          Run economic significance analysis.
        wf_n_splits               Number of IS/OOS splits for walk-forward.
        """
        dsr_result = self.run_dsr(skewness=skewness, kurtosis=kurtosis)
        mc_result = self.run_monte_carlo(n_permutations=n_permutations)

        eco = self.run_economic_significance(periods_per_year) if include_economic else None

        wf = None
        if include_walk_forward and len(self._returns) >= (wf_n_splits + 1) * 20:
            try:
                wf = self.run_walk_forward(n_splits=wf_n_splits, periods_per_year=periods_per_year)
            except ValueError:
                wf = None

        regime = None
        if include_regime and len(self._returns) >= 40:
            try:
                regime = self.run_regime_audit(
                    periods_per_year=periods_per_year,
                    n_permutations=min(n_permutations, 200),
                )
            except ValueError:
                regime = None

        rob = None
        if include_robustness and len(self._returns) >= 10:
            try:
                rob = self.run_robustness(periods_per_year=periods_per_year)
            except ValueError:
                rob = None

        return AuditReport(
            dsr_result=dsr_result,
            monte_carlo_result=mc_result,
            economic_result=eco,
            walk_forward_result=wf,
            regime_result=regime,
            robustness_report=rob,
        )
