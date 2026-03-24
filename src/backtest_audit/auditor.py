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
from .monte_carlo import monte_carlo_permutation_test

# ---------------------------------------------------------------------------
# AuditReport
# ---------------------------------------------------------------------------

@dataclass
class AuditReport:
    """
    Container for the results of all overfitting tests run by BacktestAuditor.

    Attributes
    ----------
    dsr_result          : Result dict from deflated_sharpe_ratio().
    monte_carlo_result  : Result dict from monte_carlo_permutation_test().
    pbo_result          : Result dict from probability_of_backtest_overfitting(),
                          or None if not run (requires returns_matrix input).
    sensitivity_result  : Result dict from parameter_sensitivity(),
                          or None if not run (requires param grid input).
    """

    dsr_result: dict = field(default_factory=dict)
    monte_carlo_result: dict = field(default_factory=dict)
    pbo_result: dict | None = None
    sensitivity_result: dict | None = None

    # ------------------------------------------------------------------
    # overall_verdict property
    # ------------------------------------------------------------------

    @property
    def overall_verdict(self) -> str:
        """
        Aggregate verdict across all tests that were run.

        Rules:
          - "FAIL"  if *any* test returned FAIL
          - "WARN"  if *any* test returned WARN (and none failed)
          - "PASS"  if *all* tests returned PASS
        """
        verdicts = []
        for result in (
            self.dsr_result,
            self.monte_carlo_result,
            self.pbo_result,
            self.sensitivity_result,
        ):
            if result:
                verdicts.append(result.get("verdict", "FAIL"))

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
            parts.append(
                f"DSR={self.dsr_result['dsr']:.3f} [{self.dsr_result['verdict']}]"
            )
        if self.monte_carlo_result:
            parts.append(
                f"MC%ile={self.monte_carlo_result['percentile']:.3f} "
                f"[{self.monte_carlo_result['verdict']}]"
            )
        if self.pbo_result:
            parts.append(
                f"PBO={self.pbo_result['pbo']:.3f} [{self.pbo_result['verdict']}]"
            )
        if self.sensitivity_result:
            parts.append(
                f"Sensitivity={self.sensitivity_result['sensitivity_ratio']:.3f} "
                f"[{self.sensitivity_result['verdict']}]"
            )
        verdict_str = f"Overall: {self.overall_verdict}"
        return " | ".join(parts + [verdict_str])

    def to_dict(self) -> dict:
        """Serialise to a plain dict (suitable for JSON export)."""
        return {
            "dsr": self.dsr_result,
            "monte_carlo": self.monte_carlo_result,
            "pbo": self.pbo_result,
            "sensitivity": self.sensitivity_result,
            "overall_verdict": self.overall_verdict,
        }

    # ------------------------------------------------------------------
    # Terminal pretty-print
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a formatted table of all test results to stdout."""
        _PASS_ICON = "[PASS]"
        _WARN_ICON = "[WARN]"
        _FAIL_ICON = "[FAIL]"

        def _icon(v: str) -> str:
            return {
                "PASS": _PASS_ICON,
                "WARN": _WARN_ICON,
                "FAIL": _FAIL_ICON,
            }.get(v, "[????]")

        width = 72
        border = "=" * width
        thin = "-" * width

        print(border)
        print("  BACKTEST OVERFITTING AUDIT REPORT".center(width))
        print(border)

        def _row(label: str, value: str, verdict: str) -> None:
            icon = _icon(verdict)
            line = f"  {label:<34} {value:<20} {icon}"
            print(line)

        if self.dsr_result:
            r = self.dsr_result
            _row(
                "Deflated Sharpe Ratio (DSR)",
                f"{r['dsr']:.4f}  (SR={r['obs_sharpe']:.4f})",
                r["verdict"],
            )
            _row(
                "  Benchmark SR (n_trials)",
                f"{r['benchmark_sharpe']:.4f}",
                r["verdict"],
            )
            _row("  p-value", f"{r['pvalue']:.4f}", r["verdict"])

        if self.monte_carlo_result:
            print(thin)
            r = self.monte_carlo_result
            _row(
                "Monte Carlo Permutation Test",
                f"pctile={r['percentile']:.4f}",
                r["verdict"],
            )
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
            _row(
                "Parameter Sensitivity Ratio",
                f"{r['sensitivity_ratio']:.4f}",
                r["verdict"],
            )
            _row(
                "  Stable region size",
                str(r["stable_region_size"]),
                r["verdict"],
            )
            _row(
                "  Best params",
                r["best_params"][:20],
                r["verdict"],
            )

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

    def run_dsr(
        self,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> dict:
        """
        Run the Deflated Sharpe Ratio test.

        Parameters
        ----------
        skewness : float  Third moment of return distribution (default 0).
        kurtosis : float  Fourth moment (not excess; default 3).

        Returns
        -------
        dict — see deflated_sharpe_ratio() for keys.
        """
        return deflated_sharpe_ratio(
            self._returns,
            n_trials=self._n_trials,
            skewness=skewness,
            kurtosis=kurtosis,
        )

    def run_monte_carlo(self, n_permutations: int = 1000) -> dict:
        """
        Run the Monte Carlo permutation test.

        Parameters
        ----------
        n_permutations : int  Number of random shuffles (default 1000).

        Returns
        -------
        dict — see monte_carlo_permutation_test() for keys.
        """
        return monte_carlo_permutation_test(self._returns, n_permutations=n_permutations)

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------

    def run_all(
        self,
        n_permutations: int = 1000,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> AuditReport:
        """
        Run DSR and Monte Carlo tests and return an AuditReport.

        To include PBO or sensitivity results, run those tests separately
        (they require additional inputs — returns_matrix / param_grid) and
        attach them to the report manually:

            report = auditor.run_all()
            report.pbo_result = probability_of_backtest_overfitting(matrix)
            report.sensitivity_result = parameter_sensitivity(results, grid)

        Parameters
        ----------
        n_permutations : int  Passed to run_monte_carlo().
        skewness : float      Passed to run_dsr().
        kurtosis : float      Passed to run_dsr().

        Returns
        -------
        AuditReport
        """
        dsr_result = self.run_dsr(skewness=skewness, kurtosis=kurtosis)
        mc_result = self.run_monte_carlo(n_permutations=n_permutations)

        return AuditReport(
            dsr_result=dsr_result,
            monte_carlo_result=mc_result,
        )
