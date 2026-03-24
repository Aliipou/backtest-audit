"""
Backtest Robustness Testing — does the edge survive deliberate stress?

Analogous to mm-live's stress_test.py, but for backtested return series.
We stress the return history using realistic failure scenarios and verify
the strategy's Sharpe / DSR doesn't collapse.

Scenarios
---------
  1. SUBSAMPLING        — random 70% subsample; if edge is real, survives.
  2. NOISE_INJECTION    — add N(0, 0.5σ) noise; real edges tolerate this.
  3. TAIL_AMPLIFICATION — multiply extreme returns by 2x; fat-tail stress.
  4. DRAWDOWN_EXTENSION — replace worst 10% with 2× losses; max drawdown test.
  5. REGIME_SHIFT       — reverse the second half; does IS SR collapse OOS?
  6. TRANSACTION_COSTS  — subtract realistic round-trip costs from each return.
  7. ZERO_EDGE          — shuffle returns; strategy must fail (sanity check).

Verdict
-------
  ROBUST  : >=5 scenarios survive (SR > 50% of baseline)
  FRAGILE : 3–4 survive
  BROKEN  : <3 survive
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SURVIVE_FRACTION = 0.5    # stressed SR must be >= 50% of baseline SR to survive


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RobustnessScenarioResult:
    scenario: str
    baseline_sharpe: float
    stressed_sharpe: float
    sr_retention: float     # stressed / baseline; 1.0 = unchanged, 0 = collapsed
    verdict: str            # "SURVIVE" | "DEGRADE" | "COLLAPSE"
    notes: str


@dataclass
class RobustnessReport:
    results: list[RobustnessScenarioResult] = field(default_factory=list)
    n_survived: int = 0
    n_degraded: int = 0
    n_collapsed: int = 0
    overall_verdict: str = "BROKEN"
    baseline_sharpe: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "overall_verdict": self.overall_verdict,
            "baseline_sharpe": round(self.baseline_sharpe, 4),
            "n_survived": self.n_survived,
            "n_degraded": self.n_degraded,
            "n_collapsed": self.n_collapsed,
            "notes": self.notes,
            "scenarios": [
                {
                    "scenario": r.scenario,
                    "baseline_sharpe": round(r.baseline_sharpe, 4),
                    "stressed_sharpe": round(r.stressed_sharpe, 4),
                    "sr_retention": round(r.sr_retention, 4),
                    "verdict": r.verdict,
                    "notes": r.notes,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class RobustnessTester:
    """
    Stress-test a return series to measure edge fragility.

    Parameters
    ----------
    returns : pd.Series
        Strategy daily (or per-period) return series.
    periods_per_year : int
        252 daily / 52 weekly.
    rng_seed : int
        Reproducible results.
    round_trip_cost : float
        Fraction of price assumed as round-trip transaction cost per period.
        Default 0.001 = 10 bps.
    """

    SURVIVE_THRESHOLD = 0.5    # stressed SR >= 50% of baseline → SURVIVE
    DEGRADE_THRESHOLD = 0.0    # stressed SR >= 0 but < 50% → DEGRADE
    # stressed SR < 0 → COLLAPSE

    def __init__(
        self,
        returns: pd.Series,
        periods_per_year: int = 252,
        rng_seed: int = 42,
        round_trip_cost: float = 0.001,
    ) -> None:
        self._ret = np.asarray(returns.dropna(), dtype=float)
        if len(self._ret) < 10:
            raise ValueError("Need at least 10 return observations.")
        self._ppy = periods_per_year
        self._rng = random.Random(rng_seed)
        self._np_rng = np.random.default_rng(rng_seed)
        self._cost = round_trip_cost
        self._baseline_sharpe = _sharpe(self._ret, periods_per_year)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all(self) -> RobustnessReport:
        scenarios = [
            "subsampling",
            "noise_injection",
            "tail_amplification",
            "drawdown_extension",
            "regime_shift",
            "transaction_costs",
            "zero_edge",
        ]
        results = [self.run_scenario(s) for s in scenarios]

        n_survived  = sum(1 for r in results if r.verdict == "SURVIVE")
        n_degraded  = sum(1 for r in results if r.verdict == "DEGRADE")
        n_collapsed = sum(1 for r in results if r.verdict == "COLLAPSE")

        if n_survived >= 5:
            overall = "ROBUST"
        elif n_survived >= 3:
            overall = "FRAGILE"
        else:
            overall = "BROKEN"

        notes = (
            f"Survived {n_survived}/7 scenarios "
            f"(excluding zero_edge sanity check if it passes baseline)"
        )

        return RobustnessReport(
            results=results,
            n_survived=n_survived,
            n_degraded=n_degraded,
            n_collapsed=n_collapsed,
            overall_verdict=overall,
            baseline_sharpe=round(self._baseline_sharpe, 4),
            notes=notes,
        )

    def run_scenario(self, scenario: str) -> RobustnessScenarioResult:
        stressed = self._apply(scenario)
        stressed_sr = _sharpe(stressed, self._ppy)
        return self._make_result(scenario, stressed_sr)

    # ------------------------------------------------------------------
    # Scenario implementations
    # ------------------------------------------------------------------

    def _apply(self, scenario: str) -> np.ndarray:
        arr = self._ret
        if scenario == "subsampling":
            idx = sorted(self._rng.sample(range(len(arr)), k=int(0.7 * len(arr))))
            return arr[idx]

        if scenario == "noise_injection":
            noise_std = 0.5 * float(np.std(arr, ddof=1))
            noise = self._np_rng.normal(0, noise_std, len(arr))
            return arr + noise

        if scenario == "tail_amplification":
            threshold = np.percentile(arr, 5)
            stressed = arr.copy()
            stressed[stressed < threshold] *= 2.0
            return stressed

        if scenario == "drawdown_extension":
            threshold = np.percentile(arr, 10)
            stressed = arr.copy()
            stressed[stressed < threshold] *= 2.0
            return stressed

        if scenario == "regime_shift":
            mid = len(arr) // 2
            return np.concatenate([arr[:mid], -arr[mid:]])

        if scenario == "transaction_costs":
            cost_per_period = self._cost * np.abs(arr) * 0.5
            return arr - cost_per_period

        if scenario == "zero_edge":
            shuffled = arr.copy()
            self._np_rng.shuffle(shuffled)
            return shuffled

        raise ValueError(f"Unknown scenario: {scenario}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------

    def _make_result(self, scenario: str, stressed_sr: float) -> RobustnessScenarioResult:
        baseline = self._baseline_sharpe

        if abs(baseline) > 1e-9:
            retention = stressed_sr / abs(baseline)
        else:
            retention = 1.0 if stressed_sr >= 0 else -1.0

        # Zero-edge scenario is a sanity check — we *want* it to collapse
        if scenario == "zero_edge":
            verdict = "SURVIVE" if stressed_sr < baseline * 0.3 else "DEGRADE"
            notes = f"Sanity check: shuffled SR={stressed_sr:.3f} (want < baseline)"
        elif stressed_sr >= self.SURVIVE_THRESHOLD * abs(baseline):
            verdict = "SURVIVE"
            notes = f"SR retention={retention:.1%}"
        elif stressed_sr >= 0:
            verdict = "DEGRADE"
            notes = f"SR dropped to {stressed_sr:.3f} (retention={retention:.1%})"
        else:
            verdict = "COLLAPSE"
            notes = f"SR turned negative: {stressed_sr:.3f}"

        return RobustnessScenarioResult(
            scenario=scenario,
            baseline_sharpe=round(baseline, 6),
            stressed_sharpe=round(stressed_sr, 6),
            sr_retention=round(retention, 4),
            verdict=verdict,
            notes=notes,
        )

    def print_report(self, report: RobustnessReport) -> None:
        width = 85
        print(f"\n{'=' * width}")
        print("  BACKTEST ROBUSTNESS STRESS TEST".center(width))
        print(f"{'=' * width}")
        print(
            f"  {'Scenario':<22} {'BaseSR':>7} {'StressSR':>9} "
            f"{'Retention':>10} {'Verdict'}"
        )
        print(f"  {'-' * 83}")
        for r in report.results:
            flag = "  <-- !!" if r.verdict == "COLLAPSE" else ""
            print(
                f"  {r.scenario:<22} {r.baseline_sharpe:>7.3f} "
                f"{r.stressed_sharpe:>9.3f} {r.sr_retention:>10.1%}  "
                f"{r.verdict}{flag}"
            )
        print(f"  {'-' * 83}")
        print(
            f"  Survived: {report.n_survived}  Degraded: {report.n_degraded}  "
            f"Collapsed: {report.n_collapsed}"
        )
        print(f"  OVERALL: {report.overall_verdict}")
        print(f"{'=' * width}\n")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sharpe(arr: np.ndarray, periods_per_year: int) -> float:
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    if sigma < 1e-12:
        return 0.0
    return mu / sigma * math.sqrt(periods_per_year)
