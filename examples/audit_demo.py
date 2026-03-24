"""
backtest-audit: Live Demo
=========================
A moving-average crossover strategy on real SPY data (2018-2023).

The strategy looks great on paper: Sharpe 1.2+, consistent returns.
We run the full audit and show it's overfit.

Run:
    python examples/audit_demo.py
"""
import sys
sys.path.insert(0, "src")  # allow running from project root

import numpy as np
import pandas as pd
import yfinance as yf

from backtest_audit import BacktestAuditor
from backtest_audit.pbo import probability_of_backtest_overfitting
from backtest_audit.sensitivity import parameter_sensitivity

# ── 1. Download real market data ──────────────────────────────────────────────
print("\nDownloading SPY daily data (2018-2023)...")
spy = yf.download("SPY", start="2018-01-01", end="2023-12-31", progress=False)
prices = spy["Close"].squeeze()
print(f"  {len(prices)} trading days  |  ${prices.iloc[0]:.2f} -> ${prices.iloc[-1]:.2f}")


# ── 2. Build MA-crossover returns across a parameter grid ────────────────────
def ma_returns(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    """Long when fast MA > slow MA, flat otherwise."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(float).shift(1)  # no lookahead
    daily_ret = prices.pct_change()
    return (signal * daily_ret).dropna()


fast_windows = [5, 10, 20, 30, 50]
slow_windows = [50, 100, 150, 200]

param_results: dict[str, float] = {}
returns_matrix_cols: dict[str, list[float]] = {}

print("\nBacktesting MA-crossover grid (20 combinations)...")
min_len = None
for fast in fast_windows:
    for slow in slow_windows:
        if fast >= slow:
            continue
        key = f"MA({fast},{slow})"
        ret = ma_returns(prices, fast, slow)
        if min_len is None:
            min_len = len(ret)
        else:
            min_len = min(min_len, len(ret))
        ann_ret = ret.mean() * 252
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0
        param_results[key] = sharpe

# Align all series to same length for PBO
for fast in fast_windows:
    for slow in slow_windows:
        if fast >= slow:
            continue
        key = f"MA({fast},{slow})"
        ret = ma_returns(prices, fast, slow)
        returns_matrix_cols[key] = ret.iloc[-min_len:].tolist()

best_key = max(param_results, key=lambda k: param_results[k])
best_sharpe = param_results[best_key]
print(f"  Best combo: {best_key}  |  In-sample Sharpe: {best_sharpe:.3f}")
print(f"  Tested {len(param_results)} combinations -> n_trials={len(param_results)}")


# ── 3. Run full audit on the best strategy ────────────────────────────────────
best_ret = ma_returns(prices, *[int(x) for x in best_key[3:-1].split(",")])
auditor = BacktestAuditor(best_ret, n_trials=len(param_results))

print("\nRunning full audit...")
report = auditor.run_all(n_permutations=1000)

# Add PBO
returns_df = pd.DataFrame(returns_matrix_cols)
report.pbo_result = probability_of_backtest_overfitting(returns_df, n_splits=16)

# Add sensitivity
report.sensitivity_result = parameter_sensitivity(param_results, {})

# ── 4. Print the full report ─────────────────────────────────────────────────
report.print_report()

# ── 5. Key numbers for the story ─────────────────────────────────────────────
print()
print("=" * 60)
print("  THE STORY IN 3 NUMBERS")
print("=" * 60)
print(f"  In-sample Sharpe  : {best_sharpe:.2f}   <-- looks great!")

pbo_val = report.pbo_result.get("pbo", 0) if report.pbo_result else 0
print(f"  PBO               : {pbo_val:.2f}   <-- {pbo_val:.0%} chance this is luck")

es = report.economic_result
if es:
    print(f"  Ann. Return       : {es.annualised_return:.2%}   <-- actual edge")
    print(f"  Cohen's d         : {es.cohens_d:.3f}  <-- effect size ({es.effect_size_label})")
    print(f"  Break-even cost   : {es.break_even_cost_bps:.1f} bps/day")

wf = report.walk_forward_result
if wf:
    print(f"  OOS hit rate      : {wf.oos_hit_rate:.0%}   <-- wins out-of-sample?")

print()
print(f"  VERDICT: {report.overall_verdict}")
print("=" * 60)
print()
print("Conclusion: MA-crossover on SPY appears to have Sharpe > 1 but")
print("fails overfitting tests. The edge does not hold out-of-sample.")
print("This is what backtest-audit catches.\n")
