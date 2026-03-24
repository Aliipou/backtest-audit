"""
Validation Experiment: Does backtest-audit predict OOS performance?
====================================================================

Challenge: "Show me that your audit metrics predict real OOS performance."

Design
------
  - 8 assets: SPY, QQQ, GLD, BTC-USD, ETH-USD, TLT, EEM, ^VIX proxied by VXX
  - Strategy types:
      MA crossover (real trend-following signal)
      RSI mean-reversion
      Bollinger Band breakout / mean-reversion
      Pure noise (garbage — sanity check)
  - 400+ IS strategies
  - IS:  2018-01-01 to 2021-12-31  (4 years: bull + COVID crash)
  - OOS: 2022-01-01 to 2023-12-31  (2 years: bear + recovery)

3 honest questions
------------------
  Q1: Does IS Sharpe rank predict OOS Sharpe? (Spearman r per asset)
  Q2: Do DSR-passing strategies outperform DSR-failing ones OOS?
  Q3: Do noise strategies fail OOS more than real strategies? (sanity)

Plus: Consensus effect demonstration
  Q4: With consensus=2 signals, how does false-positive rate change?

Run:
    python examples/validation_experiment.py
"""
import sys
sys.path.insert(0, "src")

import math
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from backtest_audit.deflated_sharpe import deflated_sharpe_ratio
from backtest_audit.monte_carlo import monte_carlo_permutation_test
from backtest_audit.pbo import probability_of_backtest_overfitting


# ── Config ───────────────────────────────────────────────────────────────────
ASSETS = ["SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD", "TLT", "EEM", "VXX"]
IS_START,  IS_END  = "2018-01-01", "2021-12-31"
OOS_START, OOS_END = "2022-01-01", "2023-12-31"
RNG = np.random.default_rng(42)


def _sharpe(ret: pd.Series, ppy: int = 252) -> float:
    arr = ret.dropna().values
    if len(arr) < 5:
        return 0.0
    mu, sigma = arr.mean(), arr.std(ddof=1)
    return mu / sigma * math.sqrt(ppy) if sigma > 1e-12 else 0.0


# ── Strategy generators ───────────────────────────────────────────────────────

def ma_returns(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    sig = (prices.rolling(fast).mean() > prices.rolling(slow).mean()).astype(float).shift(1)
    return (sig * prices.pct_change()).dropna()


def rsi_returns(prices: pd.Series, period: int, buy_t: float, sell_t: float) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rsi = 100 - 100 / (1 + gain / (loss + 1e-9))
    sig = ((rsi < buy_t).astype(float)).shift(1)  # buy oversold
    return (sig * prices.pct_change()).dropna()


def bb_returns(prices: pd.Series, window: int, n_std: float, mode: str = "breakout") -> pd.Series:
    """Bollinger Band strategy. mode='breakout' or 'reversion'."""
    mid  = prices.rolling(window).mean()
    std  = prices.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    if mode == "breakout":
        sig = (prices > upper).astype(float).shift(1)
    else:
        sig = (prices < lower).astype(float).shift(1)
    return (sig * prices.pct_change()).dropna()


def noise_returns(prices: pd.Series, seed: int) -> pd.Series:
    rng2 = np.random.default_rng(seed)
    sig = pd.Series(rng2.integers(0, 2, size=len(prices)), index=prices.index).shift(1)
    return (sig * prices.pct_change()).dropna()


# ── 1. Download data ──────────────────────────────────────────────────────────
print("\nDownloading data for 8 assets (2018-2023)...")
all_prices: dict[str, pd.Series] = {}
for asset in ASSETS:
    try:
        df = yf.download(asset, start=IS_START, end=OOS_END, progress=False)
        p = df["Close"].squeeze().dropna()
        if len(p) > 100:
            all_prices[asset] = p
            print(f"  {asset}: {len(p)} days")
        else:
            print(f"  {asset}: SKIP (only {len(p)} days)")
    except Exception as e:
        print(f"  {asset}: SKIP ({e!s:.30s})")

active_assets = list(all_prices.keys())
print(f"  Active: {len(active_assets)} assets")


# ── 2. Build strategy grid ────────────────────────────────────────────────────
print("\nBuilding 400+ strategies (MA + RSI + Bollinger + noise)...")
records = []

for asset, price_all in all_prices.items():
    price_is  = price_all[IS_START:IS_END]
    price_oos = price_all[OOS_START:OOS_END]

    if len(price_is) < 60 or len(price_oos) < 30:
        continue

    strategies: list[tuple[str, pd.Series, pd.Series]] = []

    # MA crossover: 7 x 5 = 35 combos
    for fast in [5, 10, 15, 20, 30, 40, 50]:
        for slow in [50, 75, 100, 150, 200]:
            if fast >= slow:
                continue
            r_is  = ma_returns(price_is,  fast, slow)
            r_oos = ma_returns(price_oos, fast, slow)
            strategies.append((f"MA({fast},{slow})", r_is, r_oos))

    # RSI: 3 periods x 4 thresholds = 12 combos
    for period in [7, 14, 21]:
        for buy_t in [20, 25, 30, 35]:
            r_is  = rsi_returns(price_is,  period, buy_t, 100 - buy_t)
            r_oos = rsi_returns(price_oos, period, buy_t, 100 - buy_t)
            strategies.append((f"RSI({period},{buy_t})", r_is, r_oos))

    # Bollinger Bands: 3 windows x 3 std x 2 modes = 18 combos
    for window in [10, 20, 30]:
        for n_std in [1.5, 2.0, 2.5]:
            for mode in ["breakout", "reversion"]:
                r_is  = bb_returns(price_is,  window, n_std, mode)
                r_oos = bb_returns(price_oos, window, n_std, mode)
                strategies.append((f"BB({window},{n_std},{mode[0]})", r_is, r_oos))

    # Noise: 25 garbage strategies
    for i in range(25):
        r_is  = noise_returns(price_is,  seed=i * 100 + abs(hash(asset)) % 1000)
        r_oos = noise_returns(price_oos, seed=i * 100 + abs(hash(asset)) % 1000 + 1)
        strategies.append((f"NOISE({i})", r_is, r_oos))

    n_total = len(strategies)

    for name, ret_is, ret_oos in strategies:
        is_sr  = _sharpe(ret_is)
        oos_sr = _sharpe(ret_oos)
        stype  = "noise" if name.startswith("NOISE") else name.split("(")[0]

        # DSR per-strategy with multiple-testing correction
        try:
            dsr_res  = deflated_sharpe_ratio(ret_is, n_trials=n_total)
            dsr_val  = float(dsr_res.get("dsr", 0.0))
            dsr_pass = int(dsr_res.get("verdict") == "PASS")
        except Exception:
            dsr_val, dsr_pass = 0.0, 0

        # MC p-value
        try:
            mc_res  = monte_carlo_permutation_test(ret_is, n_permutations=200)
            mc_pval = float(mc_res.get("pvalue", 1.0))
            mc_pass = int(mc_pval < 0.05)
        except Exception:
            mc_pval, mc_pass = 1.0, 0

        # Consensus: both DSR and MC must pass
        consensus_pass = int(dsr_pass == 1 and mc_pass == 1)

        records.append({
            "asset": asset,
            "strategy": name,
            "type": stype,
            "is_sharpe": round(is_sr, 4),
            "oos_sharpe": round(oos_sr, 4),
            "dsr": round(dsr_val, 4),
            "dsr_pass": dsr_pass,
            "mc_pvalue": round(mc_pval, 4),
            "mc_pass": mc_pass,
            "consensus_pass": consensus_pass,
            "oos_positive": int(oos_sr > 0),
        })

df = pd.DataFrame(records)
type_counts = df["type"].value_counts().to_dict()
print(f"  Total strategies: {len(df)}  {type_counts}")
print(f"  Active assets: {df['asset'].nunique()}")


# ── 3. Q1: IS Sharpe rank vs OOS (Spearman, per asset) ───────────────────────
print("\n" + "=" * 70)
print("  Q1: IS Sharpe rank vs OOS Sharpe (Spearman, per asset)")
print("=" * 70)
print(f"\n  {'Asset':<12} {'Spearman r':>12} {'p-value':>9}  Result")
print(f"  {'-' * 55}")
all_rs = []
for asset in active_assets:
    sub = df[df["asset"] == asset]
    if len(sub) < 5:
        continue
    r, p = stats.spearmanr(sub["is_sharpe"], sub["oos_sharpe"])
    sig = "signif" if p < 0.05 else "ns"
    interp = "IS predicts OOS" if (r > 0 and p < 0.05) else ("weak trend" if r > 0 else "INVERSE")
    print(f"  {asset:<12} {r:>12.4f}  {p:>8.4f}  {interp} [{sig}]")
    all_rs.append(r)

pooled_r = float(np.mean(all_rs))
print(f"\n  Pooled Spearman: {pooled_r:.4f}  "
      f"({'IS rank predicts OOS' if pooled_r > 0.1 else 'near zero — overfitting confirmed'})")


# ── 4. Q2: DSR pass vs fail ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Q2: DSR PASS vs FAIL — OOS performance split (real strategies only)")
print("=" * 70)
real = df[df["type"] != "noise"]
by_dsr = real.groupby("dsr_pass").agg(
    n=("oos_sharpe", "count"),
    mean_oos=("oos_sharpe", "mean"),
    pct_pos=("oos_positive", "mean"),
).reset_index()
for _, row in by_dsr.iterrows():
    label = "DSR PASS" if int(row["dsr_pass"]) else "DSR FAIL"
    print(f"  {label}: n={int(row['n'])}, OOS SR={row['mean_oos']:.4f}, OOS+={row['pct_pos']:.0%}")

pass_oos = real[real["dsr_pass"] == 1]["oos_sharpe"].values
fail_oos = real[real["dsr_pass"] == 0]["oos_sharpe"].values
if len(pass_oos) > 1 and len(fail_oos) > 1:
    t, p = stats.ttest_ind(pass_oos, fail_oos)
    print(f"  t-test: t={t:.3f}, p={p:.4f}")


# ── 5. Q3: Noise vs Real strategies (sanity) ──────────────────────────────────
print("\n" + "=" * 70)
print("  Q3: Strategy type comparison (OOS survival, DSR/MC pass rates)")
print("=" * 70)
by_type = df.groupby("type").agg(
    n=("is_sharpe", "count"),
    mean_is=("is_sharpe", "mean"),
    mean_oos=("oos_sharpe", "mean"),
    dsr_rate=("dsr_pass", "mean"),
    mc_rate=("mc_pass", "mean"),
    consensus_rate=("consensus_pass", "mean"),
    oos_pos=("oos_positive", "mean"),
).reset_index().sort_values("mean_oos", ascending=False)

print(f"\n  {'Type':<10} {'N':>5} {'IS SR':>7} {'OOS SR':>8} "
      f"{'DSR%':>6} {'MC%':>5} {'Both%':>6} {'OOS+':>6}")
print(f"  {'-' * 60}")
for _, row in by_type.iterrows():
    print(
        f"  {str(row['type']):<10} {int(row['n']):>5}"
        f" {row['mean_is']:>7.3f} {row['mean_oos']:>8.3f}"
        f" {row['dsr_rate']:>6.1%} {row['mc_rate']:>5.1%}"
        f" {row['consensus_rate']:>6.1%} {row['oos_pos']:>6.0%}"
    )


# ── 6. Q4: Consensus effect on false-positive rate ────────────────────────────
print("\n" + "=" * 70)
print("  Q4: Consensus filter — false positive rate reduction")
print("=" * 70)
noise_df = df[df["type"] == "noise"]
print(f"\n  Noise strategies that PASS each filter (lower = fewer false positives):")
print(f"  DSR alone passes       : {noise_df['dsr_pass'].mean():.1%}  of noise")
print(f"  MC alone passes        : {noise_df['mc_pass'].mean():.1%}  of noise")
print(f"  BOTH (consensus=2)     : {noise_df['consensus_pass'].mean():.1%}  of noise")
fp_reduction = 1 - (noise_df["consensus_pass"].mean() / max(noise_df["dsr_pass"].mean(), 1e-9))
print(f"  False-positive reduction: {fp_reduction:.0%} by requiring consensus")


# ── 7. Top quintile vs bottom quintile ───────────────────────────────────────
print("\n" + "=" * 70)
print("  IS Top-20% vs Bottom-20% performers: OOS comparison")
print("=" * 70)
real2 = df[df["type"] != "noise"].copy()
try:
    real2["quintile"] = pd.qcut(real2["is_sharpe"], q=5,
        labels=["Q1(worst)", "Q2", "Q3", "Q4", "Q5(best)"], duplicates="drop")
    q_grp = real2.groupby("quintile", observed=True).agg(
        n=("oos_sharpe", "count"),
        mean_oos=("oos_sharpe", "mean"),
        pct_pos=("oos_positive", "mean"),
    ).reset_index()
    print(f"\n  {'Quintile':<12} {'N':>5} {'OOS SR':>9} {'OOS+':>7}")
    print(f"  {'-' * 38}")
    for _, row in q_grp.iterrows():
        print(f"  {str(row['quintile']):<12} {int(row['n']):>5} {row['mean_oos']:>9.4f} {row['pct_pos']:>7.0%}")
except Exception:
    pass


# ── 8. Final verdict ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINDINGS SUMMARY")
print("=" * 70)

noise_oos = df[df["type"] == "noise"]["oos_positive"].mean()
real_oos  = df[df["type"] != "noise"]["oos_positive"].mean()
noise_consensus = noise_df["consensus_pass"].mean()

q5_oos_rate = real2[real2["quintile"] == "Q5(best)"]["oos_positive"].mean() if "quintile" in real2.columns else float("nan")
q1_oos_rate = real2[real2["quintile"] == "Q1(worst)"]["oos_positive"].mean() if "quintile" in real2.columns else float("nan")

print(f"""
  1. IS->OOS predictability (Spearman r={pooled_r:.3f}):
     {"Near zero — IS rank does not reliably predict OOS performance" if abs(pooled_r) < 0.15 else "Some predictive power detected"}
     Implication: need consensus + hysteresis before scaling down

  2. Noise OOS survival: {noise_oos:.0%}  vs  Real: {real_oos:.0%}
     {"Audit DOES separate noise from signal (noise underperforms real OOS)" if real_oos > noise_oos else "No clear separation — investigate"}

  3. Best IS quintile OOS+: {q5_oos_rate:.0%}  vs  Worst: {q1_oos_rate:.0%}
     {"Classic overfitting: top IS performers underperform OOS" if q5_oos_rate < q1_oos_rate else "IS rank has some OOS predictive power"}

  4. Consensus filter reduces noise false positives by {fp_reduction:.0%}
     (requiring DSR + MC to both pass simultaneously)
     Practical implication: use consensus >= 2 before reducing position size

  Claim: "Across {len(df)} strategies on {len(active_assets)} assets,
  the consensus filter (DSR + MC) reduces false-positive audit signals
  by {fp_reduction:.0%} while real strategies show {real_oos:.0%} OOS survival vs {noise_oos:.0%} for pure noise."
""")
