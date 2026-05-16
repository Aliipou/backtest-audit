# backtest-audit

The Sharpe Ratio of a backtest is not a reliable measure of genuine edge when you tested more than one strategy. If you evaluated 50 parameter combinations and reported the best-performing one, the expected maximum Sharpe Ratio from 50 trials on pure noise is not zero — it is substantially positive. A Sharpe of 1.4 from a single unconstrained backtest looks convincing; the same Sharpe from the best of 50 trials is unremarkable. Standard backtesting frameworks (vectorbt, backtrader, zipline) give you Sharpe Ratios. None of them tell you whether your Sharpe Ratio survived correction for the number of strategies you looked at before choosing this one.

backtest-audit implements the statistical tests from Bailey & Lopez de Prado (2014) and related literature that quantify this directly: Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO), Monte Carlo permutation test, walk-forward out-of-sample validation, regime-conditional audit, and robustness stress testing. The result is a single `PASS / WARN / FAIL` verdict per strategy and a continuous risk score suitable for position sizing.

---

## Architecture

```
BacktestAuditor
    |
    |-- run_dsr()              -- deflated_sharpe.py
    |       Corrects observed SR for n_trials, return skewness/kurtosis,
    |       and serial autocorrelation (Newey-West lag-1 correction).
    |       Returns: dsr_zscore, pvalue, obs_sharpe, benchmark_sharpe, verdict
    |
    |-- run_monte_carlo()      -- monte_carlo.py
    |       Shuffles the return series N times; observed SR percentile
    |       rank in the null distribution (White 2000 Reality Check).
    |       Returns: sharpe, percentile, pvalue, verdict
    |
    |-- run_walk_forward()     -- walk_forward.py
    |       Splits history into n_splits IS/OOS window pairs.
    |       Reports OOS hit rate (fraction of windows with SR > 0),
    |       IS/OOS Sharpe correlation, and mean OOS Sharpe.
    |
    |-- run_regime_audit()     -- regime.py
    |       Classifies returns into LOW_VOL / NORMAL / HIGH_VOL regimes
    |       using EWMA rolling volatility, then runs DSR + MC per regime.
    |       Reports consistency score and per-regime verdicts.
    |
    |-- run_robustness()       -- robustness.py
    |       7 stress scenarios: subsampling, noise injection,
    |       tail amplification, drawdown extension, regime shift,
    |       transaction cost friction, zero-edge shuffle (sanity check).
    |       Reports SR retention per scenario (>= 50% of baseline = survive).
    |
    |-- run_economic_significance()   -- economic_significance.py
    |       Cohen's d effect size, minimum detectable effect (MDE),
    |       break-even transaction cost in bps, R-squared signal variance.
    |
    |-- [optional] pbo.probability_of_backtest_overfitting()   -- pbo.py
    |       Combinatorial Purged Cross-Validation (CPCV):
    |       all C(S, S/2) train/test splits of S time blocks.
    |       Fraction of splits where best-IS strategy != best-OOS strategy.
    |
    |-- [optional] sensitivity.parameter_sensitivity()         -- sensitivity.py
    |       Measures how much SR changes as parameters vary around the optimum.
    |       High sensitivity = the optimum is a narrow peak, not a plateau.
    v
AuditReport
    overall_verdict: "PASS" | "WARN" | "FAIL"
    overall_risk_score(): float in [0, 1]  -- for position sizing
    print_report(): formatted table to stdout
    to_dict(): JSON-serializable output

FastAPI REST API  (api.py)
    POST /audit                 -- full audit
    POST /audit/dsr             -- DSR only
    POST /audit/mc              -- Monte Carlo only
    POST /audit/pbo             -- PBO (requires returns matrix)
    POST /audit/walk-forward
    POST /audit/regime
    POST /audit/robustness
    POST /audit/economic
```

---

## What DSR is and why plain Sharpe Ratio is insufficient

The Sharpe Ratio (SR) measures a strategy's mean return divided by its standard deviation. It has two problems when used to evaluate backtested strategies.

**Problem 1: multiple testing.** If you test N independent strategies, the expected maximum SR from N tests — even if all strategies have zero true edge — grows roughly as `sqrt(2) * erfinv(1 - 1/N)`. For N=50, this is approximately 0.32 per period; for N=200 it is 0.47. An observed SR below the expected maximum from N trials on pure noise is not evidence of edge at all. The DSR corrects for this by computing the z-score of your observed SR against the benchmark SR for N trials:

```
DSR_zscore = (SR_obs - SR*) / SE(SR_obs)
```

where `SR*` is the expected maximum SR for N trials (Bailey & Lopez de Prado 2014, eq. 8) and `SE(SR_obs)` is the standard error of the Sharpe Ratio adjusted for non-normality (Mertens 2002) and serial autocorrelation (Newey-West lag-1 correction). A positive z-score means your strategy outperforms the benchmark — the corrected DSR is `Phi(DSR_zscore)`, a probability in [0, 1].

**Problem 2: non-IID returns.** Standard SR assumes IID normal returns. Real strategies have skewed, fat-tailed, and autocorrelated return distributions. The SE correction term includes the third and fourth standardized moments (skewness, excess kurtosis) and a variance ratio for lag-1 autocorrelation.

---

## What PBO tells you that other tests do not

DSR corrects for multiple testing under the assumption that your N strategies are independent. PBO (Lopez de Prado & Bailey 2014) tests something different: given the actual return time series from all N strategy variants, how often does the best-performing strategy in-sample also win out-of-sample?

The method is Combinatorial Purged Cross-Validation (CPCV): divide the return history into S time blocks, enumerate all `C(S, S/2)` ways to split them into train and test halves, and for each split ask whether the IS winner is also the OOS winner. PBO is the fraction of splits where it is not. A PBO near 0.0 means the IS winner reliably wins OOS. A PBO above 0.5 means that in more than half of possible train/test splits, selecting on IS Sharpe picks the wrong strategy — the selection process itself is overfitting to the train set.

Unlike DSR (which requires only the final selected strategy's returns and a count of trials), PBO requires the full returns matrix for all strategy variants. It is more expensive to compute but makes no assumptions about strategy independence.

---

## Key Design Decisions

**Separate modules per test, orchestrated by `BacktestAuditor`, not a monolithic function.** Each statistical test (`deflated_sharpe.py`, `pbo.py`, `monte_carlo.py`, etc.) is a pure function with no state. `BacktestAuditor` orchestrates them and collects results into `AuditReport`. This allows using any test independently without the full orchestrator, and makes the test suite straightforward — each module has its own unit tests.

**DSR reported as a z-score, not as the raw CDF value.** The original Bailey & Lopez de Prado paper defines DSR as `Phi((SR - SR*) / SE)`, a probability in [0, 1]. This implementation computes both but reports the z-score as `dsr` in the output dict, because a z-score is more interpretable: negative means below the multiple-testing benchmark, positive means above it. The p-value (`1 - Phi(z)`) is also returned.

**PBO n_splits capped automatically to keep combination count below 2000.** `C(16, 8) = 12870`, which is fast. `C(20, 10) = 184756`, which is slow. The implementation starts at the requested `n_splits` and decrements by 2 until `C(s, s//2) <= 2000`. This prevents accidental multi-minute computation for large inputs.

**Walk-forward uses non-overlapping windows, not expanding windows.** An expanding window IS period grows with each fold, which inflates IS Sharpe for later folds (more data, lower variance estimate) and makes IS/OOS comparison inconsistent across folds. Fixed-size windows are directly comparable. Tradeoff: each window uses only `n / (n_splits + 1)` observations, so the test requires more data.

**Robustness stress tests include a zero-edge sanity check (shuffle).** Scenario 7 shuffles the return series, which should always fail the SR retention test (shuffled returns have the same distribution but no timing skill). If the strategy's SR retention on shuffled returns is above 50%, it means the edge is entirely in return magnitude rather than timing — itself a red flag worth knowing.

**Optional REST API separated into `api.py` with FastAPI as an optional dependency.** The core statistical library (`deflated_sharpe`, `pbo`, `monte_carlo`, etc.) depends only on numpy, pandas, and scipy. FastAPI is in `[project.optional-dependencies]` under `api`. Users who only want the Python library do not install the web server.

---

## Tech Stack

| Component | Justification |
|---|---|
| **numpy** | Core numerical operations; permutation shuffles, matrix slicing for PBO, EWMA volatility for regime detection |
| **pandas** | Return series handling; `autocorr()` for Newey-West correction, `dropna()` for safe series alignment |
| **scipy** | `norm.cdf` for DSR z-score to probability, `erfinv` for benchmark SR computation (Bailey & Lopez de Prado eq. 8) |
| **FastAPI** | Optional REST API layer; each test exposed as a typed POST endpoint with Pydantic request/response models |
| **Pydantic v2** | Input validation for the API (return series length checks, n_trials >= 1, n_splits must be even) |
| **pytest** | 124 tests covering all statistical modules, edge cases (zero-std series, n_trials=1, insufficient data), and the REST API |
| **hatchling** | Build backend; `pip install -e .` installs the `backtest_audit` package from `src/` |

---

## Running Locally

```bash
git clone https://github.com/Aliipou/backtest-audit.git
cd backtest-audit

# Core library only (numpy, pandas, scipy)
pip install -e .

# With REST API
pip install -e ".[api]"

# With development tools
pip install -e ".[dev]"

# Run tests
pytest -v

# Run test coverage
pytest --cov=src/backtest_audit --cov-report=term-missing
```

Basic Python usage:

```python
import pandas as pd
from backtest_audit import BacktestAuditor

# Your strategy's daily return series
returns = pd.Series([...])

# n_trials: how many parameter combinations you tested before choosing this one
auditor = BacktestAuditor(returns, n_trials=50)
report = auditor.run_all()
report.print_report()

print(report.overall_verdict)        # "PASS", "WARN", or "FAIL"
print(report.overall_risk_score())   # 0.0 (no overfitting) to 1.0 (strong overfitting)
```

PBO requires the full returns matrix across all strategy variants:

```python
from backtest_audit.pbo import probability_of_backtest_overfitting

# returns_matrix: rows = time periods, columns = strategy variants
result = probability_of_backtest_overfitting(returns_matrix, n_splits=16)
print(result["pbo"])       # fraction of CV splits where IS winner != OOS winner
print(result["verdict"])   # "PASS" if pbo < 0.3, "WARN" if < 0.5, "FAIL" otherwise
```

Start the REST API:

```bash
uvicorn backtest_audit.api:app --port 8080

curl -X POST http://localhost:8080/audit/dsr \
  -H "Content-Type: application/json" \
  -d '{"returns": [0.001, -0.002, 0.003, ...], "n_trials": 50}'
```

See `examples/audit_demo.py` for a full end-to-end demo using real SPY data with a moving-average crossover grid (requires `yfinance`).

---

## Deployment

The library is stateless — no database, no broker. Deploy the REST API as a single container:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[api]"
CMD ["uvicorn", "backtest_audit.api:app", "--host", "0.0.0.0", "--port", "8080"]
```

The API includes a built-in sliding-window rate limiter (60 req/min per IP) implemented in memory. For multi-instance deployments this should be moved to Redis. See `Dockerfile` and `docker-compose.yml` in the repository root.

---

## Known Limitations / TODO

- **DSR assumes strategies are independent.** The multiple-testing correction (`SR*` benchmark) uses the formula for independent, IID-normal strategies. Correlated strategies (e.g., MA crossover with slightly different windows) share return variance, which means the effective N is lower than the trial count. PBO does not make this assumption.
- **Regime classification uses only volatility and trend direction.** The `LOW_VOL / NORMAL / HIGH_VOL` classification is based on 20-period EWMA volatility normalized to the full-sample standard deviation. It does not detect structural breaks, correlation regime changes, or liquidity regime shifts.
- **Walk-forward uses fixed equal-size windows.** Non-stationarity means earlier windows may be from a different market regime than later ones. Purged cross-validation (where test sets are time-separated from train sets by a gap) is not implemented — there is no embargo period between IS and OOS windows.
- **Robustness scenarios use hardcoded percentages.** The tail amplification multiplier (2x), subsample fraction (70%), and noise scale (0.5 sigma) are constants. These should be configurable.
- **No persistence of audit results.** The REST API returns JSON; results are not stored. Add a database layer if you need historical audit trails across strategy versions.
- **`n_splits` auto-cap for PBO is heuristic.** The 2000 combination cap was chosen to keep computation under ~5 seconds on a laptop CPU. For large returns matrices with many strategy columns, even 2000 combinations can be slow.
