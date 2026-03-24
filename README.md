# backtest-audit

[![CI](https://github.com/Aliipou/backtest-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/Aliipou/backtest-audit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-124%20passing-brightgreen)](https://github.com/Aliipou/backtest-audit/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Catch overfitting before it costs you money.**

A strategy with Sharpe 1.2 on SPY data. Looks great — until you run the audit:

```
Deflated Sharpe Ratio     DSR=-75.6   [FAIL]   <- selected from 19 combos
Probability of Overfitting  PBO=1.00  [FAIL]   <- 100% chance this is luck
Economic Significance       d=0.051   [WARN]   <- negligible effect size
Regime: trend_down          SR=-3.48  [FAIL]   <- collapses in bear markets
```

**VERDICT: FAIL** — the edge isn't real. This is what backtest-audit catches.

---

## Live Demo

```bash
git clone https://github.com/Aliipou/backtest-audit
cd backtest-audit
pip install -e ".[dev]" && pip install yfinance
python examples/audit_demo.py
```

Downloads real SPY data, backtests a moving-average strategy grid (19 combos),
runs the full 8-test audit, and shows you exactly why the "best" strategy is overfit.

---

## What it audits

| Module | Method | What it catches |
|--------|--------|-----------------|
| **Deflated Sharpe Ratio** | Bailey & Lopez de Prado (2014) | Multiple-testing inflation |
| **Monte Carlo Permutation** | White (2000) | Returns order not mattering |
| **PBO** | Bailey & Lopez de Prado (2014) | IS winners losing OOS |
| **Parameter Sensitivity** | — | Narrow, brittle parameter windows |
| **Economic Significance** | Cohen (1988) | Statistically significant but economically useless |
| **Walk-Forward OOS** | — | In-sample edge not holding out-of-sample |
| **Regime Audit** | — | Edge disappearing in high-vol or bear regimes |
| **Robustness Stress Test** | — | Edge collapsing under noise / cost / tail events |

---

## Quickstart

```bash
pip install backtest-audit
```

```python
import pandas as pd
from backtest_audit import BacktestAuditor

returns = pd.read_csv("my_strategy_returns.csv").squeeze()
auditor = BacktestAuditor(returns, n_trials=50)  # 50 param combos tried
report  = auditor.run_all()
report.print_report()

print(report.overall_verdict)  # "PASS" | "WARN" | "FAIL"
print(report.summary())        # one-line summary of all tests
```

### What you get back

```python
report.dsr_result           # Deflated Sharpe (dict)
report.monte_carlo_result   # MC permutation test (dict)
report.pbo_result           # Probability of overfitting (dict)
report.economic_result      # Cohen's d, MDE, R^2, break-even cost
report.walk_forward_result  # OOS hit rate, IS/OOS correlation
report.regime_result        # Per-regime DSR+MC (low/high vol, trend)
report.robustness_report    # 7-scenario stress test survival
report.to_dict()            # Full JSON-serialisable report
```

---

## REST API

```bash
pip install "backtest-audit[api]"
uvicorn backtest_audit.api:app --reload
# -> http://localhost:8000/docs
```

9 endpoints:

| Method | Endpoint | What |
|--------|----------|------|
| POST | `/audit` | Full audit — all 8 tests |
| POST | `/audit/dsr` | DSR only |
| POST | `/audit/mc` | Monte Carlo only |
| POST | `/audit/pbo` | PBO (requires returns matrix) |
| POST | `/audit/sensitivity` | Parameter sensitivity |
| POST | `/audit/economic` | Economic significance |
| POST | `/audit/walk-forward` | Walk-forward OOS |
| POST | `/audit/regime` | Regime-conditional audit |
| POST | `/audit/robustness` | Stress test battery |
| GET | `/health` | Liveness |
| GET | `/metrics` | Request counters |

---

## Docker

```bash
docker build --target production -t backtest-audit .
docker run -p 8000:8000 backtest-audit
```

---

## Development

```bash
pip install -e ".[dev,api]"
pytest tests/ -v          # 124 tests, ~5s, zero network calls
ruff check src/ tests/    # lint
```

---

## References

- Bailey, D. & Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management.
- White, H. (2000). *A Reality Check for Data Snooping.* Econometrica, 68(5).
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences.*
