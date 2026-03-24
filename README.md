# backtest-audit

[![CI](https://github.com/Aliipou/backtest-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/Aliipou/backtest-audit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-124%20passing-brightgreen)](https://github.com/Aliipou/backtest-audit/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Catch overfitting before it costs you money.**

---

## Evidence it works

Validation experiment: 264 strategies (MA crossover + RSI + pure noise) across SPY, QQQ, GLD, BTC-USD.
IS period: 2018-2021. OOS period: 2022-2023.

| Finding | Result | What it means |
|---------|--------|---------------|
| IS Sharpe rank vs OOS | Spearman r = -0.02 | IS winners do NOT predict OOS winners |
| Best IS quintile OOS+ | 78% | Top IS performers have lowest OOS survival |
| Worst IS quintile OOS+ | 86% | Worst IS performers survive OOS more often |
| Noise OOS survival | 71% vs 86% (real) | Audit separates garbage from signal |

**Key claim:** Across 264 strategies on 4 assets, the best in-sample strategies had lower OOS survival than the worst — confirming that backtest selection bias is real and measurable.

```bash
python examples/validation_experiment.py   # reproduce in ~2 minutes
```

---

## What a strategy with Sharpe 1.2 actually looks like

```
Deflated Sharpe Ratio     DSR=-75.6   [FAIL]   <- selected from 19 combos
Probability of Overfitting  PBO=1.00  [FAIL]   <- 100% chance this is luck
Economic Significance       d=0.051   [WARN]   <- negligible effect size
Regime: trend_down          SR=-3.48  [FAIL]   <- collapses in bear markets
Robustness: 3/7 survived    FRAGILE   [WARN]   <- edge breaks under stress

OVERALL VERDICT: FAIL
```

```bash
python examples/audit_demo.py   # real SPY data, runs in 30 seconds
```

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
```

### Full output

```python
report.economic_result      # Cohen's d, MDE, R^2, break-even cost
report.walk_forward_result  # OOS hit rate, IS/OOS Sharpe correlation
report.regime_result        # Per-regime DSR+MC (low/high vol, trend/counter)
report.robustness_report    # 7-scenario stress test survival rate
report.to_dict()            # JSON-serialisable — pipe to any dashboard
```

---

## REST API

```bash
pip install "backtest-audit[api]"
uvicorn backtest_audit.api:app --reload
# -> http://localhost:8000/docs
```

| Endpoint | What |
|----------|------|
| `POST /audit` | Full 8-test audit |
| `POST /audit/dsr` | DSR only |
| `POST /audit/mc` | Monte Carlo only |
| `POST /audit/pbo` | PBO (returns matrix) |
| `POST /audit/sensitivity` | Parameter sensitivity |
| `POST /audit/economic` | Effect size, MDE, R^2 |
| `POST /audit/walk-forward` | OOS validation |
| `POST /audit/regime` | Regime-conditional audit |
| `POST /audit/robustness` | Stress battery |

---

## Docker

```bash
docker build --target production -t backtest-audit .
docker run -p 8000:8000 backtest-audit
# -> http://localhost:8000/health
```

---

## Development

```bash
pip install -e ".[dev,api]"
pytest tests/ -v          # 124 tests, ~5s, zero network calls
ruff check src/ tests/    # lint clean
```

---

## References

- Bailey, D. & Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal of Portfolio Management.
- White, H. (2000). *A Reality Check for Data Snooping.* Econometrica, 68(5).
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences.*
