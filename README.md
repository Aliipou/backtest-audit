# backtest-audit

[![CI](https://github.com/Aliipou/backtest-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/Aliipou/backtest-audit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Statistical overfitting audit tools for algorithmic trading backtests.

Implements peer-reviewed methods to detect whether your backtest results are genuine or the product of overfitting / data-snooping bias.

## Tests included

| Test | Reference |
|------|-----------|
| **Deflated Sharpe Ratio (DSR)** | Bailey & Lopez de Prado (2014) |
| **Monte Carlo Permutation Test** | White (2000) |
| **Probability of Backtest Overfitting (PBO)** | Bailey & Lopez de Prado (2014) |
| **Parameter Sensitivity Analysis** | — |

## Quick start

```bash
pip install backtest-audit
```

```python
import pandas as pd
from backtest_audit import BacktestAuditor

returns = pd.read_csv("returns.csv", squeeze=True)
auditor = BacktestAuditor(returns, n_trials=50)  # tested 50 param combos
report  = auditor.run_all()
report.print_report()
print(report.overall_verdict)  # "PASS" | "WARN" | "FAIL"
```

## REST API

```bash
pip install "backtest-audit[api]"
uvicorn backtest_audit.api:app --reload
# → http://localhost:8000/docs
```

## Docker

```bash
docker compose up api
```

## Development

```bash
pip install -e ".[dev,api]"
pytest tests/ -v
```
