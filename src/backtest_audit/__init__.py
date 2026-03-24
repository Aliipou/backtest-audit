"""
backtest-audit
==============
Statistical overfitting audit tools for algorithmic trading backtests.

Public API
----------
BacktestAuditor   – orchestrates all tests given a return series.
AuditReport       – dataclass returned by BacktestAuditor.run_all().

Individual test functions are also importable directly:

    from backtest_audit.deflated_sharpe import deflated_sharpe_ratio
    from backtest_audit.pbo             import probability_of_backtest_overfitting
    from backtest_audit.monte_carlo     import monte_carlo_permutation_test
    from backtest_audit.sensitivity     import parameter_sensitivity
"""

from .auditor import AuditReport, BacktestAuditor

__all__ = ["BacktestAuditor", "AuditReport"]
__version__ = "0.1.0"
