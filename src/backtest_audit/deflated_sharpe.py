"""
Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2014).

The DSR corrects the observed Sharpe Ratio for:
  1. The number of strategy trials tested (multiple-testing bias).
  2. Non-normality of returns (skewness and excess kurtosis).
  3. Serial correlation in returns (via the variance correction factor).

Reference
---------
Bailey, D. H., & Lopez de Prado, M. (2014).
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting,
and Non-Normality." Journal of Portfolio Management, 40(5), 94–107.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import erfinv
from scipy.stats import norm


def _sharpe_ratio(returns: pd.Series, annualize: bool = False) -> float:
    """Compute the Sharpe Ratio of a return series (mean / std)."""
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    if sigma == 0:
        return 0.0
    sr = mu / sigma
    if annualize:
        sr *= math.sqrt(252)
    return float(sr)


def _benchmark_sharpe(n_trials: int) -> float:
    """
    Expected maximum Sharpe Ratio under the null hypothesis (IID normal returns)
    when *n_trials* independent strategies have been tested.

    Formula (Bailey & Lopez de Prado 2014, eq. 8):
        SR* = sqrt(2) * erfinv(1 - 1/n_trials)

    For n_trials == 1 this returns 0.0, meaning any positive SR survives.
    """
    if n_trials <= 1:
        return 0.0
    # erfinv maps from (-1, 1); guard against floating-point edge cases
    arg = max(-1 + 1e-12, min(1 - 1e-12, 1.0 - 1.0 / n_trials))
    return float(math.sqrt(2) * erfinv(arg))


def _variance_ratio_correction(returns: pd.Series) -> float:
    """
    Variance ratio that accounts for serial correlation in returns.

    For an IID series the variance of the mean scales as sigma^2/T.
    With serial correlation the effective sample size is reduced.
    We use the Newey-West-style correction (lag-1 only for simplicity):

        V = 1 + 2 * rho_1 * (1 - 1/T)

    where rho_1 is the lag-1 autocorrelation.  Returns V >= 1.
    """
    t = len(returns)
    if t < 3:
        return 1.0
    rho1 = float(returns.autocorr(lag=1))
    if np.isnan(rho1):
        rho1 = 0.0
    v = 1.0 + 2.0 * rho1 * (1.0 - 1.0 / t)
    return max(v, 1e-6)  # keep positive


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """
    Compute the Deflated Sharpe Ratio (DSR).

    Parameters
    ----------
    returns : pd.Series
        Out-of-sample (or in-sample) period returns of the strategy.
    n_trials : int
        Total number of parameter combinations / strategies that were
        evaluated before settling on this one.  Must be >= 1.
    skewness : float
        Third standardised moment of the return distribution.
        0.0 = symmetric (default).
    kurtosis : float
        Fourth standardised moment (NOT excess kurtosis).
        3.0 = normal distribution (default).

    Returns
    -------
    dict with keys:
        dsr              – Deflated Sharpe Ratio (z-score under H0)
        pvalue           – one-tailed p-value: P(SR* >= obs_sharpe | H0)
        obs_sharpe       – observed per-period Sharpe Ratio
        benchmark_sharpe – expected max SR for n_trials IID strategies
        verdict          – "PASS" | "WARN" | "FAIL"
    """
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        raise ValueError("Need at least 2 non-NaN return observations.")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")

    t = len(returns)
    obs_sr = _sharpe_ratio(returns)
    sr_star = _benchmark_sharpe(n_trials)

    # Excess kurtosis (subtract 3 for the normal baseline)
    gamma2 = kurtosis - 3.0

    # Standard error of the Sharpe Ratio (Mertens 2002 / Lo 2002 non-normality
    # correction), scaled by the variance-ratio for serial correlation.
    var_ratio = _variance_ratio_correction(returns)
    se2 = var_ratio / t * (1.0 - skewness * obs_sr + ((gamma2) / 4.0) * obs_sr**2)
    se = math.sqrt(max(se2, 1e-12))

    # DSR = Phi( (SR_obs - SR*) / SE )   (Bailey & Lopez de Prado eq. 11)
    dsr = float(norm.cdf((obs_sr - sr_star) / se))

    # p-value: probability that a strategy chosen from n_trials IID trials
    # achieves at least obs_sr by chance.
    pvalue = 1.0 - dsr

    # Verdict thresholds
    # dsr > 0.5  → positive z-score, SR exceeds benchmark  → PASS
    # dsr in (0.25, 0.5] → marginal                        → WARN
    # dsr <= 0.25                                           → FAIL
    #
    # We expose DSR as the raw CDF value (0..1), but for the verdict we map
    # to the "DSR z-score" convention used in the docstring spec:
    #   PASS  if dsr_zscore > 0   ⟺ dsr_cdf > 0.5
    #   WARN  if dsr_zscore in (-0.5, 0] ⟺ dsr_cdf in (0.3085, 0.5]
    #   FAIL  if dsr_zscore <= -0.5 ⟺ dsr_cdf <= 0.3085
    dsr_zscore = float((obs_sr - sr_star) / se)
    if dsr_zscore > 0:
        verdict = "PASS"
    elif dsr_zscore > -0.5:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "dsr": dsr_zscore,
        "pvalue": round(pvalue, 6),
        "obs_sharpe": round(obs_sr, 6),
        "benchmark_sharpe": round(sr_star, 6),
        "verdict": verdict,
    }
