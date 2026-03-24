"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def good_returns(rng: np.random.Generator) -> pd.Series:
    """252 daily returns with genuine positive edge (~SR 1.5)."""
    return pd.Series(rng.normal(loc=0.001, scale=0.01, size=252))


@pytest.fixture
def bad_returns(rng: np.random.Generator) -> pd.Series:
    """252 daily returns with zero edge."""
    return pd.Series(rng.normal(loc=0.0, scale=0.01, size=252))


@pytest.fixture
def short_returns() -> pd.Series:
    return pd.Series([0.01, -0.005, 0.003])


@pytest.fixture
def returns_matrix(rng: np.random.Generator) -> pd.DataFrame:
    """5 strategies × 200 periods for PBO tests."""
    return pd.DataFrame(
        {f"s{i}": rng.normal(loc=0.0005 * i, scale=0.01, size=200) for i in range(5)}
    )
