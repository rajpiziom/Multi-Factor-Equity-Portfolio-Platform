from __future__ import annotations

import pandas as pd

from .risk import factor_regression


def run_attribution(strategy_returns: pd.Series, ff_daily: pd.DataFrame) -> pd.DataFrame:
    capm = factor_regression(strategy_returns, ff_daily, factors=["Mkt-RF"])
    ff3 = factor_regression(strategy_returns, ff_daily, factors=["Mkt-RF", "SMB", "HML"])

    table = pd.DataFrame([capm, ff3], index=["CAPM", "FF3"])
    return table
