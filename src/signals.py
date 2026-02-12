from __future__ import annotations

import numpy as np
import pandas as pd


def compute_month_end_signals(
    returns_panel: pd.DataFrame,
    momentum_lookback_months: int = 12,
    momentum_skip_days: int = 21,
    lowvol_window_days: int = 60,
) -> pd.DataFrame:
    df = returns_panel.copy().sort_values(["ticker", "date"])

    lookback_days = momentum_lookback_months * 21

    grouped = df.groupby("ticker", group_keys=False)
    df["cum_lookback"] = grouped["ret"].rolling(lookback_days).apply(lambda x: np.prod(1 + x) - 1, raw=True).reset_index(level=0, drop=True)
    df["cum_skip"] = grouped["ret"].rolling(momentum_skip_days).apply(lambda x: np.prod(1 + x) - 1, raw=True).reset_index(level=0, drop=True)
    df["momentum"] = ((1 + df["cum_lookback"]) / (1 + df["cum_skip"])) - 1

    df["lowvol"] = -grouped["ret"].rolling(lowvol_window_days).std().reset_index(level=0, drop=True)

    df["month"] = df["date"].dt.to_period("M")
    month_end = df.groupby(["ticker", "month"], as_index=False).tail(1)

    signals = month_end[["date", "ticker", "momentum", "lowvol"]].dropna()
    return signals
