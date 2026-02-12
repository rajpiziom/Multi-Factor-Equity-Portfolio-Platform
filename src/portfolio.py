from __future__ import annotations

import numpy as np
import pandas as pd


def _vol_scaled_weights(candidates: pd.DataFrame) -> pd.Series:
    inv = 1.0 / candidates["realized_vol"].replace(0, np.nan)
    inv = inv.fillna(inv.median())
    w = inv / inv.sum()
    return w


def make_target_weights(
    signals: pd.DataFrame,
    returns_panel: pd.DataFrame,
    factor_col: str,
    quantile: float = 0.2,
    long_short: bool = False,
    weighting: str = "equal",
    vol_window: int = 60,
) -> pd.DataFrame:
    rets = returns_panel[["date", "ticker", "ret"]].copy()
    rets = rets.sort_values(["ticker", "date"])
    rets["realized_vol"] = rets.groupby("ticker")["ret"].rolling(vol_window).std().reset_index(level=0, drop=True)

    sig = signals[["date", "ticker", factor_col]].dropna().copy()
    sig = sig.merge(rets[["date", "ticker", "realized_vol"]], on=["date", "ticker"], how="left")

    all_weights = []
    for dt, frame in sig.groupby("date"):
        frame = frame.dropna(subset=[factor_col]).sort_values(factor_col, ascending=False)
        if frame.empty:
            continue

        n = max(1, int(len(frame) * quantile))
        long_bucket = frame.head(n).copy()

        if weighting == "vol_scaled":
            long_bucket["weight"] = _vol_scaled_weights(long_bucket)
        else:
            long_bucket["weight"] = 1.0 / len(long_bucket)

        long_bucket["side"] = 1
        selected = [long_bucket]

        if long_short:
            short_bucket = frame.tail(n).copy()
            if weighting == "vol_scaled":
                short_bucket["weight"] = -_vol_scaled_weights(short_bucket)
            else:
                short_bucket["weight"] = -1.0 / len(short_bucket)
            short_bucket["side"] = -1
            selected.append(short_bucket)

        combined = pd.concat(selected, ignore_index=True)
        gross = combined["weight"].abs().sum()
        if gross > 0:
            combined["weight"] = combined["weight"] / gross
        all_weights.append(combined[["date", "ticker", "weight"]])

    return pd.concat(all_weights, ignore_index=True)


def make_target_weights_with_universe(
    signals: pd.DataFrame,
    returns_panel: pd.DataFrame,
    factor_col: str,
    universe_by_date: dict[pd.Timestamp, list[str]],
    quantile: float = 0.2,
    long_short: bool = False,
    weighting: str = "equal",
    vol_window: int = 60,
) -> pd.DataFrame:
    rets = returns_panel[["date", "ticker", "ret"]].copy().sort_values(["ticker", "date"])
    rets["realized_vol"] = rets.groupby("ticker")["ret"].rolling(vol_window).std().reset_index(level=0, drop=True)
    sig = signals[["date", "ticker", factor_col]].dropna().copy()
    sig = sig.merge(rets[["date", "ticker", "realized_vol"]], on=["date", "ticker"], how="left")

    out: list[pd.DataFrame] = []
    for dt, frame in sig.groupby("date"):
        allowed = set(universe_by_date.get(pd.Timestamp(dt), []))
        frame = frame[frame["ticker"].isin(allowed)]
        frame = frame.dropna(subset=[factor_col]).sort_values(factor_col, ascending=False)
        if frame.empty:
            continue

        n = max(1, int(len(frame) * quantile))
        long_bucket = frame.head(n).copy()
        long_bucket["weight"] = _vol_scaled_weights(long_bucket) if weighting == "vol_scaled" else (1.0 / len(long_bucket))
        selected = [long_bucket]

        if long_short:
            short_bucket = frame.tail(n).copy()
            short_bucket["weight"] = -_vol_scaled_weights(short_bucket) if weighting == "vol_scaled" else (-1.0 / len(short_bucket))
            selected.append(short_bucket)

        combined = pd.concat(selected, ignore_index=True)
        gross = combined["weight"].abs().sum()
        if gross > 0:
            combined["weight"] = combined["weight"] / gross
        out.append(combined[["date", "ticker", "weight"]])

    if not out:
        return pd.DataFrame(columns=["date", "ticker", "weight"])
    return pd.concat(out, ignore_index=True)
