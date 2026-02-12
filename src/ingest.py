from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .utils import ensure_parent


def fetch_stooq_history(stooq_ticker: str, start: str, end: str, cache_dir: str = "data/cache") -> pd.DataFrame:
    cache_path = Path(cache_dir) / f"{stooq_ticker}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    url = f"https://stooq.com/q/d/l/?s={stooq_ticker.lower()}&i=d"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]
    if df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume", "Open": "open", "High": "high", "Low": "low"})
    df = df.set_index("date").sort_index()
    df.columns = [c.lower() for c in df.columns]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def build_price_panel(
    universe: pd.DataFrame,
    start: str,
    end: str,
    output_path: str,
    min_obs: int = 252,
) -> pd.DataFrame:
    frames = []
    for row in universe.itertuples(index=False):
        try:
            raw = fetch_stooq_history(row.stooq_ticker, start, end)
        except Exception:
            continue

        if len(raw) < min_obs:
            continue

        use_cols = [c for c in ["close", "volume"] if c in raw.columns]
        if "close" not in use_cols:
            continue

        sub = raw[use_cols].reset_index().copy()
        sub["ticker"] = row.ticker
        sub = sub.rename(columns={"close": "adj_close"})
        frames.append(sub)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"])
    panel["date"] = pd.to_datetime(panel["date"])

    ensure_parent(output_path)
    panel.to_parquet(output_path, index=False)
    return panel


def compute_return_panel(prices: pd.DataFrame, output_path: str | None = None) -> pd.DataFrame:
    returns = prices.copy()
    returns["ret"] = returns.groupby("ticker")["adj_close"].pct_change()
    returns = returns.dropna(subset=["ret"])
    if output_path:
        ensure_parent(output_path)
        returns.to_parquet(output_path, index=False)
    return returns
