from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .utils import ensure_parent

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
GITHUB_SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
STOOQ_SP500_URL = "https://stooq.com/db/l/?g=major&i=sp500"
GITHUB_SP500_HIST_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes.csv"


def _normalize_ticker(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def _to_stooq_ticker(symbol: str) -> str:
    return f"{_normalize_ticker(symbol)}.US"


def _from_stooq_ticker(stooq_ticker: str) -> str:
    out = str(stooq_ticker).strip().upper()
    return out[:-3] if out.endswith(".US") else out


def _read_with_retries(url: str, timeout: int, headers: dict[str, str] | None = None, retries: int = 3) -> str:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.5**i)
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def _is_cache_fresh(cache_path: Path, max_age_days: int) -> bool:
    if not cache_path.exists():
        return False
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(cache_path.stat().st_mtime, tz=timezone.utc)
    return age <= timedelta(days=max_age_days)


def _normalize_universe_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "Symbol": "ticker",
        "Ticker": "ticker",
        "symbol": "ticker",
        "ticker": "ticker",
        "Name": "security",
        "Security": "security",
        "company": "security",
        "Sector": "sector",
        "GICS Sector": "sector",
        "subIndustry": "sub_industry",
        "GICS Sub-Industry": "sub_industry",
    }
    out = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns}).copy()
    if "ticker" not in out.columns:
        raise ValueError("No ticker column found in universe data")

    for col in ["security", "sector", "sub_industry"]:
        if col not in out.columns:
            out[col] = ""

    out["ticker"] = out["ticker"].map(_normalize_ticker)
    out = out[out["ticker"].astype(bool)]
    out = out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    out["stooq_ticker"] = out["ticker"].map(_to_stooq_ticker)
    return out[["ticker", "security", "sector", "sub_industry", "stooq_ticker"]]


def _from_github_csv(_: int) -> pd.DataFrame:
    return _normalize_universe_columns(pd.read_csv(GITHUB_SP500_URL))


def _from_stooq_html(timeout: int) -> pd.DataFrame:
    html = _read_with_retries(STOOQ_SP500_URL, timeout=timeout)
    tables = pd.read_html(StringIO(html))
    for table in tables:
        cols = [str(c).lower() for c in table.columns]
        if any(c in cols for c in ["symbol", "ticker", "name"]):
            candidate = table.copy()
            if "symbol" in [str(c).lower() for c in candidate.columns]:
                sym_col = [c for c in candidate.columns if str(c).lower() == "symbol"][0]
                candidate = candidate.rename(columns={sym_col: "ticker"})
            return _normalize_universe_columns(candidate)
    raise ValueError("Could not locate ticker table on Stooq S&P 500 page")


def _from_wikipedia_html(timeout: int) -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    html = _read_with_retries(WIKI_SP500_URL, timeout=timeout, headers=headers, retries=4)
    table = pd.read_html(StringIO(html))[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    return _normalize_universe_columns(table)


def load_sp500_universe(config: dict[str, Any]) -> pd.DataFrame:
    uni_cfg = config["universe"]
    cache_path = Path(uni_cfg["cache_path"])
    cache_max_age_days = int(uni_cfg.get("cache_max_age_days", 7))
    timeout = int(uni_cfg.get("request_timeout_seconds", 20))
    source_priority = uni_cfg.get("source_priority", ["github_csv", "stooq_html", "wikipedia_html"])
    max_tickers = uni_cfg.get("max_tickers")

    if _is_cache_fresh(cache_path, cache_max_age_days):
        return pd.read_csv(cache_path)

    loaders = {"github_csv": _from_github_csv, "stooq_html": _from_stooq_html, "wikipedia_html": _from_wikipedia_html}
    attempted: list[str] = []
    last_error: str | None = None

    for source in source_priority:
        attempted.append(source)
        try:
            df = loaders[source](timeout)
            if max_tickers:
                df = df.head(int(max_tickers))
            ensure_parent(cache_path)
            df.to_csv(cache_path, index=False)
            cache_path.with_suffix(cache_path.suffix + ".meta.json").write_text(
                json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "source": source, "row_count": int(len(df))}, indent=2),
                encoding="utf-8",
            )
            return df
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

    if cache_path.exists():
        print(f"WARNING: live universe fetch failed for {attempted}; using stale cache {cache_path}. Last error: {last_error}")
        return pd.read_csv(cache_path)

    raise RuntimeError(f"Unable to load S&P 500 latest universe. Attempted {attempted}. Last error: {last_error}")


def load_sp500_membership_history(config: dict[str, Any]) -> pd.DataFrame:
    uni_cfg = config["universe"]
    cache_path = Path(uni_cfg["historical_cache_path"])
    cache_max_age_days = int(uni_cfg.get("historical_cache_max_age_days", 30))
    timeout = int(uni_cfg.get("request_timeout_seconds", 20))

    raw_df: pd.DataFrame | None = None
    if _is_cache_fresh(cache_path, cache_max_age_days):
        raw_df = pd.read_csv(cache_path)
    else:
        attempted = []
        last_error = None
        for source in uni_cfg.get("historical_source_priority", ["github_historical_csv"]):
            attempted.append(source)
            try:
                if source != "github_historical_csv":
                    raise ValueError(f"unsupported historical source: {source}")
                raw_df = pd.read_csv(GITHUB_SP500_HIST_URL)
                ensure_parent(cache_path)
                raw_df.to_csv(cache_path, index=False)
                cache_path.with_suffix(cache_path.suffix + ".meta.json").write_text(
                    json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "source": source, "row_count": int(len(raw_df))}, indent=2),
                    encoding="utf-8",
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)

        if raw_df is None:
            if cache_path.exists():
                print(f"WARNING: historical fetch failed for {attempted}; using stale cache {cache_path}. Last error: {last_error}")
                raw_df = pd.read_csv(cache_path)
            else:
                raise RuntimeError(
                    f"Unable to load historical S&P 500 membership. Attempted {attempted}. Last error: {last_error}"
                )

    hist = raw_df[["date", "tickers"]].copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date")

    rows: list[dict[str, Any]] = []
    for row in hist.itertuples(index=False):
        date = row.date
        if pd.isna(row.tickers):
            continue
        for token in str(row.tickers).split(","):
            token = token.strip()
            if not token:
                continue
            token = re.sub(r"-\d{6}$", "", token)
            ticker = _normalize_ticker(token)
            rows.append({"date": date, "ticker": ticker, "stooq_ticker": _to_stooq_ticker(ticker), "in_index": True})

    return pd.DataFrame(rows).drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)


def get_universe_for_date(asof_date: pd.Timestamp, membership_history_df: pd.DataFrame) -> list[str]:
    asof = pd.Timestamp(asof_date)
    available_dates = membership_history_df.loc[membership_history_df["date"] <= asof, "date"]
    if available_dates.empty:
        return []
    snapshot_date = available_dates.max()
    snap = membership_history_df[membership_history_df["date"] == snapshot_date]
    return sorted(snap["ticker"].unique().tolist())
