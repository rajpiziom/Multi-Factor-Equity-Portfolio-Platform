from __future__ import annotations

from io import BytesIO, StringIO
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

from .utils import annualize_return, annualize_vol

FF_DAILY_ZIP_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"


def performance_metrics(daily_returns: pd.Series, rf_daily: pd.Series | None = None) -> pd.Series:
    dr = daily_returns.dropna()
    if rf_daily is None:
        rf_daily = pd.Series(0.0, index=dr.index)
    excess = dr - rf_daily.reindex(dr.index).fillna(0)

    ann_ret = annualize_return(dr)
    ann_vol = annualize_vol(dr)
    sharpe = (excess.mean() / dr.std()) * np.sqrt(252) if dr.std() != 0 else 0.0

    eq = (1 + dr).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1
    max_dd = dd.min() if not dd.empty else 0.0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    monthly = (1 + dr).resample("M").prod() - 1
    return pd.Series(
        {
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "hit_rate_monthly": (monthly > 0).mean(),
            "best_month": monthly.max() if not monthly.empty else np.nan,
            "worst_month": monthly.min() if not monthly.empty else np.nan,
        }
    )


def fetch_ff_daily() -> pd.DataFrame:
    resp = requests.get(FF_DAILY_ZIP_URL, timeout=30)
    resp.raise_for_status()
    with ZipFile(BytesIO(resp.content)) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith('.csv')][0]
        raw = zf.read(name).decode("utf-8", errors="ignore")

    lines = raw.splitlines()
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.startswith(",Mkt-RF"):
            start = i + 1
            continue
        if start is not None and line.strip() == "":
            end = i
            break
    if start is None:
        raise RuntimeError("Unable to parse Ken French daily factors file")
    body = "\n".join(lines[start:end])
    ff = pd.read_csv(StringIO(body), header=None, names=["date", "Mkt-RF", "SMB", "HML", "RF"])
    ff["date"] = pd.to_datetime(ff["date"].astype(str), format="%Y%m%d", errors="coerce")
    ff = ff.dropna(subset=["date"]).set_index("date").sort_index()
    ff = ff / 100.0
    return ff


def factor_regression(strategy_returns: pd.Series, ff_daily: pd.DataFrame, factors: list[str] | None = None) -> pd.Series:
    if factors is None:
        factors = ["Mkt-RF", "SMB", "HML"]

    y = strategy_returns.rename("strategy").to_frame()
    merged = y.join(ff_daily[factors + ["RF"]], how="inner").dropna()
    merged["excess"] = merged["strategy"] - merged["RF"]

    x = sm.add_constant(merged[factors])
    model = sm.OLS(merged["excess"], x).fit()

    out = {
        "alpha_annualized": model.params["const"] * 252,
        "r_squared": model.rsquared,
    }
    for f in factors:
        out[f"beta_{f}"] = model.params.get(f, np.nan)
        out[f"t_{f}"] = model.tvalues.get(f, np.nan)
    out["t_alpha"] = model.tvalues.get("const", np.nan)
    return pd.Series(out)
