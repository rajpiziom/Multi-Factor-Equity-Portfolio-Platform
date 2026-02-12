from __future__ import annotations

import pandas as pd


def run_backtest(
    returns_panel: pd.DataFrame,
    target_weights: pd.DataFrame,
    transaction_cost_bps: float = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns_panel = returns_panel[["date", "ticker", "ret"]].copy()
    returns_panel["date"] = pd.to_datetime(returns_panel["date"])
    target_weights["date"] = pd.to_datetime(target_weights["date"])

    returns_by_date = {
        d: frame.set_index("ticker")["ret"] for d, frame in returns_panel.groupby("date")
    }
    targets_by_date = {
        d: frame.set_index("ticker")["weight"] for d, frame in target_weights.groupby("date")
    }

    dates = sorted(returns_by_date.keys())
    current_weights = pd.Series(dtype=float)
    daily = []
    turnover_rows = []
    tc_rate = transaction_cost_bps / 10000.0

    for dt in dates:
        rets = returns_by_date[dt]

        if dt in targets_by_date:
            tgt = targets_by_date[dt]
            union = current_weights.index.union(tgt.index)
            turnover = (current_weights.reindex(union, fill_value=0) - tgt.reindex(union, fill_value=0)).abs().sum() / 2
            cost = turnover * tc_rate
            current_weights = tgt.copy()
        else:
            turnover = 0.0
            cost = 0.0

        if current_weights.empty:
            gross_ret = 0.0
        else:
            aligned = current_weights.reindex(rets.index, fill_value=0)
            gross_ret = (aligned * rets).sum()

        net_ret = gross_ret - cost
        daily.append({"date": dt, "gross_ret": gross_ret, "net_ret": net_ret, "cost": cost})
        turnover_rows.append({"date": dt, "turnover": turnover, "cost": cost})

        if not current_weights.empty:
            post = current_weights * (1 + rets.reindex(current_weights.index).fillna(0))
            denom = post.abs().sum()
            current_weights = post / denom if denom > 0 else post

    return pd.DataFrame(daily), pd.DataFrame(turnover_rows)
