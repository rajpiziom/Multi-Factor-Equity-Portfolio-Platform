from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .attribution import run_attribution
from .backtest import run_backtest
from .ingest import build_price_panel, compute_return_panel, fetch_stooq_history
from .portfolio import make_target_weights, make_target_weights_with_universe
from .risk import fetch_ff_daily, performance_metrics
from .signals import compute_month_end_signals
from .universe import get_universe_for_date, load_sp500_membership_history, load_sp500_universe
from .utils import ensure_parent, load_config


def _save_chart(fig, path: str) -> None:
    ensure_parent(path)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _equity_drawdown_charts(strategy: pd.Series, benchmark: pd.Series, chart_dir: Path) -> None:
    eq = (1 + strategy).cumprod()
    bq = (1 + benchmark).cumprod()

    fig, ax = plt.subplots(figsize=(10, 5))
    eq.plot(ax=ax, label="Strategy")
    bq.plot(ax=ax, label="Benchmark", alpha=0.8)
    ax.set_title("Equity Curve")
    ax.legend()
    _save_chart(fig, chart_dir / "equity_curve.png")

    dd = eq / eq.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    dd.plot(ax=ax, color="firebrick")
    ax.set_title("Drawdown")
    _save_chart(fig, chart_dir / "drawdown.png")


def _rolling_sharpe_chart(strategy: pd.Series, chart_dir: Path) -> None:
    roll = strategy.rolling(252).mean() / strategy.rolling(252).std()
    roll = roll * (252**0.5)
    fig, ax = plt.subplots(figsize=(10, 4))
    roll.plot(ax=ax, color="darkgreen")
    ax.set_title("Rolling 12M Sharpe")
    _save_chart(fig, chart_dir / "rolling_sharpe.png")


def _monthly_hist_chart(strategy: pd.Series, chart_dir: Path) -> None:
    monthly = (1 + strategy).resample("M").prod() - 1
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.hist(ax=ax, bins=25, color="steelblue", alpha=0.9)
    ax.set_title("Monthly Return Distribution")
    _save_chart(fig, chart_dir / "monthly_return_hist.png")


def _factor_exposure_chart(regression_df: pd.DataFrame, chart_dir: Path) -> None:
    cols = [c for c in regression_df.columns if c.startswith("beta_")]
    ff3 = regression_df.loc["FF3", cols]
    fig, ax = plt.subplots(figsize=(8, 4))
    ff3.plot(kind="bar", ax=ax, color="slategray")
    ax.set_title("FF3 Beta Exposures")
    _save_chart(fig, chart_dir / "factor_betas.png")


def _compute_historical_universe_and_coverage(
    cfg: dict,
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    membership_history: pd.DataFrame,
) -> tuple[dict[pd.Timestamp, list[str]], pd.DataFrame, list[str]]:
    min_cov = float(cfg["universe"].get("min_price_coverage_pct", 0.85))
    warnings: list[str] = []

    signal_by_date = signals.groupby("date")["ticker"].apply(set)
    priced_by_date = returns.groupby("date")["ticker"].apply(set)

    universe_by_date: dict[pd.Timestamp, list[str]] = {}
    rows = []

    for dt in sorted(signals["date"].unique()):
        dt = pd.Timestamp(dt)
        members = set(get_universe_for_date(dt, membership_history))
        priced = priced_by_date.get(dt, set())
        signalable = signal_by_date.get(dt, set())

        priced_members = members.intersection(priced).intersection(signalable)
        members_count = len(members)
        priced_count = len(priced_members)
        coverage_pct = (priced_count / members_count) if members_count else 0.0

        if coverage_pct < min_cov:
            warnings.append(
                f"Coverage warning on {dt.date()}: {coverage_pct:.1%} ({priced_count}/{members_count}) below threshold {min_cov:.1%}"
            )

        universe_by_date[dt] = sorted(priced_members)
        rows.append(
            {
                "date": dt,
                "members_count": members_count,
                "priced_members_count": priced_count,
                "coverage_pct": coverage_pct,
            }
        )

    coverage_df = pd.DataFrame(rows)
    return universe_by_date, coverage_df, warnings


def run_pipeline(config_path: str) -> None:
    cfg = load_config(config_path)

    universe_mode = cfg["universe"].get("mode", "latest")
    membership_history = None
    coverage_df = pd.DataFrame(columns=["date", "members_count", "priced_members_count", "coverage_pct"])
    coverage_warnings: list[str] = []

    if universe_mode == "historical":
        membership_history = load_sp500_membership_history(cfg)
        start_dt = pd.to_datetime(cfg["dates"]["start"])
        end_dt = pd.to_datetime(cfg["dates"]["end"])
        scoped = membership_history[(membership_history["date"] >= start_dt) & (membership_history["date"] <= end_dt)]
        if scoped.empty:
            scoped = membership_history[membership_history["date"] <= end_dt]
        tickers = sorted(scoped["ticker"].unique().tolist())
        max_tickers = cfg["universe"].get("max_tickers")
        if max_tickers:
            tickers = tickers[: int(max_tickers)]
        universe = pd.DataFrame({"ticker": tickers})
        universe["stooq_ticker"] = universe["ticker"].map(lambda t: f"{t}.US")
        universe["security"] = ""
        universe["sector"] = ""
        universe["sub_industry"] = ""
        universe = universe[["ticker", "security", "sector", "sub_industry", "stooq_ticker"]]
    else:
        universe = load_sp500_universe(cfg)

    ensure_parent(cfg["paths"]["universe_csv"])
    universe.to_csv(cfg["paths"]["universe_csv"], index=False)

    prices = build_price_panel(
        universe=universe,
        start=cfg["dates"]["start"],
        end=cfg["dates"]["end"],
        output_path=cfg["paths"]["prices_parquet"],
    )

    returns = compute_return_panel(prices, output_path=cfg["paths"]["returns_parquet"])
    returns = returns.set_index("date").sort_index()

    signals = compute_month_end_signals(
        returns.reset_index(),
        momentum_lookback_months=cfg["signals"]["momentum_lookback_months"],
        momentum_skip_days=cfg["signals"]["momentum_skip_days"],
        lowvol_window_days=cfg["signals"]["lowvol_window_days"],
    )

    if universe_mode == "historical" and membership_history is not None:
        universe_by_date, coverage_df, coverage_warnings = _compute_historical_universe_and_coverage(
            cfg, signals, returns.reset_index(), membership_history
        )
        target_weights = make_target_weights_with_universe(
            signals=signals,
            returns_panel=returns.reset_index(),
            factor_col="momentum",
            universe_by_date=universe_by_date,
            quantile=cfg["portfolio"]["quantile"],
            long_short=cfg["portfolio"]["long_short"],
            weighting=cfg["portfolio"]["weighting"],
        )
    else:
        target_weights = make_target_weights(
            signals=signals,
            returns_panel=returns.reset_index(),
            factor_col="momentum",
            quantile=cfg["portfolio"]["quantile"],
            long_short=cfg["portfolio"]["long_short"],
            weighting=cfg["portfolio"]["weighting"],
        )

    daily, turnover = run_backtest(
        returns.reset_index(),
        target_weights,
        transaction_cost_bps=cfg["portfolio"]["transaction_cost_bps"],
    )
    daily = daily.set_index("date").sort_index()

    bench = fetch_stooq_history(cfg["benchmark"]["ticker"] + ".US", cfg["dates"]["start"], cfg["dates"]["end"])
    bench_ret = bench["close"].sort_index().pct_change().dropna()

    ff = fetch_ff_daily()
    perf = performance_metrics(daily["net_ret"], ff["RF"])
    regression = run_attribution(daily["net_ret"], ff)

    monthly_returns = (1 + daily["net_ret"]).resample("M").prod() - 1

    default_coverage_csv = "outputs/tables/universe_coverage.csv"
    coverage_csv = cfg["paths"].get("coverage_csv", default_coverage_csv)

    for key in ["monthly_returns_csv", "performance_csv", "regression_csv", "turnover_csv", "holdings_csv", "report_md"]:
        ensure_parent(cfg["paths"][key])
    ensure_parent(coverage_csv)

    monthly_returns.to_csv(cfg["paths"]["monthly_returns_csv"], header=["net_monthly_return"])
    perf.to_frame("value").to_csv(cfg["paths"]["performance_csv"])
    regression.to_csv(cfg["paths"]["regression_csv"])
    turnover.to_csv(cfg["paths"]["turnover_csv"], index=False)
    coverage_df.to_csv(coverage_csv, index=False)

    latest = target_weights[target_weights["date"] == target_weights["date"].max()]
    latest = latest.sort_values("weight", ascending=False).head(20)
    latest.to_csv(cfg["paths"]["holdings_csv"], index=False)

    chart_dir = Path("outputs/charts")
    _equity_drawdown_charts(daily["net_ret"], bench_ret, chart_dir)
    _rolling_sharpe_chart(daily["net_ret"], chart_dir)
    _monthly_hist_chart(daily["net_ret"], chart_dir)
    _factor_exposure_chart(regression, chart_dir)

    avg_cov = float(coverage_df["coverage_pct"].mean()) if not coverage_df.empty else float("nan")
    worst_row = coverage_df.sort_values("coverage_pct").head(1)
    worst_cov = float(worst_row["coverage_pct"].iloc[0]) if not worst_row.empty else float("nan")
    worst_date = str(pd.to_datetime(worst_row["date"].iloc[0]).date()) if not worst_row.empty else "N/A"

    hist_unique = int(membership_history["ticker"].nunique()) if membership_history is not None else int(universe["ticker"].nunique())
    priced_unique = int(returns.reset_index()["ticker"].nunique())

    warnings_block = "\n".join([f"- {w}" for w in coverage_warnings]) if coverage_warnings else "- None"

    report_md = Path(cfg["paths"]["report_md"])
    summary = f"""# Multi-Factor Equity Portfolio Report

## Data & Universe
- Universe mode: {universe_mode}.
- Universe source: {'historical S&P 500 membership snapshots (GitHub + cache)' if universe_mode == 'historical' else 'latest constituents multi-source loader'}.
- Prices: Stooq daily close + volume.
- Risk-free and factor data: Kenneth French daily factors.

## Signals
- Momentum: 12-1 construction (lookback {cfg['signals']['momentum_lookback_months']} months, skip {cfg['signals']['momentum_skip_days']} trading days).
- Low Volatility: negative rolling volatility ({cfg['signals']['lowvol_window_days']} trading-day window).

## Portfolio Construction
- Rebalance: monthly month-end.
- Selection: top {int(cfg['portfolio']['quantile']*100)}% by momentum.
- Weighting: {cfg['portfolio']['weighting']}.
- Transaction costs: {cfg['portfolio']['transaction_cost_bps']} bps per unit turnover.

## Coverage Diagnostics (historical universe)
- Average coverage: {avg_cov:.2%}
- Worst coverage: {worst_cov:.2%} on {worst_date}
- Total unique historical members: {hist_unique}
- Total unique priced tickers: {priced_unique}
- Low coverage warnings:
{warnings_block}

## Results (headline)
- Annualized Return: {perf['annual_return']:.2%}
- Annualized Volatility: {perf['annual_vol']:.2%}
- Sharpe: {perf['sharpe']:.2f}
- Max Drawdown: {perf['max_drawdown']:.2%}

## Robustness & Limitations
- Historical membership reduces survivorship bias compared with fixed latest-constituent universe.
- Delistings/mergers and free-data price availability can still create residual bias; coverage diagnostics quantify this.
- Transaction costs are approximated using turnover only.
"""
    report_md.write_text(summary, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-factor equity platform pipeline.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
