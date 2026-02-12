# Multi-Factor Equity Portfolio Platform

Research-grade, config-driven Python platform for factor investing with free data.

## What it does
- Loads S&P 500 universes in either:
  - `latest` mode (current constituents), or
  - `historical` mode (point-in-time membership by rebalance date).
- Downloads daily historical prices from Stooq with local caching.
- Computes **Momentum (12-1)** and **Low Volatility** signals.
- Forms monthly rebalanced portfolios (long-only or long-short).
- Supports equal-weight and volatility-scaled weighting.
- Models turnover and transaction costs.
- Evaluates performance and runs CAPM/FF3 attribution using Ken French factors.
- Produces tables, charts, and a Markdown report in `outputs/`.

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## One-command run
```bash
python -m src.report --config config.yaml
```

## Universe sourcing, historical membership, and survivorship bias
### Why this matters
Using today’s S&P 500 constituents for old backtests creates **survivorship bias**. Historical mode reduces this by using constituents that were actually in the index on each rebalance date.

### Latest mode
Latest mode uses a source-priority fallback:
1. `github_csv`
2. `stooq_html`
3. `wikipedia_html`

### Historical mode
Historical mode uses a GitHub-hosted historical membership dataset (`fja05680/sp500`) and builds as-of universes by rebalance date.

### Caching and fallback
- Latest and historical universe files are cached to local CSV paths in `config.yaml`.
- If a live fetch fails and cache exists, stale cache is used with a warning.
- If no cache exists and fetch fails, pipeline raises a clear error including attempted sources.

### Ticker normalization
- `.` is replaced with `-`
- `.US` suffix is appended for Stooq routing

### Coverage diagnostics (historical mode)
For each rebalance date the pipeline computes and reports:
- `members_count`
- `priced_members_count`
- `coverage_pct`

It also reports:
- average coverage
- worst coverage and date
- unique historical members vs unique priced tickers

> Remaining limitation: even with historical membership, missing prices for delisted/merged tickers can still create residual bias. Coverage diagnostics are included to make this transparent.

## Key outputs
- `outputs/tables/performance_summary.csv`
- `outputs/tables/factor_regression.csv`
- `outputs/tables/turnover_costs.csv`
- `outputs/tables/top_holdings.csv`
- `outputs/tables/universe_coverage.csv`
- `outputs/charts/equity_curve.png`
- `outputs/charts/drawdown.png`
- `outputs/charts/rolling_sharpe.png`
- `outputs/charts/factor_betas.png`
- `outputs/report.md`
