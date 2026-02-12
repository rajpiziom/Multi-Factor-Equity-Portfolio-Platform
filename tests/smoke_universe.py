from __future__ import annotations

import tempfile
import time
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import universe as uni
from src.portfolio import make_target_weights_with_universe


def test_load_historical_from_cache() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "sp500_membership_history.csv"
        raw = pd.DataFrame(
            {
                "date": ["2020-01-02", "2024-01-02"],
                "tickers": ["AAPL,MSFT,BRK.B", "AAPL,MSFT,NVDA,TSLA"],
            }
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(cache, index=False)

        cfg = {
            "universe": {
                "historical_cache_path": str(cache),
                "historical_cache_max_age_days": 30,
                "historical_source_priority": ["github_historical_csv"],
                "request_timeout_seconds": 20,
            }
        }

        hist = uni.load_sp500_membership_history(cfg)
        u1 = uni.get_universe_for_date(pd.Timestamp("2020-01-31"), hist)
        u2 = uni.get_universe_for_date(pd.Timestamp("2024-01-31"), hist)
        assert len(u1) != len(u2)
        assert "BRK-B" in u1
        assert "NVDA" in u2


def test_ticker_normalization_historical_pathway() -> None:
    assert uni._normalize_ticker(" brk.b ") == "BRK-B"
    assert uni._to_stooq_ticker("bf.b") == "BF-B.US"


def test_portfolio_uses_asof_universe_only() -> None:
    signals = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")] * 3,
            "ticker": ["AAPL", "MSFT", "TSLA"],
            "momentum": [0.2, 0.1, 0.9],
        }
    )
    returns = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-31")] * 3 + [pd.Timestamp("2024-01-30")] * 3,
            "ticker": ["AAPL", "MSFT", "TSLA", "AAPL", "MSFT", "TSLA"],
            "ret": [0.0, 0.0, 0.0, 0.01, 0.02, -0.01],
        }
    )
    universe_by_date = {pd.Timestamp("2024-01-31"): ["AAPL", "MSFT"]}

    w = make_target_weights_with_universe(
        signals=signals,
        returns_panel=returns,
        factor_col="momentum",
        universe_by_date=universe_by_date,
        quantile=0.5,
    )
    assert set(w["ticker"].unique()) <= {"AAPL", "MSFT"}
    assert "TSLA" not in set(w["ticker"].unique())


def main() -> None:
    start = time.time()
    test_load_historical_from_cache()
    test_ticker_normalization_historical_pathway()
    test_portfolio_uses_asof_universe_only()
    print(f"smoke_universe: PASS in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
