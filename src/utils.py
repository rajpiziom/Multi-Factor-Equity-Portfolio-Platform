from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def annualize_return(daily_returns, periods_per_year: int = 252) -> float:
    if daily_returns.empty:
        return 0.0
    total = (1 + daily_returns).prod()
    years = len(daily_returns) / periods_per_year
    return total ** (1 / years) - 1 if years > 0 else 0.0


def annualize_vol(daily_returns, periods_per_year: int = 252) -> float:
    return daily_returns.std() * (periods_per_year ** 0.5)
