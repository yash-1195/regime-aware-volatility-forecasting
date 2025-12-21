"""
Preprocessing for Notebook 00:
- compute log returns
- compute squared returns (variance proxy)
- basic validation helpers

Notebook 00 intentionally avoids regime labels and modeling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnConfig:
    price_col: str = "price"
    return_col: str = "log_return"
    squared_return_col: str = "squared_return"
    eps: float = 1e-12


def compute_log_returns(df: pd.DataFrame, cfg: ReturnConfig = ReturnConfig()) -> pd.DataFrame:
    if cfg.price_col not in df.columns:
        raise ValueError(f"Missing required column '{cfg.price_col}'.")

    out = df.copy()
    p = out[cfg.price_col].astype(float)

    if (p <= 0).any():
        bad = out[p <= 0].head(5)
        raise ValueError(
            "Non-positive prices found. Log returns require strictly positive prices. "
            f"Example rows:\n{bad}"
        )

    out[cfg.return_col] = np.log(p).diff()
    out[cfg.squared_return_col] = out[cfg.return_col] ** 2
    return out


def count_missing(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum().sort_values(ascending=False)


def check_date_continuity(
    df: pd.DataFrame,
    freq: Optional[str] = None,
) -> dict:
    """
    Check for missing timestamps given an expected frequency.

    If `freq` is None, this returns basic diagnostics only.
    For daily market data, users often prefer business days ('B').
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    info = {
        "start": idx.min(),
        "end": idx.max(),
        "n_rows": len(df),
        "is_monotonic": idx.is_monotonic_increasing,
    }

    if freq is None:
        return info

    expected = pd.date_range(start=idx.min(), end=idx.max(), freq=freq, tz=idx.tz)
    missing = expected.difference(idx)
    info["expected_n"] = len(expected)
    info["missing_n"] = len(missing)
    info["missing_head"] = missing[:10].to_list()
    return info

def save_processed_returns(
    data: pd.DataFrame,
    return_col: str,
    squared_return_col: str,
    output_dir: Path,
    logger=None
) -> None:
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save returns only
    returns_path = output_dir / "returns.csv"
    data[[return_col]].to_csv(returns_path)
    if logger:
        logger.info(f"Saved returns to: {returns_path}")
    
    # Save squared returns only
    squared_path = output_dir / "squared_returns.csv"
    data[[squared_return_col]].to_csv(squared_path)
    if logger:
        logger.info(f"Saved squared returns to: {squared_path}")
