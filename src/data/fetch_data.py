from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


def download_sp500(
    out_path: str | Path,
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    ticker: str = "^GSPC"
) -> pd.DataFrame:
    """
    Download S&P 500 daily data using yfinance and save a clean CSV.

    Output schema (guaranteed):
    - date
    - open
    - high
    - low
    - close
    - adj_close
    - volume
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    prices = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        group_by="column",
        multi_level_index=False,
        progress=False
    )

    if prices.empty:
        raise RuntimeError("yfinance returned an empty DataFrame.")

    # Defensive check against unexpected multi-index
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    prices = prices.reset_index()

    # Ensure flat columns
    prices = prices.copy()
    prices.columns = [c.lower().replace(" ", "_") for c in prices.columns]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_path, index=False)

    return prices
