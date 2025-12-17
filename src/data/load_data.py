"""
Data loading utilities.

Notebook 00 expects a CSV with at least:
- date column (parseable)
- close/adj_close column (price level)

The loader is intentionally strict to prevent silent errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class PriceDataSchema:
    date_col: str = "date"
    price_col: str = "adj_close"


def load_prices_csv(
    csv_path: str | Path,
    schema: PriceDataSchema = PriceDataSchema(),
    tz: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load price data from CSV and return a clean DataFrame with a DatetimeIndex.

    Notes:
    - `tz` is optional. If provided, localizes naive timestamps to this timezone.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Price CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for col in (schema.date_col, schema.price_col):
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in CSV. Found columns: {list(df.columns)}"
            )

    df = df[[schema.date_col, schema.price_col]].copy()
    df[schema.date_col] = pd.to_datetime(df[schema.date_col], errors="coerce")

    if df[schema.date_col].isna().any():
        bad = df[df[schema.date_col].isna()].head(5)
        raise ValueError(f"Some dates could not be parsed. Example rows:\n{bad}")

    df = df.sort_values(schema.date_col).drop_duplicates(subset=[schema.date_col])
    df = df.set_index(schema.date_col)

    if tz is not None:
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)

    df = df.rename(columns={schema.price_col: "price"})
    return df
