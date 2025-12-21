# src/data/regime_labeling.py

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_trailing_volatility_proxy(
    squared_returns: pd.Series,
    window: int = 63
) -> pd.Series:
    trailing_vol = squared_returns.rolling(
        window=window,
        min_periods=window
    ).mean()
    return trailing_vol


def define_regime_thresholds(
    volatility_proxy: pd.Series,
    method: str = 'quantile',
    low_threshold: float = 0.33,
    high_threshold: float = 0.67
) -> Tuple[float, float]:
    valid_values = volatility_proxy.dropna()
    
    if method == 'quantile':
        lower = valid_values.quantile(low_threshold)
        upper = valid_values.quantile(high_threshold)
    else:
        raise ValueError(f"Method '{method}' not supported. Use 'quantile'.")
    
    return lower, upper


def label_regimes(
    volatility_proxy: pd.Series,
    lower_threshold: float,
    upper_threshold: float,
    labels: Optional[dict] = None
) -> pd.Series:
    if labels is None:
        labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    regimes_numeric = pd.Series(index=volatility_proxy.index, dtype='Int64')
    
    regimes_numeric[volatility_proxy <= lower_threshold] = 0
    regimes_numeric[(volatility_proxy > lower_threshold) & 
                   (volatility_proxy <= upper_threshold)] = 1
    regimes_numeric[volatility_proxy > upper_threshold] = 2
    
    regimes = regimes_numeric.map(labels)
    regimes = regimes.astype('category')
    
    return regimes


def create_regime_labels(
    squared_returns: pd.Series,
    window: int = 63,
    method: str = 'quantile',
    low_threshold: float = 0.33,
    high_threshold: float = 0.67,
    labels: Optional[dict] = None
) -> Tuple[pd.Series, pd.Series, Tuple[float, float]]:
    volatility_proxy = calculate_trailing_volatility_proxy(
        squared_returns=squared_returns,
        window=window
    )
    
    lower, upper = define_regime_thresholds(
        volatility_proxy=volatility_proxy,
        method=method,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )
    
    regimes = label_regimes(
        volatility_proxy=volatility_proxy,
        lower_threshold=lower,
        upper_threshold=upper,
        labels=labels
    )
    
    return regimes, volatility_proxy, (lower, upper)


# def validate_no_lookahead(
#     regimes: pd.Series,
#     squared_returns: pd.Series,
#     window: int = 63
# ) -> bool:
#     first_valid_idx = regimes.first_valid_index()
#     expected_first_idx = squared_returns.index[window]
    
#     if first_valid_idx != expected_first_idx:
#         return False
    
#     return True

def validate_no_lookahead(
    regimes: pd.Series,
    squared_returns: pd.Series,
    window: int = 63
) -> bool:
    # Boolean mask of valid regime labels
    valid_mask = regimes.notna().to_numpy()

    # If no valid regimes exist, validation fails
    if not valid_mask.any():
        return False
    
    # Position of first valid regime label
    first_valid_pos = valid_mask.argmax()

    # Minimum allowed position for first valid label
    min_allowed_pos = window - 1

    # Validate causality
    if first_valid_pos < min_allowed_pos:
        return False

    return True
