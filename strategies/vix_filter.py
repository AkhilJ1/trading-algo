"""
VIX Regime Filter — Overlay for any strategy.
----------------------------------------------
Fetches ^VIX data and classifies market regime:
  - VIX < 15:  Low vol   — full sizing, trend strategies favored
  - VIX 15-25: Normal    — standard sizing
  - VIX 25-35: Elevated  — half sizing, suppress trend buys
  - VIX > 35:  Crisis    — cash only, suppress ALL buy signals
"""

import numpy as np
import pandas as pd

from data_fetcher import fetch_stock_data


def fetch_vix(period: str = '5y') -> pd.Series:
    """Fetch VIX close prices as a Series indexed by date."""
    df = fetch_stock_data('^VIX', period=period)
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df['Close'].rename('vix')


def classify_vix_regime(vix_value: float) -> str:
    """Classify VIX level into regime category."""
    if np.isnan(vix_value):
        return 'unknown'
    if vix_value < 15:
        return 'low_vol'
    elif vix_value < 25:
        return 'normal'
    elif vix_value < 35:
        return 'elevated'
    return 'crisis'


def apply_vix_filter(
    df: pd.DataFrame,
    signal_col: str = 'strategy_signal',
    vix_series: pd.Series = None,
    suppress_elevated: bool = True,
) -> pd.DataFrame:
    """
    Apply VIX regime filter to an existing signal DataFrame.

    - Crisis (VIX > 35): suppress all buy signals
    - Elevated (VIX 25-35): suppress buy signals if suppress_elevated=True
    - Normal/Low: no change

    Adds columns: vix, vix_regime, vix_filtered_signal
    """
    df = df.copy()

    if vix_series is None:
        vix_series = fetch_vix()
    if vix_series.empty:
        df['vix'] = np.nan
        df['vix_regime'] = 'unknown'
        df['vix_filtered_signal'] = df.get(signal_col, 0)
        return df

    # Align VIX to df index via forward-fill
    vix_aligned = vix_series.reindex(df.index, method='ffill')
    df['vix'] = vix_aligned
    df['vix_regime'] = df['vix'].apply(classify_vix_regime)

    # Filter signals
    filtered = df.get(signal_col, pd.Series(0, index=df.index)).copy()

    # Suppress buys in crisis
    crisis_mask = df['vix_regime'] == 'crisis'
    filtered.loc[crisis_mask & (filtered == 1)] = 0

    # Optionally suppress buys in elevated
    if suppress_elevated:
        elevated_mask = df['vix_regime'] == 'elevated'
        filtered.loc[elevated_mask & (filtered == 1)] = 0

    df['vix_filtered_signal'] = filtered.astype(int)
    return df


def get_vix_sizing_multiplier(vix_value: float) -> float:
    """
    Position size multiplier based on VIX regime.
    Low vol: 1.0, Normal: 1.0, Elevated: 0.5, Crisis: 0.0
    """
    regime = classify_vix_regime(vix_value)
    return {'low_vol': 1.0, 'normal': 1.0, 'elevated': 0.5,
            'crisis': 0.0, 'unknown': 0.5}[regime]
