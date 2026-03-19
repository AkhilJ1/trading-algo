"""
Fractal Trading Strategy — Backtestable Signals
-------------------------------------------------
Multi-timeframe fractal system:
  - Weekly fractals for major structure (trend direction)
  - Daily fractals for entry/exit timing
  - Fractal dimension as regime filter (only trade in trending regimes)

Entry (+1): Fractal low confirmed + trending regime + price bounces above support
Exit  (-1): Support break OR regime shifts to choppy OR fractal high rejection
"""

import numpy as np
import pandas as pd

from strategies.fractal_indicators import (
    add_williams_fractals,
    calculate_fractal_dimension,
)
from config import FRACTAL_PERIOD, FRACTAL_DIM_WINDOW


def add_weekly_fractals(df: pd.DataFrame, period: int = 3) -> pd.DataFrame:
    """
    Resample daily data to weekly, compute Williams fractals on weekly bars,
    then merge back to daily DataFrame.

    Adds columns: weekly_fractal_high, weekly_fractal_low, weekly_trend
    weekly_trend: +1 if last weekly fractal low > previous one (higher lows),
                  -1 if last weekly fractal high < previous one (lower highs),
                   0 otherwise
    """
    df = df.copy()

    # Resample to weekly OHLCV
    weekly = df.resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum',
    }).dropna()

    if len(weekly) < 2 * period + 3:
        df['weekly_fractal_high'] = np.nan
        df['weekly_fractal_low'] = np.nan
        df['weekly_trend'] = 0
        return df

    weekly = add_williams_fractals(weekly, period=period)

    # Determine weekly trend from fractal structure
    wf_highs = weekly['fractal_high'].dropna()
    wf_lows = weekly['fractal_low'].dropna()

    weekly['weekly_trend'] = 0
    # Higher lows = bullish structure
    if len(wf_lows) >= 2:
        if wf_lows.iloc[-1] > wf_lows.iloc[-2]:
            weekly['weekly_trend'] = 1
        elif wf_lows.iloc[-1] < wf_lows.iloc[-2]:
            weekly['weekly_trend'] = -1

    # Forward-fill weekly fractals and trend to daily
    weekly_reindexed = weekly[['fractal_high', 'fractal_low', 'weekly_trend']].rename(
        columns={'fractal_high': 'weekly_fractal_high', 'fractal_low': 'weekly_fractal_low'}
    )
    # Reindex to daily, forward fill
    weekly_daily = weekly_reindexed.reindex(df.index, method='ffill')
    df['weekly_fractal_high'] = weekly_daily['weekly_fractal_high']
    df['weekly_fractal_low'] = weekly_daily['weekly_fractal_low']
    df['weekly_trend'] = weekly_daily['weekly_trend'].fillna(0).astype(int)

    return df


def generate_signals(
    df: pd.DataFrame,
    fractal_period: int = FRACTAL_PERIOD,
    fd_trending_threshold: float = 1.40,
    fd_choppy_threshold: float = 1.55,
    lookback: int = 5,
    sma_filter: int = 50,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate fractal-based trading signals.

    Entry (+1 buy) requires ALL:
      1. Fractal low confirmed within last `lookback` bars
      2. Fractal dimension < fd_trending_threshold (trending regime)
      3. Price closes above the most recent fractal low (bounce)
      4. Weekly trend is not bearish (>= 0)

    Exit (-1 sell) on ANY:
      1. Price closes below the most recent fractal low (support break)
      2. Fractal dimension > fd_choppy_threshold (regime shift)
      3. Fractal high confirmed AND price closes below it (resistance rejection)

    Parameters
    ----------
    df : DataFrame with OHLCV columns
    fractal_period : int, bars each side for Williams fractal
    fd_trending_threshold : float, FD below this = trending
    fd_choppy_threshold : float, FD above this = choppy
    lookback : int, how many bars back to look for recent fractal confirmation
    sma_filter : int, only buy when price > SMA(sma_filter) for trend confirmation

    Returns
    -------
    DataFrame with added columns: fractal_high, fractal_low, fractal_dimension,
                                   weekly_trend, strategy_signal
    """
    df = df.copy()

    # Add daily fractals
    df = add_williams_fractals(df, period=fractal_period)

    # Add fractal dimension
    fd = calculate_fractal_dimension(df, window=FRACTAL_DIM_WINDOW)
    df['fractal_dimension'] = fd

    # Add SMA trend filter
    df['sma_filter'] = df['Close'].rolling(sma_filter).mean()

    # Add weekly fractals for multi-TF confirmation
    df = add_weekly_fractals(df)

    # Track most recent CONFIRMED fractal levels.
    # A fractal at bar i is only confirmed at bar i + fractal_period.
    # At bar i, we can only use fractals from bar i - fractal_period or earlier.
    n = len(df)
    signals = np.zeros(n, dtype=int)
    last_fractal_low = np.nan
    last_fractal_high = np.nan
    in_position = False

    for i in range(fractal_period * 2 + FRACTAL_DIM_WINDOW, n):
        close = df['Close'].iloc[i]
        current_fd = df['fractal_dimension'].iloc[i]
        weekly_trend = df['weekly_trend'].iloc[i]

        # Only use fractals confirmed by now (at least fractal_period bars ago)
        confirm_limit = i - fractal_period
        for j in range(confirm_limit, -1, -1):
            if not np.isnan(df['fractal_low'].iloc[j]):
                last_fractal_low = df['fractal_low'].iloc[j]
                break
        for j in range(confirm_limit, -1, -1):
            if not np.isnan(df['fractal_high'].iloc[j]):
                last_fractal_high = df['fractal_high'].iloc[j]
                break

        # Check for recently confirmed fractal low
        recent_frac_low = False
        for j in range(max(0, confirm_limit - lookback), confirm_limit + 1):
            if not np.isnan(df['fractal_low'].iloc[j]):
                recent_frac_low = True
                break

        sma_val = df['sma_filter'].iloc[i]
        above_sma = close > sma_val if not np.isnan(sma_val) else False

        if not in_position:
            # ── ENTRY CONDITIONS ──
            if (recent_frac_low
                    and not np.isnan(last_fractal_low)
                    and not np.isnan(current_fd)
                    and current_fd < fd_trending_threshold
                    and close > last_fractal_low
                    and weekly_trend >= 0
                    and above_sma):
                signals[i] = 1
                in_position = True
        else:
            # ── EXIT CONDITIONS ──
            support_break = (not np.isnan(last_fractal_low)
                             and close < last_fractal_low)
            regime_shift = (not np.isnan(current_fd)
                            and current_fd > fd_choppy_threshold)
            # Resistance rejection: confirmed fractal high AND price below it
            confirmed_frac_high = not np.isnan(df['fractal_high'].iloc[confirm_limit]) if confirm_limit >= 0 else False
            resistance_reject = (confirmed_frac_high
                                 and not np.isnan(last_fractal_high)
                                 and close < last_fractal_high)

            if support_break or regime_shift or resistance_reject:
                signals[i] = -1
                in_position = False

    df['strategy_signal'] = signals
    return df
