"""
RSI + Bollinger Band Buy Signal
--------------------------------
Conditions for a BUY signal:
  1. RSI (14) < 30  — stock is oversold on the daily chart
  2. At least one of the last N candles has a wick (low) that touched or
     pierced the lower Bollinger Band (20-period, 2 std dev)

Signal strength (0–100) is a composite of how oversold RSI is and how
many recent candles touched the lower band.
"""

import pandas as pd
import numpy as np

from config import BB_PERIOD, BB_STD, BB_WICK_LOOKBACK, RSI_OVERSOLD, RSI_PERIOD


def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
    """RSI via Wilder's smoothed EMA method."""
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = BB_PERIOD,
    std_dev: float = BB_STD,
) -> tuple:
    """Returns (upper, mid, lower) Bollinger Band Series."""
    rolling = df['Close'].rolling(period)
    mid = rolling.mean()
    std = rolling.std()
    return mid + std_dev * std, mid, mid - std_dev * std


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['RSI'] = calculate_rsi(df)
    df['BB_upper'], df['BB_mid'], df['BB_lower'] = calculate_bollinger_bands(df)
    return df


def get_buy_signal(
    df: pd.DataFrame,
    rsi_threshold: float = RSI_OVERSOLD,
    wick_lookback: int = BB_WICK_LOOKBACK,
    min_wick_touches: int = 1,
) -> dict:
    """
    Evaluate RSI + lower Bollinger Band wick conditions and return a signal dict.

    A wick touch is defined as: candle Low <= lower Bollinger Band.
    """
    df = add_indicators(df)

    current = df.iloc[-1]
    current_rsi = float(current['RSI'])
    close = float(current['Close'])
    bb_lower = float(current['BB_lower'])
    bb_mid = float(current['BB_mid'])
    bb_upper = float(current['BB_upper'])

    # --- Condition 1: RSI oversold ---
    rsi_oversold = current_rsi < rsi_threshold

    # --- Condition 2: Recent wick(s) touched or pierced the lower BB ---
    recent = df.tail(wick_lookback)
    wick_touches = recent[recent['Low'] <= recent['BB_lower']]
    wick_touched = len(wick_touches) >= min_wick_touches

    buy_signal = rsi_oversold and wick_touched

    # Signal strength (0–100): RSI component + wick frequency component
    rsi_score = max(0.0, rsi_threshold - current_rsi) / rsi_threshold * 50
    wick_score = min(len(wick_touches) / wick_lookback, 1.0) * 50
    strength = round(rsi_score + wick_score, 1)

    touch_dates = [
        str(d.date()) if hasattr(d, 'date') else str(d)
        for d in wick_touches.index
    ]

    return {
        'buy_signal': buy_signal,
        'strength': strength,
        'rsi': round(current_rsi, 2),
        'rsi_oversold': rsi_oversold,
        'close': round(close, 2),
        'bb_lower': round(bb_lower, 2),
        'bb_mid': round(bb_mid, 2),
        'bb_upper': round(bb_upper, 2),
        'wick_touches': len(wick_touches),
        'wick_touched_bb': wick_touched,
        'wick_dates': touch_dates,
    }
