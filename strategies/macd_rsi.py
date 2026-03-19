"""
MACD + RSI Combination Strategy
---------------------------------
Entry: MACD line crosses above signal line AND RSI < rsi_entry (oversold confirmation)
Exit:  MACD line crosses below signal line OR RSI > rsi_exit (overbought)

Documented performance: 73-86% win rate on US indices (QuantifiedStrategies).
Best suited for: trending markets with momentum pullbacks.
"""

import pandas as pd
import numpy as np


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD']        = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta   = df['Close'].diff()
    gain    = delta.clip(lower=0)
    loss    = -delta.clip(upper=0)
    avg_g   = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_l   = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs      = avg_g / avg_l.replace(0, np.nan)
    df['RSI_MACD'] = 100 - (100 / (1 + rs))
    return df


def generate_signals(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    rsi_period: int = 14,
    rsi_entry: float = 45.0,
    rsi_exit: float = 65.0,
) -> pd.DataFrame:
    """
    Returns df with 'strategy_signal' column:
      +1  = buy (enter long)
      -1  = sell (exit long)
       0  = hold
    """
    df = add_macd(df, fast, slow, signal)
    df = add_rsi(df, rsi_period)

    macd_cross_up   = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
    macd_cross_down = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))
    rsi_ok_entry    = df['RSI_MACD'] < rsi_entry
    rsi_overbought  = df['RSI_MACD'] > rsi_exit

    df['strategy_signal'] = 0
    df.loc[macd_cross_up & rsi_ok_entry, 'strategy_signal']     =  1
    df.loc[macd_cross_down | rsi_overbought, 'strategy_signal'] = -1
    return df
