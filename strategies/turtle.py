"""
Turtle Trading System 2 — Donchian Channel Breakout
-----------------------------------------------------
Based on the original rules taught by Richard Dennis & William Eckhardt
to the "Turtle Traders" in 1983.

System 2 (longer-term):
  Entry : price breaks above the 55-day highest high  → long
  Exit  : price breaks below the 20-day lowest low
  Stop  : 2 × ATR(20) from entry price

Position sizing uses ATR-based "N-units":
  Unit = (Account × 1%) / ATR
  Max 4 units per position (pyramid on 0.5N moves).
  The backtest here uses simplified fixed-capital allocation.

Historical note: the Turtle system achieves low win rates (~30-40%) but
profits from the few very large winners — a classic "let profits run" strategy.
Best suited for: strongly trending markets; suffers in choppy/range-bound conditions.
"""

import pandas as pd
import numpy as np


def _atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def generate_signals(
    df: pd.DataFrame,
    entry_n: int = 55,
    exit_n: int  = 20,
    atr_n: int   = 20,
) -> pd.DataFrame:
    """
    Returns df with 'strategy_signal' column:
      +1  = buy (55-day high breakout)
      -1  = sell/exit (20-day low breakdown)
       0  = hold
    Also adds 'Turtle_ATR' column for position sizing reference.
    """
    df = df.copy()

    upper_entry = df['High'].rolling(entry_n).max().shift(1)
    lower_exit  = df['Low'].rolling(exit_n).min().shift(1)
    df['Turtle_ATR'] = _atr(df, atr_n)

    long_entry = df['High'] > upper_entry
    long_exit  = df['Low']  < lower_exit

    df['strategy_signal'] = 0
    df.loc[long_entry, 'strategy_signal'] =  1
    df.loc[long_exit,  'strategy_signal'] = -1
    return df
