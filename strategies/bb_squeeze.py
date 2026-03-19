"""
Bollinger Band Squeeze — Volatility Contraction Breakout
----------------------------------------------------------
A "squeeze" occurs when Bollinger Bands (20, 2σ) contract inside
Keltner Channels (20-period EMA ± 1.5 × ATR). This compression of
volatility precedes explosive directional moves.

Entry: BB expands back outside Keltner Channels AND price closes above
       upper BB (long) or below lower BB (short, disabled for long-only).
Exit:  Price closes back below the BB midline (mean reversion complete).

Documented performance: Sharpe 0.5–0.9 on US equities daily bars.
Best suited for: low-volatility consolidation periods that resolve into breakouts.
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


def add_squeeze(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float  = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> pd.DataFrame:
    df = df.copy()

    # Bollinger Bands
    bb_mid   = df['Close'].rolling(bb_period).mean()
    bb_std_s = df['Close'].rolling(bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_s
    bb_lower = bb_mid - bb_std * bb_std_s

    # Keltner Channels
    kc_mid   = df['Close'].ewm(span=kc_period, adjust=False).mean()
    atr_val  = _atr(df, kc_period)
    kc_upper = kc_mid + kc_mult * atr_val
    kc_lower = kc_mid - kc_mult * atr_val

    df['BB_SQ_upper'] = bb_upper
    df['BB_SQ_mid']   = bb_mid
    df['BB_SQ_lower'] = bb_lower
    df['KC_upper']    = kc_upper
    df['KC_lower']    = kc_lower

    # Squeeze: BB inside Keltner
    df['SQ_active'] = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # Squeeze release: was squeezing last bar, not squeezing this bar
    df['SQ_release'] = df['SQ_active'].shift(1) & ~df['SQ_active']

    return df


def generate_signals(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float  = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> pd.DataFrame:
    """
    Returns df with 'strategy_signal' column:
      +1  = buy on squeeze breakout above upper BB
      -1  = exit when price falls back below BB midline
       0  = hold
    """
    df = add_squeeze(df, bb_period, bb_std, kc_period, kc_mult)

    long_entry = df['SQ_release'] & (df['Close'] > df['BB_SQ_upper'])
    long_exit  = df['Close'] < df['BB_SQ_mid']

    df['strategy_signal'] = 0
    df.loc[long_entry, 'strategy_signal'] =  1
    df.loc[long_exit,  'strategy_signal'] = -1
    return df
