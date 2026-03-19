"""
Moving Average Crossover Strategy
----------------------------------
Golden cross  (SMA_short crosses above SMA_long) → BUY signal
Death  cross  (SMA_short crosses below SMA_long) → SELL signal
"""

import pandas as pd

from config import MA_LONG, MA_SHORT


def add_moving_averages(
    df: pd.DataFrame,
    short: int = MA_SHORT,
    long: int = MA_LONG,
) -> pd.DataFrame:
    df = df.copy()
    df[f'SMA_{short}'] = df['Close'].rolling(short).mean()
    df[f'SMA_{long}'] = df['Close'].rolling(long).mean()
    return df


def generate_signals(
    df: pd.DataFrame,
    short: int = MA_SHORT,
    long: int = MA_LONG,
) -> pd.DataFrame:
    """
    Add ma_signal column: +1 on golden cross, -1 on death cross, 0 otherwise.
    Also adds SMA columns.
    """
    df = add_moving_averages(df, short, long)
    short_col, long_col = f'SMA_{short}', f'SMA_{long}'

    df['ma_signal'] = 0
    above = (df[short_col] > df[long_col]).astype(int)
    crossover = above.diff()
    df.loc[crossover == 1, 'ma_signal'] = 1    # Golden cross
    df.loc[crossover == -1, 'ma_signal'] = -1  # Death cross
    return df


def current_signal(
    df: pd.DataFrame,
    short: int = MA_SHORT,
    long: int = MA_LONG,
) -> dict:
    """Return current trend state and most recent crossover event."""
    df = generate_signals(df, short, long)
    short_col, long_col = f'SMA_{short}', f'SMA_{long}'
    last = df.iloc[-1]

    trend = 'bullish' if last[short_col] > last[long_col] else 'bearish'

    signal_rows = df[df['ma_signal'] != 0]
    last_crossover = None
    if not signal_rows.empty:
        row = signal_rows.iloc[-1]
        last_crossover = {
            'type': 'golden_cross' if row['ma_signal'] == 1 else 'death_cross',
            'date': row.name,
            'price': round(float(row['Close']), 2),
        }

    return {
        'trend': trend,
        f'sma_{short}': round(float(last[short_col]), 2),
        f'sma_{long}': round(float(last[long_col]), 2),
        'last_crossover': last_crossover,
    }
