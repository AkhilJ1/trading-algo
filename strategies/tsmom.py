"""
Time-Series Momentum (TSMOM) — AQR / Man AHL Institutional Standard
---------------------------------------------------------------------
"Is this asset's own trailing return positive or negative?"

If the trailing 12-month return (skipping the most recent month to avoid
short-term reversal) is positive → hold long.
If negative → exit to cash (long-only version) or go short.

Position size is scaled inversely to recent realized volatility so that
each position contributes equal risk regardless of the stock's volatility.

Academic source: Moskowitz, Ooi & Pedersen (2012) — AQR.
Documented Sharpe across asset classes: 0.6–0.9.
Best suited for: strong directional trends; suffers in choppy/whipsaw markets.
"""

import pandas as pd
import numpy as np


def _realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility over a rolling window."""
    returns = close.pct_change()
    return returns.rolling(window).std() * np.sqrt(252)


def generate_signals(
    df: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int     = 1,
    vol_window: int      = 20,
    target_vol: float    = 0.15,
    multi_speed: bool    = True,
) -> pd.DataFrame:
    """
    Returns df with 'strategy_signal' column:
      +1  = go long (positive momentum)
      -1  = exit / go short (negative momentum)
       0  = flat (no signal yet — not enough history)

    multi_speed=True blends 1-month, 3-month, 6-month, and 12-month signals
    (equal weight) for smoother performance — the institutional standard.
    """
    df = df.copy()

    # Helper: momentum signal for a given lookback
    def _signal(lb_months: int) -> pd.Series:
        lb_days   = lb_months * 21
        skip_days = skip_months * 21
        ret = df['Close'].shift(skip_days) / df['Close'].shift(lb_days) - 1
        return ret.apply(lambda x: 0 if pd.isna(x) else (1 if x > 0 else -1))

    if multi_speed:
        signals = (_signal(1) + _signal(3) + _signal(6) + _signal(lookback_months)) / 4
        raw_signal = signals.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    else:
        raw_signal = _signal(lookback_months)

    # Volatility scaling (optional information column; backtest uses binary long/flat)
    rv = _realized_vol(df['Close'], vol_window)
    df['TSMOM_vol_scale'] = (target_vol / rv.replace(0, np.nan)).clip(upper=3.0)

    df['strategy_signal'] = raw_signal
    return df
