"""
Fractal Indicators — Bill Williams Fractals & Fractal Dimension
---------------------------------------------------------------
Pure price-based indicators for market structure analysis.
"""

import numpy as np
import pandas as pd

from config import FRACTAL_PERIOD, FRACTAL_DIM_WINDOW


def add_williams_fractals(
    df: pd.DataFrame,
    period: int = FRACTAL_PERIOD,
) -> pd.DataFrame:
    """
    Add Bill Williams fractal columns.

    A fractal high: bar whose High is the highest in a (2*period + 1) window.
    A fractal low:  bar whose Low is the lowest in the same window.

    Adds columns: fractal_high, fractal_low (float or NaN).
    """
    df = df.copy()
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)
    frac_h = np.full(n, np.nan)
    frac_l = np.full(n, np.nan)

    for i in range(period, n - period):
        window_h = highs[i - period: i + period + 1]
        if highs[i] == window_h.max() and np.sum(window_h == highs[i]) == 1:
            frac_h[i] = highs[i]

        window_l = lows[i - period: i + period + 1]
        if lows[i] == window_l.min() and np.sum(window_l == lows[i]) == 1:
            frac_l[i] = lows[i]

    df['fractal_high'] = frac_h
    df['fractal_low'] = frac_l
    return df


def get_recent_fractal_levels(
    df: pd.DataFrame,
    n_levels: int = 5,
) -> dict:
    """
    Extract the most recent N fractal high and low levels.

    Returns {
        'resistance_levels': [(date, price), ...],  # newest first
        'support_levels':    [(date, price), ...],
    }
    """
    if 'fractal_high' not in df.columns:
        df = add_williams_fractals(df)

    highs = df.dropna(subset=['fractal_high']).tail(n_levels)
    lows = df.dropna(subset=['fractal_low']).tail(n_levels)

    return {
        'resistance_levels': [
            (idx, float(row['fractal_high']))
            for idx, row in highs.iloc[::-1].iterrows()
        ],
        'support_levels': [
            (idx, float(row['fractal_low']))
            for idx, row in lows.iloc[::-1].iterrows()
        ],
    }


def calculate_fractal_dimension(
    df: pd.DataFrame,
    window: int = FRACTAL_DIM_WINDOW,
) -> pd.Series:
    """
    Fractal Dimension via rescaled range (R/S) analysis.

    Returns Series: ~1.0 = strong trend, ~1.5 = random walk, ~2.0 = choppy.
    """
    close = df['Close'].values
    fd = np.full(len(close), np.nan)

    for i in range(window, len(close)):
        seg = close[i - window: i]
        mean_val = np.mean(seg)
        devs = np.cumsum(seg - mean_val)
        r = np.max(devs) - np.min(devs)
        s = np.std(seg, ddof=1)
        if s > 0 and r > 0:
            h = np.log(r / s) / np.log(len(seg))
            h = np.clip(h, 0.0, 1.0)
            fd[i] = 2.0 - h
        else:
            fd[i] = 1.5

    return pd.Series(fd, index=df.index, name='fractal_dimension')


def classify_regime(fd_value: float) -> str:
    """Classify market regime from fractal dimension value."""
    if np.isnan(fd_value):
        return 'unknown'
    if fd_value < 1.35:
        return 'trending'
    elif fd_value > 1.65:
        return 'choppy'
    return 'transitional'


def compute_range_containment(
    df: pd.DataFrame,
    window: int = 60,
    vol_window: int = 20,
) -> dict:
    """
    Historical range accuracy test using realized volatility as IV proxy.

    For each of the last `window` trading days, computes:
      - 1-sigma range: close +/- (close * realized_vol * sqrt(1/252))
      - 2-sigma range: close +/- 2 * (close * realized_vol * sqrt(1/252))
    Then checks if the NEXT day's close landed within the predicted range.

    Returns dict with containment rates and detailed results.
    """
    close = df['Close']
    log_ret = np.log(close / close.shift(1))

    # Rolling realized volatility (annualized)
    realized_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)

    results_1s = []
    results_2s = []
    daily_results = []

    start_idx = max(vol_window + 1, len(df) - window)
    for i in range(start_idx, len(df) - 1):
        today_close = close.iloc[i]
        rv = realized_vol.iloc[i]
        if np.isnan(rv) or rv <= 0:
            continue

        next_close = close.iloc[i + 1]
        daily_move = today_close * rv * np.sqrt(1 / 252)

        low_1s = today_close - daily_move
        high_1s = today_close + daily_move
        low_2s = today_close - 2 * daily_move
        high_2s = today_close + 2 * daily_move

        in_1s = low_1s <= next_close <= high_1s
        in_2s = low_2s <= next_close <= high_2s

        results_1s.append(in_1s)
        results_2s.append(in_2s)
        daily_results.append({
            'date': df.index[i],
            'close': round(today_close, 2),
            'next_close': round(next_close, 2),
            'rv': round(rv, 4),
            'range_low_1s': round(low_1s, 2),
            'range_high_1s': round(high_1s, 2),
            'range_low_2s': round(low_2s, 2),
            'range_high_2s': round(high_2s, 2),
            'in_1sigma': in_1s,
            'in_2sigma': in_2s,
        })

    n = len(results_1s)
    return {
        'days_tested': n,
        'containment_1sigma_pct': round(sum(results_1s) / n * 100, 1) if n > 0 else 0,
        'containment_2sigma_pct': round(sum(results_2s) / n * 100, 1) if n > 0 else 0,
        'expected_1sigma_pct': 68.3,
        'expected_2sigma_pct': 95.4,
        'daily_results': daily_results,
    }
