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


# ── Vectors & Neurals (Fractal-Exchange-style structure) ──────────────────
#
# Two structural primitives, built on the same Williams pivots above, that mimic
# how the Fractal Exchange / RCG read a chart:
#
#   * NEURALS  — *horizontal* support/resistance zones that earn strength by
#     being re-tested and *held* (their "multiple bounces" idea). A level that
#     has rejected price five times is a stronger neural than a one-touch level.
#
#   * VECTORS  — *sloped* (dynamic) support/resistance lines anchored to the two
#     most recent same-type pivots and projected to the current bar. A vector
#     acts as support while price holds above it and FLIPS to resistance once
#     price crosses below (and vice-versa) — the "vector cross" they trade.
#
# Everything here is pure and offline (DataFrame in, dicts out) so it is unit-
# testable without a network, exactly like the grading logic in track_record.py.


def _fractal_pivot_positions(df: pd.DataFrame, period: int = FRACTAL_PERIOD):
    """
    Return (highs, lows) where each is a list of (positional_index, price) for
    confirmed Williams pivots, oldest first. Positional index is 0..n-1 so we
    can compute a slope per bar regardless of the (datetime) index.
    """
    if 'fractal_high' not in df.columns or 'fractal_low' not in df.columns:
        df = add_williams_fractals(df, period=period)
    highs, lows = [], []
    h = df['fractal_high'].values
    l = df['fractal_low'].values
    for pos in range(len(df)):
        if not np.isnan(h[pos]):
            highs.append((pos, float(h[pos])))
        if not np.isnan(l[pos]):
            lows.append((pos, float(l[pos])))
    return highs, lows


def _build_vector(pivots, last_pos, last_close, anchored_role, max_age):
    """
    Build one vector dict from the two most recent same-type pivots.

    `anchored_role` is the role the line plays in its origin trend:
      'support'    for a line through swing LOWS (rising/falling support),
      'resistance' for a line through swing HIGHS.
    The *current* role flips if price has crossed to the other side.
    """
    if len(pivots) < 2:
        return None
    (x1, y1), (x2, y2) = pivots[-2], pivots[-1]
    if x2 == x1:
        return None
    if (last_pos - x2) > max_age:        # anchor too stale to trust
        return None

    slope = (y2 - y1) / (x2 - x1)
    current = y2 + slope * (last_pos - x2)
    if current <= 0:
        return None

    # Flip-on-cross: a low-anchored line is support while price is above it.
    if anchored_role == 'support':
        crossed = last_close < current
        role = 'resistance' if crossed else 'support'
    else:
        crossed = last_close > current
        role = 'support' if crossed else 'resistance'

    distance = current - last_close
    return {
        'anchor_role': anchored_role,
        'role': role,
        'crossed': bool(crossed),
        'slope_per_bar': round(slope, 6),
        'direction': 'rising' if slope > 0 else ('falling' if slope < 0 else 'flat'),
        'current_value': round(current, 2),
        'anchors': [(int(x1), round(y1, 2)), (int(x2), round(y2, 2))],
        'age_bars': int(last_pos - x2),
        'distance': round(distance, 2),
        'distance_pct': round(distance / last_close * 100, 3) if last_close else None,
    }


def compute_vectors(
    df: pd.DataFrame,
    period: int = FRACTAL_PERIOD,
    max_age: int = 120,
) -> dict:
    """
    Derive the two active "vectors" — a low-anchored (support-origin) line and a
    high-anchored (resistance-origin) line — projected to the latest bar.

    Returns {
        'support_vector':    {...} | None,   # anchored on the last two swing lows
        'resistance_vector': {...} | None,   # anchored on the last two swing highs
        'spot': <last close>,
    }
    Each vector dict carries its current `role` (which flips on a cross),
    `current_value` (the line projected to now), slope/direction, the two
    anchor pivots, how stale the newer anchor is, and signed distance to price.
    """
    if df is None or len(df) < (2 * period + 2):
        return {'support_vector': None, 'resistance_vector': None,
                'spot': float(df['Close'].iloc[-1]) if df is not None and len(df) else None}

    if 'fractal_high' not in df.columns:
        df = add_williams_fractals(df, period=period)

    highs, lows = _fractal_pivot_positions(df, period=period)
    last_pos = len(df) - 1
    last_close = float(df['Close'].iloc[-1])

    return {
        'support_vector': _build_vector(lows, last_pos, last_close, 'support', max_age),
        'resistance_vector': _build_vector(highs, last_pos, last_close, 'resistance', max_age),
        'spot': round(last_close, 2),
    }


def cluster_levels(prices, tolerance_pct: float = 0.4):
    """
    Greedy 1-D clustering of price levels: group values that sit within
    `tolerance_pct` of the running cluster mean. Returns a list of
    {'center', 'count', 'members'} sorted by center ascending.
    """
    vals = sorted(float(p) for p in prices if p is not None and not np.isnan(p))
    clusters = []
    for p in vals:
        if clusters and abs(p - clusters[-1]['center']) <= clusters[-1]['center'] * tolerance_pct / 100.0:
            c = clusters[-1]
            c['members'].append(p)
            c['center'] = sum(c['members']) / len(c['members'])
            c['count'] = len(c['members'])
        else:
            clusters.append({'center': p, 'count': 1, 'members': [p]})
    for c in clusters:
        c['center'] = round(c['center'], 2)
        c['members'] = [round(m, 2) for m in c['members']]
    return clusters


def _zone_strength(df: pd.DataFrame, center: float, tolerance_pct: float, side: str) -> dict:
    """
    Count, over `df`, how many bars tested a horizontal zone at `center` and how
    many of those *held* (rejected price) — the neural "bounce count".

      side='support':    a touch = bar Low enters the band; a bounce = it also
                         closed back ABOVE the band (rejection from below-support).
      side='resistance': a touch = bar High enters the band; a bounce = it also
                         closed back BELOW the band.
    """
    tol = center * tolerance_pct / 100.0
    lo, hi = center - tol, center + tol
    touches = bounces = 0
    for _, bar in df.iterrows():
        if side == 'support':
            # The bar's low dipped into the band (a test of support).
            if lo <= bar['Low'] <= hi:
                touches += 1
                if bar['Close'] > hi:        # closed back above → it held
                    bounces += 1
        else:
            # The bar's high pushed into the band (a test of resistance).
            if lo <= bar['High'] <= hi:
                touches += 1
                if bar['Close'] < lo:        # closed back below → it held
                    bounces += 1
    return {'touches': touches, 'bounces': bounces}


def score_neural_levels(
    df: pd.DataFrame,
    period: int = FRACTAL_PERIOD,
    tolerance_pct: float = 0.4,
    lookback: int = 120,
) -> dict:
    """
    Strength-weighted horizontal "neural" zones.

    Cluster recent Williams pivots into price zones, then score each zone by how
    many times price tested it and *bounced* (rejected) over the lookback — the
    Fractal-Exchange "multiple bounces" notion that a repeatedly-held level is a
    stronger neural than a one-touch level.

    Returns {
      'support_zones':    [ {center, n_pivots, touches, bounces, strength}, ... ],
      'resistance_zones': [ ... ],
    }
    each sorted strongest-first (strength = bounces, tie-broken by touches).
    """
    if df is None or len(df) < (2 * period + 2):
        return {'support_zones': [], 'resistance_zones': []}

    if 'fractal_high' not in df.columns:
        df = add_williams_fractals(df, period=period)

    window = df.tail(lookback)
    res_clusters = cluster_levels(window['fractal_high'].dropna().tolist(), tolerance_pct)
    sup_clusters = cluster_levels(window['fractal_low'].dropna().tolist(), tolerance_pct)

    def _score(clusters, side):
        out = []
        for c in clusters:
            s = _zone_strength(window, c['center'], tolerance_pct, side)
            out.append({
                'center': c['center'],
                'n_pivots': c['count'],
                'touches': s['touches'],
                'bounces': s['bounces'],
                # Strength blends pivot agreement with realized bounces so a zone
                # is "strong" both for being a repeated pivot and for holding.
                'strength': s['bounces'] + c['count'] - 1,
            })
        return sorted(out, key=lambda z: (z['strength'], z['bounces'], z['touches']), reverse=True)

    return {
        'support_zones': _score(sup_clusters, 'support'),
        'resistance_zones': _score(res_clusters, 'resistance'),
    }


def nearest_neural(zones, spot: float, side: str):
    """
    Strongest neural zone on the correct side of spot:
      side='support'    → strongest zone at/below spot,
      side='resistance' → strongest zone at/above spot.
    `zones` is already strength-sorted, so the first eligible one wins.
    """
    for z in zones or []:
        if side == 'support' and z['center'] <= spot:
            return z
        if side == 'resistance' and z['center'] >= spot:
            return z
    return None


def confluence_score(
    spot: float,
    vectors: dict = None,
    neurals: dict = None,
    pc_bias: str = None,
    gamma_strength: float = None,
    near_pct: float = 0.6,
) -> dict:
    """
    Tally independent pieces of evidence that align at the current price — the
    Fractal-Exchange "high-probability" confluence stack: a neural bounce + a
    vector reclaim/reject + options-flow + the gamma/pin regime.

    Returns {
      'direction': 'bullish'|'bearish'|'neutral',
      'score':   net signed count (bull − bear),
      'bull', 'bear': raw tallies,
      'factors': [ {name, direction, detail}, ... ],
      'label':   'high'|'medium'|'low',
      'pin':     bool,   # sticky long-gamma regime (mean-reverting)
    }
    """
    factors = []
    bull = bear = 0

    def add(name, direction, detail):
        nonlocal bull, bear
        factors.append({'name': name, 'direction': direction, 'detail': detail})
        if direction == 'bullish':
            bull += 1
        elif direction == 'bearish':
            bear += 1

    band = spot * near_pct / 100.0 if spot else 0.0

    # Neurals: sitting just above a strong support zone is bullish; just below a
    # strong resistance zone is bearish.
    if neurals:
        sup = nearest_neural(neurals.get('support_zones'), spot, 'support')
        res = nearest_neural(neurals.get('resistance_zones'), spot, 'resistance')
        if sup and sup.get('strength', 0) >= 1 and 0 <= (spot - sup['center']) <= band * 3:
            add('neural_support', 'bullish', f"strong support {sup['center']} (x{sup['bounces']} bounces)")
        if res and res.get('strength', 0) >= 1 and 0 <= (res['center'] - spot) <= band * 3:
            add('neural_resistance', 'bearish', f"strong resistance {res['center']} (x{res['bounces']} bounces)")

    # Vectors: a held support vector (price above) is bullish; a held resistance
    # vector (price below) is bearish; a fresh cross flips the sign.
    if vectors:
        sv = vectors.get('support_vector')
        rv = vectors.get('resistance_vector')
        if sv:
            if sv['role'] == 'support':
                add('support_vector', 'bullish', f"holding rising/declining support @ {sv['current_value']}")
            else:
                add('support_vector', 'bearish', f"lost support vector @ {sv['current_value']} (flipped)")
        if rv:
            if rv['role'] == 'resistance':
                add('resistance_vector', 'bearish', f"capped by resistance vector @ {rv['current_value']}")
            else:
                add('resistance_vector', 'bullish', f"reclaimed resistance vector @ {rv['current_value']} (flipped)")

    # Options-flow sentiment.
    if pc_bias in ('bullish', 'bearish'):
        add('put_call_flow', pc_bias, f"P/C bias {pc_bias}")

    pin = bool(gamma_strength is not None and gamma_strength > 0.5)

    score = bull - bear
    if score > 0:
        direction = 'bullish'
    elif score < 0:
        direction = 'bearish'
    else:
        direction = 'neutral'
    aligned = max(bull, bear)
    label = 'high' if aligned >= 3 else ('medium' if aligned == 2 else 'low')

    return {
        'direction': direction,
        'score': score,
        'bull': bull,
        'bear': bear,
        'factors': factors,
        'label': label,
        'pin': pin,
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
