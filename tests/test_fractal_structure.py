"""
Tests for the Fractal-Exchange-style structure primitives (item 4 extension):

  * NEURALS  — strength-weighted *horizontal* support/resistance zones that earn
    weight by being re-tested and held (the "multiple bounces" idea),
  * VECTORS  — *sloped* dynamic support/resistance anchored to the last two
    same-type swing pivots, projected to now, that FLIP role on a cross,
  * CONFLUENCE — how many independent structure/flow signals align at price,
  * and the neural/vector bracketing of the dealer-pin estimated close.

Everything is pure and offline (hand-built OHLC frames and dicts in, dicts out),
mirroring the grading tests in test_track_record.py.
"""
import numpy as np
import pandas as pd

from strategies.fractal_indicators import (
    cluster_levels,
    _build_vector,
    _zone_strength,
    compute_vectors,
    score_neural_levels,
    nearest_neural,
    confluence_score,
)
from strategies.fractal_options import compute_dealer_pin_close


def _zigzag():
    """
    Deterministic OHLC zigzag: swing lows climbing 99.5 → 100.0 → 100.5 (a
    rising support) and swing highs flat at 110.5 (a resistance), last close 103.
    Fractals form cleanly with period=2.
    """
    pts = [(0, 105), (3, 100), (6, 110), (9, 100.5), (12, 110), (15, 101), (18, 110), (21, 103)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = xs[-1] + 1
    close = np.interp(np.arange(n), xs, ys)
    high = close.copy()
    low = close.copy()
    for x, y in pts:
        if y >= 108:
            high[x] = y + 0.5
        else:
            low[x] = y - 0.5
    idx = pd.date_range('2025-01-01', periods=n, freq='B')
    return pd.DataFrame({'Open': close, 'High': high, 'Low': low, 'Close': close}, index=idx)


# ── cluster_levels ───────────────────────────────────────────────────────────

def test_cluster_levels_groups_within_tolerance_and_splits_beyond():
    clusters = cluster_levels([100.0, 100.2, 100.1, 110.0, 110.3], tolerance_pct=0.5)
    assert len(clusters) == 2
    assert clusters[0]['count'] == 3 and abs(clusters[0]['center'] - 100.1) < 1e-6
    assert clusters[1]['count'] == 2 and abs(clusters[1]['center'] - 110.15) < 1e-6


def test_cluster_levels_empty_is_empty():
    assert cluster_levels([]) == []
    assert cluster_levels([float('nan')]) == []


# ── _build_vector (slope projection + flip-on-cross) ─────────────────────────

def test_support_vector_projects_and_holds_when_price_above():
    # Two rising lows: (10, 100) → (20, 101). Slope 0.1/bar; projected to bar 25.
    v = _build_vector([(10, 100.0), (20, 101.0)], last_pos=25, last_close=104.0,
                      anchored_role='support', max_age=120)
    assert v['slope_per_bar'] == 0.1
    assert v['current_value'] == 101.5          # 101 + 0.1*(25-20)
    assert v['role'] == 'support' and v['crossed'] is False
    assert v['direction'] == 'rising'


def test_support_vector_flips_to_resistance_on_cross():
    # Same line, but price is now BELOW the projected support → it flips.
    v = _build_vector([(10, 100.0), (20, 101.0)], last_pos=25, last_close=101.0,
                      anchored_role='support', max_age=120)
    assert v['current_value'] == 101.5
    assert v['crossed'] is True and v['role'] == 'resistance'


def test_resistance_vector_flips_to_support_when_reclaimed():
    v = _build_vector([(10, 110.0), (20, 110.0)], last_pos=25, last_close=112.0,
                      anchored_role='resistance', max_age=120)
    assert v['role'] == 'support' and v['crossed'] is True


def test_vector_needs_two_pivots_and_fresh_anchor():
    assert _build_vector([(10, 100.0)], 25, 104.0, 'support', 120) is None
    # Newer anchor 20, last_pos 200 → age 180 > max_age 120 → too stale.
    assert _build_vector([(10, 100.0), (20, 101.0)], 200, 104.0, 'support', 120) is None


# ── _zone_strength (touch + bounce counting) ─────────────────────────────────

def test_zone_strength_counts_support_bounces():
    # Two bars wick into a 100 band and close back above (bounces); one closes
    # inside the band (touch, not a bounce).
    df = pd.DataFrame({
        'High': [103, 103, 101],
        'Low':  [99.9, 99.8, 99.9],
        'Close': [102, 102, 100.0],   # first two close above band, last closes in-band
    })
    s = _zone_strength(df, center=100.0, tolerance_pct=0.4, side='support')
    assert s['touches'] == 3
    assert s['bounces'] == 2


def test_zone_strength_counts_resistance_bounces():
    df = pd.DataFrame({
        'High': [110.1, 110.2, 110.0],
        'Low':  [107, 107, 109.9],
        'Close': [108, 108, 110.0],   # first two reject (close below), last closes in-band
    })
    s = _zone_strength(df, center=110.0, tolerance_pct=0.4, side='resistance')
    assert s['touches'] == 3
    assert s['bounces'] == 2


# ── compute_vectors / score_neural_levels (integration on the zigzag) ────────

def test_compute_vectors_on_zigzag():
    v = compute_vectors(_zigzag(), period=2, max_age=50)
    sv, rv = v['support_vector'], v['resistance_vector']
    assert sv is not None and rv is not None
    assert sv['direction'] == 'rising' and sv['role'] == 'support'
    assert abs(sv['current_value'] - 101.0) < 0.5
    assert rv['role'] == 'resistance'
    assert abs(rv['current_value'] - 110.5) < 0.5
    # Support sits below price, resistance above.
    assert sv['current_value'] < v['spot'] < rv['current_value']


def test_compute_vectors_insufficient_data_is_safe():
    df = _zigzag().head(3)
    v = compute_vectors(df, period=2)
    assert v['support_vector'] is None and v['resistance_vector'] is None


def test_score_neural_levels_finds_repeated_zones():
    nz = score_neural_levels(_zigzag(), period=2, tolerance_pct=1.0, lookback=50)
    assert nz['support_zones'] and nz['resistance_zones']
    sup = nz['support_zones'][0]
    res = nz['resistance_zones'][0]
    # Three swing lows cluster into one ~100 zone; three highs into one ~110.5.
    assert abs(sup['center'] - 100.0) < 1.0 and sup['n_pivots'] == 3
    assert abs(res['center'] - 110.5) < 1.0 and res['n_pivots'] == 3
    # Strength rewards repeated pivots even before realized bounces.
    assert sup['strength'] >= 2


def test_score_neural_levels_insufficient_data_is_safe():
    nz = score_neural_levels(_zigzag().head(3), period=2)
    assert nz == {'support_zones': [], 'resistance_zones': []}


# ── nearest_neural ───────────────────────────────────────────────────────────

def test_nearest_neural_picks_correct_side_strongest_first():
    zones_sup = [
        {'center': 90.0, 'strength': 5},   # strongest but far below
        {'center': 98.0, 'strength': 2},
    ]
    # Strength-sorted input → first eligible (<= spot) wins.
    assert nearest_neural(zones_sup, spot=100.0, side='support')['center'] == 90.0
    # A support zone above spot is ineligible.
    zones_above = [{'center': 105.0, 'strength': 9}]
    assert nearest_neural(zones_above, spot=100.0, side='support') is None
    assert nearest_neural(None, 100.0, 'support') is None


# ── confluence_score ─────────────────────────────────────────────────────────

def _vectors(sv_role='support', rv_role='resistance'):
    return {
        'support_vector': {'role': sv_role, 'current_value': 99.0},
        'resistance_vector': {'role': rv_role, 'current_value': 110.0},
    }


def test_confluence_bullish_stack_is_high_when_three_align():
    neurals = {
        'support_zones': [{'center': 99.7, 'strength': 4, 'bounces': 4}],
        'resistance_zones': [{'center': 120.0, 'strength': 4, 'bounces': 4}],
    }
    c = confluence_score(spot=100.0, vectors=_vectors(), neurals=neurals,
                         pc_bias='bullish', gamma_strength=0.7)
    assert c['direction'] == 'bullish'
    assert c['bull'] >= 3 and c['label'] == 'high'
    assert c['pin'] is True            # sticky long-gamma regime


def test_confluence_vector_flip_turns_bearish():
    # Support vector lost (flipped to resistance) + resistance caps + bearish flow.
    c = confluence_score(spot=100.0, vectors=_vectors(sv_role='resistance'),
                         neurals=None, pc_bias='bearish', gamma_strength=0.2)
    assert c['direction'] == 'bearish'
    assert c['bear'] >= 2
    assert c['pin'] is False


def test_confluence_empty_is_neutral_low():
    c = confluence_score(spot=100.0)
    assert c['direction'] == 'neutral'
    assert c['score'] == 0 and c['label'] == 'low'
    assert c['factors'] == []


# ── dealer-pin neural/vector bracketing ──────────────────────────────────────

_IV = {'daily_expected_move': 4.0, 'expected_move_1sigma': 5.0, 'days_to_expiry': 1}
_FR = {'support_levels': [(None, 592.0)], 'resistance_levels': [(None, 608.0)]}


def _gex(rows):
    return pd.DataFrame([{'strike': s, 'net_gex': g} for s, g in rows])


def test_strong_neural_resistance_caps_the_pin():
    gex = _gex([(600, 5e8), (603, 9e8), (606, 3e8)])
    neurals = {
        'support_zones': [{'center': 595.0, 'strength': 4}],
        'resistance_zones': [{'center': 602.0, 'strength': 5}],
    }
    out = compute_dealer_pin_close(600.0, 605.0, gex, _FR, _IV, 'transitional',
                                   neurals=neurals)
    assert out['estimated_close'] <= 602.0 + 1e-9


def test_weak_neural_does_not_constrain():
    # Strength below NEURAL_BRACKET_MIN_STRENGTH must be ignored.
    gex = _gex([(600, 5e8), (603, 9e8), (606, 3e8)])
    neurals = {
        'support_zones': [],
        'resistance_zones': [{'center': 602.0, 'strength': 1}],
    }
    out = compute_dealer_pin_close(600.0, 605.0, gex, _FR, _IV, 'transitional',
                                   neurals=neurals)
    base = compute_dealer_pin_close(600.0, 605.0, gex, _FR, _IV, 'transitional')
    assert out['estimated_close'] == base['estimated_close']


def test_held_support_vector_floors_the_pin():
    # Pin would drift down toward max-pain 595, but a held support vector at 598
    # (beyond the near-spot bracket gate of 0.35 * daily_em = 1.4) keeps the
    # estimate at/above 598.
    gex = _gex([(595, 6e8), (596, 4e8)])
    vectors = {
        'support_vector': {'role': 'support', 'current_value': 598.0},
        'resistance_vector': None,
    }
    out = compute_dealer_pin_close(600.0, 595.0, gex, _FR, _IV, 'transitional',
                                   vectors=vectors)
    assert out['estimated_close'] >= 598.0 - 1e-9


def test_vector_hugging_spot_cannot_clamp_the_pin():
    # A vector inside the bracket gate (599.5, only 0.5 from spot with
    # min_gap = 0.35 * 4.0 = 1.4) says nothing about the close — without the
    # gate it glued the estimate onto spot all session.
    gex = _gex([(595, 6e8), (596, 4e8)])
    vectors = {
        'support_vector': {'role': 'support', 'current_value': 599.5},
        'resistance_vector': None,
    }
    out = compute_dealer_pin_close(600.0, 595.0, gex, _FR, _IV, 'transitional',
                                   vectors=vectors)
    base = compute_dealer_pin_close(600.0, 595.0, gex, _FR, _IV, 'transitional')
    assert out['estimated_close'] == base['estimated_close']
    assert out['estimated_close'] < 599.5


# ── fractal-dimension degenerate windows (the vertical-spike artifact) ──────

def test_fd_flat_window_yields_nan_not_random_walk():
    """A flat tape (zero variance — typical thin pre-market 5m bars) has no
    meaningful dimension. The old 1.5 fallback made the intraday FD line hop
    vertically between 1.5 and the ~1.0 clip floor of near-flat windows."""
    import numpy as np
    import pandas as pd
    from strategies.fractal_indicators import calculate_fractal_dimension

    n = 80
    close = np.full(n, 100.0)               # perfectly flat
    df = pd.DataFrame({'Close': close},
                      index=pd.date_range('2026-06-09 04:00', periods=n, freq='5min'))
    fd = calculate_fractal_dimension(df, window=30)
    assert fd.iloc[30:].isna().all()        # no fabricated 1.5 values


def test_fd_normal_series_still_computes():
    import numpy as np
    import pandas as pd
    from strategies.fractal_indicators import calculate_fractal_dimension

    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 0.3, 200))
    df = pd.DataFrame({'Close': close},
                      index=pd.date_range('2026-01-01', periods=200, freq='B'))
    fd = calculate_fractal_dimension(df, window=30)
    vals = fd.dropna()
    assert len(vals) > 150
    assert ((vals >= 1.0) & (vals <= 2.0)).all()
