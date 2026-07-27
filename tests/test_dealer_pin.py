"""
Unit tests for the dealer-pin estimated-close model (item 4).

`compute_dealer_pin_close` answers: given current dealer positioning
(max-pain + gamma) and recent swing structure, where does hedging flow most
likely drag price into the close of the nearest expiry?

These tests are intentionally self-contained — they exercise the pure helper
with synthetic chains and never touch the network — so they run standalone or
as part of the wider offline suite.
"""
import pandas as pd

from strategies.fractal_options import compute_dealer_pin_close, compute_max_pain


SPOT = 600.0
IV_RANGE = {
    'daily_expected_move': 4.0,
    'expected_move_1sigma': 5.0,
    'days_to_expiry': 1,
}
FRACTALS = {
    'support_levels': [(None, 592.0)],
    'resistance_levels': [(None, 608.0)],
}


def _gex(strikes_net):
    return pd.DataFrame([{'strike': s, 'net_gex': g} for s, g in strikes_net])


def test_positive_gamma_pins_price_toward_the_magnet():
    """Long/positive-gamma regime is sticky: price is dragged to the pin."""
    gex = _gex([(596, -2e8), (600, 5e8), (603, 9e8), (606, 3e8)])
    out = compute_dealer_pin_close(SPOT, 601.0, gex, FRACTALS, IV_RANGE, 'transitional')

    assert out['gamma_regime'].startswith('positive')
    assert out['gamma_strength'] > 0.7
    # Pin target sits between max-pain (601) and the gamma magnet (603).
    assert 601.0 <= out['pin_target'] <= 603.0
    # Sticky regime => meaningful pull and an estimate above spot, toward the pin.
    assert out['pull_fraction'] > 0.5
    assert SPOT < out['estimated_close'] <= out['pin_target']
    assert out['direction'] == 'up'


def test_negative_gamma_is_slippery_and_barely_pins():
    """Short/negative-gamma regime: weak pull, estimate stays near spot."""
    gex = _gex([(596, -7e8), (600, -4e8), (603, 1e8), (606, -5e8)])
    out = compute_dealer_pin_close(SPOT, 601.0, gex, FRACTALS, IV_RANGE, 'trending')

    assert out['gamma_regime'].startswith('negative')
    assert out['gamma_strength'] < 0.3
    assert out['pull_fraction'] < 0.2
    # With almost no pinning force the close estimate hugs spot.
    assert abs(out['estimated_close'] - SPOT) < 1.0


def test_unreachable_pin_is_clamped_to_expected_move_envelope():
    """A max-pain far beyond the 1-sigma move can't be the close — clamp it."""
    gex = _gex([(610, 4e8), (615, 9e8)])
    out = compute_dealer_pin_close(SPOT, 615.0, gex, FRACTALS, IV_RANGE, 'choppy')

    # Pin target is the far strike, but the estimate is bounded by spot + EM.
    assert out['pin_target'] >= 614.0
    assert out['estimated_close'] <= SPOT + IV_RANGE['expected_move_1sigma'] + 1e-6
    # Reachability penalty keeps confidence below the easy-pin case.
    assert out['confidence'] < 90


def test_gamma_magnet_prefers_strike_evaluated_gex():
    """REGRESSION (pivot-snaps-to-spot): when the profile carries the
    spot-invariant strike-evaluated column (net_gex_k), the magnet must come
    from it — net_gex peaks at whatever strike is nearest spot (a
    near-delta-spike on 0DTE) and made the pin chase the tape."""
    gex = pd.DataFrame([
        # net_gex says the magnet is the ATM 600 strike (spot-gamma artifact);
        # net_gex_k says the dealer commitment actually lives at 603. Both sit
        # inside the reachability window so this test isolates COLUMN choice —
        # window behaviour is covered separately below.
        {'strike': 600.0, 'net_gex': 9e8, 'net_gex_k': 2e8},
        {'strike': 603.0, 'net_gex': 1e8, 'net_gex_k': 8e8},
    ])
    out = compute_dealer_pin_close(SPOT, 601.0, gex, FRACTALS, IV_RANGE, 'transitional')
    assert out['gamma_pin_strike'] == 603.0
    # Aggregate regime still reads the spot-evaluated column.
    assert out['gamma_strength'] == 1.0


def test_empty_gex_falls_back_to_max_pain_anchor():
    """No usable gamma profile => anchor on max-pain with the default pull."""
    out = compute_dealer_pin_close(SPOT, 599.0, pd.DataFrame(), FRACTALS, IV_RANGE, 'transitional')

    assert out['gamma_pin_strike'] is None
    assert out['pin_target'] == 599.0
    # Default 0.45 pull from 600 toward 599 lands just below spot.
    assert 599.0 <= out['estimated_close'] < SPOT
    assert out['direction'] == 'down'


def test_estimate_band_brackets_the_point_estimate():
    """The low/high band straddles the estimate by half a daily expected move."""
    gex = _gex([(600, 5e8), (603, 9e8)])
    out = compute_dealer_pin_close(SPOT, 601.0, gex, FRACTALS, IV_RANGE, 'transitional')

    assert out['estimate_low'] < out['estimated_close'] < out['estimate_high']
    half = IV_RANGE['daily_expected_move'] / 2.0
    assert out['estimate_high'] - out['estimated_close'] == round(half, 2)


def test_handles_zero_expected_move_without_dividing_by_zero():
    """Degenerate IV input must not raise — EM falls back to ~1% of spot."""
    iv = {'daily_expected_move': 0.0, 'expected_move_1sigma': 0.0, 'days_to_expiry': 1}
    gex = _gex([(600, 5e8), (603, 9e8)])
    out = compute_dealer_pin_close(SPOT, 601.0, gex, FRACTALS, iv, 'transitional')

    assert out['estimated_close'] > 0
    assert isinstance(out['confidence'], float)


def test_session_open_anchor_decouples_estimate_from_live_spot():
    """With a fixed anchor, intraday spot drift must not re-center the pull.

    Same positioning, two different live spots: anchored estimates differ only
    through the (spot-based) reachability clamp — here both are reachable, so
    they are identical. Unanchored they re-center on each spot.
    """
    gex = _gex([(596, -2e8), (600, 5e8), (603, 9e8), (606, 3e8)])
    a = compute_dealer_pin_close(600.0, 601.0, gex, FRACTALS, IV_RANGE,
                                 'transitional', anchor_price=600.0)
    b = compute_dealer_pin_close(602.0, 601.0, gex, FRACTALS, IV_RANGE,
                                 'transitional', anchor_price=600.0)
    assert a['estimated_close'] == b['estimated_close']
    assert a['anchor_source'] == 'session_open'

    ua = compute_dealer_pin_close(600.0, 601.0, gex, FRACTALS, IV_RANGE, 'transitional')
    ub = compute_dealer_pin_close(602.0, 601.0, gex, FRACTALS, IV_RANGE, 'transitional')
    assert ua['estimated_close'] != ub['estimated_close']
    assert ua['anchor_source'] == 'spot'


def test_anchored_estimate_still_respects_reachability_from_spot():
    """The anchor moves the pull origin, not the physics: the estimate stays
    inside spot ± expected move."""
    gex = _gex([(610, 4e8), (615, 9e8)])
    out = compute_dealer_pin_close(SPOT, 615.0, gex, FRACTALS, IV_RANGE,
                                   'choppy', anchor_price=608.0)
    assert out['estimated_close'] <= SPOT + IV_RANGE['expected_move_1sigma'] + 1e-6


def test_max_pain_prefers_volume_on_request():
    """0DTE chains carry stale overnight OI; prefer_volume weights today's flow.

    OI concentrates at 605, today's volume at 595 — default max pain follows
    OI, the 0DTE path follows volume.
    """
    calls = pd.DataFrame({'strike': [595.0, 605.0],
                          'openInterest': [0, 50000], 'volume': [40000, 0]})
    puts = pd.DataFrame({'strike': [595.0, 605.0],
                         'openInterest': [0, 50000], 'volume': [40000, 0]})

    assert compute_max_pain(calls, puts, 600.0) == 605.0
    assert compute_max_pain(calls, puts, 600.0, prefer_volume=True) == 595.0


def test_max_pain_prefer_volume_falls_back_to_oi_when_no_volume():
    calls = pd.DataFrame({'strike': [595.0, 605.0],
                          'openInterest': [0, 50000], 'volume': [0, 0]})
    puts = pd.DataFrame({'strike': [595.0, 605.0],
                         'openInterest': [0, 50000], 'volume': [0, 0]})
    assert compute_max_pain(calls, puts, 600.0, prefer_volume=True) == 605.0


# ── Directional asymmetry regression ────────────────────────────────────
# Audit of 104 logged forecasts found the magnet sat a median +2.28 daily EM
# ABOVE spot and beyond 1 EM 94% of the time, because argmax over all strikes
# just returns the board's largest wall (in SPY, the long-gamma call stack).
# With `estimate = anchor + pull * (pin_target - anchor)` and pull >= 0, that
# made a downward estimate mathematically unreachable: 0 of 30 graded pin
# forecasts ever called "down". These lock the fix in place.

def test_magnet_ignores_unreachable_wall_beyond_expected_move():
    """The biggest wall on the board is not a pin candidate if price cannot
    reach it by expiry. Window is 1.0 * daily_em = 4.0 -> [596, 604]."""
    gex = _gex([(603, 2e8), (640, 9e9)])
    out = compute_dealer_pin_close(SPOT, 601.0, gex, FRACTALS, IV_RANGE, 'transitional')

    assert out['gamma_pin_strike'] == 603.0, "far 640 wall must not win the argmax"
    assert out['magnet_windowed'] is True


def test_magnet_can_sit_below_spot_when_put_gamma_dominates():
    """THE core regression: near-money put-side gamma must be able to win, so
    the estimate can resolve DOWNWARD. Previously impossible by construction."""
    gex = _gex([(597, 9e8), (603, 1e8)])
    out = compute_dealer_pin_close(SPOT, 598.0, gex, FRACTALS, IV_RANGE, 'transitional')

    assert out['gamma_pin_strike'] == 597.0
    assert out['pin_target'] < SPOT
    assert out['estimated_close'] < SPOT, "a down pin must be expressible"
    assert out['direction'] == 'down'


def test_magnet_falls_back_to_full_board_when_window_is_empty():
    """If nothing positive sits inside the window, keep the old behaviour
    rather than losing the magnet entirely."""
    gex = _gex([(640, 9e9)])
    out = compute_dealer_pin_close(SPOT, 601.0, gex, FRACTALS, IV_RANGE, 'transitional')

    assert out['gamma_pin_strike'] == 640.0
    assert out['magnet_windowed'] is False


# ── Short-gamma drift ───────────────────────────────────────────────────

def _short_gamma_gex():
    """Overwhelmingly negative GEX -> gamma_strength below the drift threshold."""
    return _gex([(597, -9e8), (600, -8e8), (603, -9e8)])


def test_short_gamma_projects_drift_instead_of_collapsing_onto_spot():
    """Short gamma is amplification, not a weak pin. Previously pull -> 0 put
    the estimate exactly on spot and it was graded as a directional miss."""
    out = compute_dealer_pin_close(
        SPOT, 601.0, _short_gamma_gex(), FRACTALS, IV_RANGE, 'trending',
        momentum_sign=-1.0,
    )

    assert out['pin_mode'] == 'drift'
    assert out['estimated_close'] < SPOT, "down momentum must project downward"
    assert out['direction'] == 'down'
    assert abs(out['estimated_close'] - SPOT) > 0.01, "must not land on spot"


def test_short_gamma_drift_follows_momentum_sign_upward():
    out = compute_dealer_pin_close(
        SPOT, 601.0, _short_gamma_gex(), FRACTALS, IV_RANGE, 'trending',
        momentum_sign=1.0,
    )
    assert out['pin_mode'] == 'drift'
    assert out['estimated_close'] > SPOT
    assert out['direction'] == 'up'


def test_short_gamma_without_momentum_stays_on_the_pin_path():
    """No usable momentum sign => nothing to continue. Fall back to pinning
    rather than inventing a direction."""
    out = compute_dealer_pin_close(
        SPOT, 601.0, _short_gamma_gex(), FRACTALS, IV_RANGE, 'trending',
        momentum_sign=None,
    )
    assert out['pin_mode'] == 'pin'


def test_long_gamma_ignores_momentum_and_still_pins():
    """Momentum must not leak into the pin path — long gamma is mean-reverting,
    so a strong momentum sign should not turn it into a drift forecast."""
    gex = _gex([(603, 9e8), (600, 5e8)])
    out = compute_dealer_pin_close(
        SPOT, 601.0, gex, FRACTALS, IV_RANGE, 'transitional', momentum_sign=-1.0,
    )
    assert out['pin_mode'] == 'pin'
    assert out['estimated_close'] > SPOT, "should still be dragged to the 603 magnet"
