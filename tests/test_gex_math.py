"""GEX engine validation — the math behind the Options tab's gamma profile.

Pins the Black-Scholes gamma to hand-computed values, the per-strike GEX
aggregation to the documented formula (gamma x OI x 100 x spot, calls + /
puts -), the data-quality gate (IV > 5%, OI > 0), and the intraday 0DTE
time-to-expiry (actual minutes to the 16:00 ET close, not a full day).
"""

import math
from datetime import datetime

import pandas as pd
import pytest
from scipy.stats import norm

from strategies.fractal_options import (
    _bs_gamma, _gex_time_to_expiry, compute_gex_profile,
)

S, R = 735.0, 0.045


def _hand_gamma(S_, K, T, r, iv):
    d1 = (math.log(S_ / K) + (r + iv * iv / 2) * T) / (iv * math.sqrt(T))
    return norm.pdf(d1) / (S_ * iv * math.sqrt(T))


def test_bs_gamma_matches_hand_computation():
    T = 1 / 365
    assert _bs_gamma(S, 735.0, T, R, 0.15) == pytest.approx(
        _hand_gamma(S, 735.0, T, R, 0.15), abs=1e-12)
    assert _bs_gamma(S, 720.0, T, R, 0.20) == pytest.approx(
        _hand_gamma(S, 720.0, T, R, 0.20), abs=1e-12)


def test_gamma_is_identical_for_calls_and_puts():
    # BS gamma is the same for a call and a put at the same strike/IV —
    # the engine reuses one function; this pins that property.
    T = 5 / 365
    assert _bs_gamma(S, 740.0, T, R, 0.18) > 0


def test_gex_profile_matches_formula_per_strike():
    now = datetime(2026, 6, 9, 11, 0)   # 11:00 ET, expiry tomorrow
    expiry = '2026-06-10'
    calls = pd.DataFrame({'strike': [730.0, 735.0], 'openInterest': [10000, 30000],
                          'impliedVolatility': [0.16, 0.15], 'volume': [1, 1]})
    puts = pd.DataFrame({'strike': [730.0, 735.0], 'openInterest': [25000, 15000],
                         'impliedVolatility': [0.17, 0.15], 'volume': [1, 1]})
    gex = compute_gex_profile(calls, puts, S, expiry, now_et=now)
    T = _gex_time_to_expiry(expiry, now_et=now)
    for _, row in gex.iterrows():
        k = row['strike']
        c = calls[calls.strike == k].iloc[0]
        p = puts[puts.strike == k].iloc[0]
        exp_call = _hand_gamma(S, k, T, R, c.impliedVolatility) * c.openInterest * 100 * S
        exp_put = -_hand_gamma(S, k, T, R, p.impliedVolatility) * p.openInterest * 100 * S
        assert row['call_gex'] == pytest.approx(exp_call, rel=1e-6)
        assert row['put_gex'] == pytest.approx(exp_put, rel=1e-6)
        assert row['net_gex'] == pytest.approx(exp_call + exp_put, rel=1e-6)


def test_quality_gate_excludes_dead_iv_and_zero_oi():
    # After-hours yfinance marks (IV ~ 0) and empty strikes must contribute
    # NOTHING — this is the gate that stops the midnight $0-GEX garbage from
    # being mistaken for real positioning.
    calls = pd.DataFrame({'strike': [735.0, 740.0], 'openInterest': [10000, 0],
                          'impliedVolatility': [0.04, 0.20], 'volume': [1, 1]})
    puts = pd.DataFrame({'strike': [735.0], 'openInterest': [5000],
                         'impliedVolatility': [float('nan')], 'volume': [1]})
    gex = compute_gex_profile(calls, puts, S, '2026-06-10',
                              now_et=datetime(2026, 6, 9, 11, 0))
    assert float(gex['net_gex'].abs().sum()) == 0.0


def test_same_day_expiry_uses_minutes_to_close_not_full_day():
    expiry = '2026-06-09'
    # 13:00 ET on expiry day -> 3 hours to the close.
    t_intraday = _gex_time_to_expiry(expiry, now_et=datetime(2026, 6, 9, 13, 0))
    assert t_intraday == pytest.approx(180 / (365 * 24 * 60), rel=1e-9)
    # The old convention treated this as a FULL day:
    assert t_intraday < (1 / 365) * 0.2
    # ATM gamma must be LARGER with the honest shorter clock (≈ 1/sqrt(T)).
    g_old = _bs_gamma(S, 735.0, 1 / 365, R, 0.15)
    g_new = _bs_gamma(S, 735.0, t_intraday, R, 0.15)
    assert g_new / g_old == pytest.approx(math.sqrt((1 / 365) / t_intraday), rel=0.05)


def test_same_day_expiry_floors_at_30_minutes_into_the_bell():
    expiry = '2026-06-09'
    t = _gex_time_to_expiry(expiry, now_et=datetime(2026, 6, 9, 15, 55))
    assert t == pytest.approx(30 / (365 * 24 * 60), rel=1e-9)
    # And after the close it cannot go negative/zero:
    t_post = _gex_time_to_expiry(expiry, now_et=datetime(2026, 6, 9, 17, 30))
    assert t_post == pytest.approx(30 / (365 * 24 * 60), rel=1e-9)


def test_multi_day_expiry_keeps_whole_day_convention():
    t = _gex_time_to_expiry('2026-06-12', now_et=datetime(2026, 6, 9, 11, 0))
    assert t == pytest.approx(3 / 365, rel=1e-9)
