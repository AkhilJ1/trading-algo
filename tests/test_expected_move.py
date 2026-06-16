"""
Tests for the intraday-honest expected move (strategies.fractal_options
.compute_iv_expected_move).

A 0DTE chain expires at today's close, so its expected move must shrink with the
fraction of the cash session still ahead — full daily move at/before the open
(the recorder's pre-open path is unchanged), a fraction of it by mid-afternoon.
Multi-day expiries are dominated by the days ahead and keep the whole-day move.
Pure/offline: hand-built frames in, dict out, with `now_et` injected so the
clock is deterministic.
"""
import math
from datetime import datetime, date, time as dtime, timedelta

import pandas as pd

from strategies.fractal_options import compute_iv_expected_move


def _chain(spot=600.0, iv=0.16):
    strikes = [spot - 2, spot - 1, spot, spot + 1, spot + 2]
    df = pd.DataFrame({
        'strike': strikes,
        'impliedVolatility': [iv] * 5,
        'openInterest': [1000] * 5,
        'volume': [500] * 5,
    })
    return df, df


def _at(hh, mm):
    return datetime.combine(date.today(), dtime(hh, mm))


def test_0dte_preopen_equals_full_daily_move():
    calls, puts = _chain()
    today = date.today().strftime('%Y-%m-%d')
    pre = compute_iv_expected_move(calls, puts, 600.0, today, now_et=_at(9, 0))
    assert pre['days_to_expiry'] == 0
    assert pre['session_fraction_remaining'] == 1.0
    # Pre-open 0DTE move is unchanged from the old full-day convention, so the
    # GitHub pre-open recorder records exactly what it did before.
    assert pre['daily_expected_move'] == round(600.0 * 0.16 * math.sqrt(1 / 365.0), 2)
    assert pre['expected_move_1sigma'] == pre['daily_expected_move']


def test_0dte_move_shrinks_into_the_bell():
    calls, puts = _chain()
    today = date.today().strftime('%Y-%m-%d')
    pre = compute_iv_expected_move(calls, puts, 600.0, today, now_et=_at(9, 0))
    noon = compute_iv_expected_move(calls, puts, 600.0, today, now_et=_at(12, 30))
    late = compute_iv_expected_move(calls, puts, 600.0, today, now_et=_at(15, 30))
    assert 1.0 > noon['session_fraction_remaining'] > late['session_fraction_remaining']
    assert pre['daily_expected_move'] > noon['daily_expected_move'] > late['daily_expected_move']
    assert pre['expected_move_1sigma'] > noon['expected_move_1sigma'] > late['expected_move_1sigma']


def test_multiday_move_is_time_invariant():
    calls, puts = _chain()
    exp = (date.today() + timedelta(days=7)).strftime('%Y-%m-%d')
    morning = compute_iv_expected_move(calls, puts, 600.0, exp, now_et=_at(9, 0))
    afternoon = compute_iv_expected_move(calls, puts, 600.0, exp, now_et=_at(15, 0))
    assert morning['days_to_expiry'] == 7
    # Multi-day expiries are not session-scaled: the daily move is the same all day.
    assert morning['session_fraction_remaining'] == 1.0
    assert afternoon['session_fraction_remaining'] == 1.0
    assert morning['daily_expected_move'] == afternoon['daily_expected_move']
    assert morning['expected_move_1sigma'] == afternoon['expected_move_1sigma']
