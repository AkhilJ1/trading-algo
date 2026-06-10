"""Feature-log builder — the ML-ready point-in-time snapshot row.

Pure-function tests: a fake composite-analysis result in, the FeatureLog row
out. Pins the research-driven derived features (gap %, close location value,
range vs expected move, momentum signs, max-pain distance) and the rule that
a missing input logs blank (None) instead of raising or fabricating.
"""

import numpy as np
import pandas as pd
import pytest

from feature_log import build_feature_row, _mom_sign, _rsi14
from sheets_logger import FEATURE_HEADERS, _feature_row_list


def _hist(n=260, start=700.0, step=0.2, last_bar=None):
    idx = pd.date_range('2025-06-01', periods=n, freq='B')
    close = start + step * np.arange(n)
    df = pd.DataFrame({
        'Open': close - 1.0, 'High': close + 2.0, 'Low': close - 2.0,
        'Close': close, 'Volume': np.full(n, 1_000_000.0),
    }, index=idx)
    if last_bar:
        for k, v in last_bar.items():
            df.iloc[-1, df.columns.get_loc(k)] = v
    return df


def _result(**over):
    base = {
        'ticker': 'SPY', 'expiry': '2026-06-10', 'spot_price': 735.7,
        'spot_source': 'live_override', 'source': 'schwab', 'stale': False,
        'floor': 729.9, 'ceiling': 741.6, 'bias': 'BULLISH', 'confidence': 55,
        'max_pain': 735.0, 'market_regime': 'choppy', 'fractal_dimension': 1.42,
        'parkinson_rv': 0.14,
        'estimated_close': {'estimated_close': 735.5, 'pin_target': 735.0,
                            'gamma_strength': 0.4, 'gamma_pin_strike': 736.0,
                            'pull_fraction': 0.3},
        'gex_df': pd.DataFrame({'net_gex': [1e8, -3e7]}),
        'gex_boundaries': {'gex_floor': 730.0, 'gex_ceiling': 740.0},
        'options_walls': {'strongest_call_wall': 740.0, 'strongest_put_wall': 730.0},
        'put_call_ratios': {'pc_ratio_oi': 1.45, 'pc_ratio_volume': 1.2},
        'iv_skew': {'skew_ratio': 1.08},
        'iv_range': {'iv_used': 0.15, 'daily_expected_move': 6.0,
                     'expected_move_1sigma': 6.0, 'days_to_expiry': 1},
        'vrp': {'vrp_pct': 12.5},
        'vix_term_structure': {'vix_spot': 18.9, 'vix_3m': 20.1,
                               'ratio': 0.94, 'structure': 'contango'},
        'price_df': _hist(),
    }
    base.update(over)
    return base


def test_row_has_every_header_field_except_stamps():
    row = build_feature_row(_result(), 'pre_open')
    missing = [h for h in FEATURE_HEADERS
               if h not in row and h not in ('date', 'timestamp')]
    assert missing == []


def test_dealer_block_and_net_gex_sum():
    row = build_feature_row(_result(), 'pre_open')
    assert row['gex_net'] == 7e7                     # 1e8 + (-3e7)
    assert row['max_pain'] == 735.0
    # spot 735.7 vs max pain 735 → about -0.095%
    assert row['dist_max_pain_pct'] == pytest.approx(-0.0951, abs=2e-3)
    assert row['pcr_oi'] == 1.45 and row['call_wall'] == 740.0


def test_gap_pct_only_pre_open_and_uses_prior_close():
    ovn = {'prior_close': 730.0, 'ovn_high': 738.0, 'ovn_low': 729.0}
    pre = build_feature_row(_result(), 'pre_open', overnight=ovn)
    post = build_feature_row(_result(), 'post_close', overnight=ovn)
    assert pre['gap_pct'] == pytest.approx((735.7 / 730.0 - 1) * 100, abs=1e-3)
    assert post['gap_pct'] is None                   # gap is a pre-open concept
    assert pre['ovn_range_pct'] == pytest.approx((738.0 / 729.0 - 1) * 100, abs=1e-3)


def test_close_location_value_and_range_vs_em():
    # Last bar: High 740, Low 730, Close 738 → CLV (738-730)/10 = 0.8;
    # range 10 vs daily EM 6 → 1.667.
    res = _result(price_df=_hist(last_bar={'High': 740.0, 'Low': 730.0,
                                           'Close': 738.0}))
    row = build_feature_row(res, 'post_close')
    assert row['bar_clv'] == pytest.approx(0.8, abs=1e-6)
    assert row['range_vs_em'] == pytest.approx(10.0 / 6.0, abs=1e-3)


def test_momentum_signs_on_rising_series():
    row = build_feature_row(_result(), 'pre_open')
    assert (row['mom_1m_sign'], row['mom_12m_sign']) == (1, 1)
    falling = _hist(step=-0.2, start=900.0)
    row2 = build_feature_row(_result(price_df=falling), 'pre_open')
    assert (row2['mom_1m_sign'], row2['mom_12m_sign']) == (-1, -1)


def test_missing_inputs_log_blank_not_crash():
    res = _result(gex_df=None, price_df=pd.DataFrame(),
                  estimated_close=None, iv_range={}, vrp={},
                  vix_term_structure={}, put_call_ratios={}, iv_skew={},
                  options_walls={}, gex_boundaries={}, max_pain=None)
    row = build_feature_row(res, 'post_close')
    assert row['gex_net'] is None and row['est_close'] is None
    assert row['rsi14'] is None and row['range_vs_em'] is None
    # And the serialized sheet row is fully JSON-safe (no NaN/None leakage).
    cells = _feature_row_list(row)
    assert len(cells) == len(FEATURE_HEADERS)
    assert all(c == c for c in cells)                # no NaN survives


def test_mom_sign_and_rsi_edge_cases():
    short = pd.Series([1.0, 2.0, 3.0])
    assert _mom_sign(short, 12) is None
    assert _rsi14(short) is None
