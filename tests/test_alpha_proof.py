"""
Alpha-proof tests (Step 4).

These are the automated proof that our "alpha" is measured honestly:

  * the backtest engine charges realistic transaction costs and the cost
    actually bites (more cost -> less money, exactly as modeled),
  * "alpha" is defined as strategy return minus the buy-and-hold benchmark
    over the same window, and
  * the out-of-sample scorecard only stamps ALPHA CONFIRMED when the edge
    beats buy & hold, clears a Sharpe floor, and is not overfit — net of costs.

Everything here is deterministic and offline: synthetic price paths with
hand-built signals, plus the pure scorecard reducer. No network, no RNG.
"""
import numpy as np
import pandas as pd

from backtest import _run_backtest_on_df
from walk_forward import summarize_scorecard


def _frame(closes):
    """Build a minimal OHLC frame (engine only needs Close) on business days."""
    idx = pd.bdate_range('2020-01-01', periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({'Open': c, 'High': c, 'Low': c, 'Close': c}, index=idx)


def _avoids_the_crash_frame():
    """
    Round-trip price path that ends exactly where it started (buy & hold = 0%):
        100 -> 110 -> 120 (peak) -> 100 -> 80 (trough) -> 90 -> 100 -> 100
    A strategy that is long into the peak, flat through the crash, and long on
    the recovery should produce clearly positive alpha.
    """
    df = _frame([100, 110, 120, 100, 80, 90, 100, 100])
    sig = [0] * len(df)
    sig[0] = 1   # enter at 100
    sig[2] = -1  # exit at the 120 peak  (+20%)
    sig[5] = 1   # re-enter at 90 after the trough
    df['strategy_signal'] = sig
    return df


# ── Cost model ────────────────────────────────────────────────────────────

def test_alpha_is_strategy_minus_buy_and_hold():
    df = _avoids_the_crash_frame()
    r = _run_backtest_on_df(df, 10_000.0, 'strategy_signal', commission_pct=0.0)
    assert r['alpha_pct'] == round(r['total_return_pct'] - r['buy_hold_return_pct'], 2)
    # Buy & hold round-trips to flat; the crash-avoiding strategy makes money.
    assert abs(r['buy_hold_return_pct']) < 1e-6
    assert r['total_return_pct'] > 25.0
    assert r['alpha_pct'] > 25.0


def test_costs_reduce_returns_monotonically():
    df = _avoids_the_crash_frame()
    gross = _run_backtest_on_df(df, 10_000.0, 'strategy_signal', commission_pct=0.0)
    cheap = _run_backtest_on_df(df, 10_000.0, 'strategy_signal', commission_pct=0.0005)
    pricey = _run_backtest_on_df(df, 10_000.0, 'strategy_signal', commission_pct=0.005)
    assert gross['total_return_pct'] > cheap['total_return_pct'] > pricey['total_return_pct']
    # Buy & hold never trades, so its return is unaffected by commission.
    assert gross['buy_hold_return_pct'] == pricey['buy_hold_return_pct']


def test_round_trip_cost_matches_the_model_exactly():
    """Flat price, one round trip: final capital == initial * (1 - c)^2."""
    df = _frame([100, 100, 100])
    df['strategy_signal'] = [1, -1, 0]  # buy bar 0, sell bar 1
    c = 0.01
    r = _run_backtest_on_df(df, 10_000.0, 'strategy_signal', commission_pct=c)
    expected_final = 10_000.0 * (1 - c) ** 2
    assert abs(r['final_capital'] - expected_final) < 1e-6
    assert abs(r['total_return_pct'] - ((1 - c) ** 2 - 1) * 100) < 1e-6


def test_no_trades_leaves_capital_untouched_and_zero_alpha_on_flat():
    df = _frame([100, 100, 100, 100])
    df['strategy_signal'] = [0, 0, 0, 0]
    r = _run_backtest_on_df(df, 10_000.0, 'strategy_signal', commission_pct=0.005)
    assert r['num_trades'] == 0
    assert r['final_capital'] == 10_000.0
    assert r['total_return_pct'] == 0.0
    assert r['alpha_pct'] == 0.0


# ── Scorecard gate ─────────────────────────────────────────────────────────

def _wf(avg_test_sharpe=0.8, oos_alpha=3.0, overfit_ratio=1.4, **extra):
    base = {
        'ticker': 'SPY', 'strategy': 'Fractal', 'n_folds': 6,
        'avg_train_sharpe': avg_test_sharpe * overfit_ratio,
        'avg_test_sharpe': avg_test_sharpe,
        'oos_return': 5.0, 'oos_buy_hold': 5.0 - oos_alpha, 'oos_alpha': oos_alpha,
        'oos_win_rate': 55.0, 'overfit_ratio': overfit_ratio,
        'commission_pct': 0.0005,
    }
    base.update(extra)
    return base


def test_scorecard_confirms_real_oos_alpha():
    card = summarize_scorecard(_wf(avg_test_sharpe=0.8, oos_alpha=3.0, overfit_ratio=1.4))
    assert card['passed'] is True
    assert card['verdict'] == 'ALPHA CONFIRMED'
    assert card['checks'] == {'beats_benchmark': True, 'clears_sharpe': True, 'not_overfit': True}
    assert card['reasons'] == []


def test_scorecard_fails_when_it_cannot_beat_buy_and_hold():
    card = summarize_scorecard(_wf(oos_alpha=-1.5))
    assert card['passed'] is False
    assert card['verdict'] == 'NOT PROVEN'
    assert card['checks']['beats_benchmark'] is False
    assert any('beat buy & hold' in r for r in card['reasons'])


def test_scorecard_fails_on_weak_sharpe():
    card = summarize_scorecard(_wf(avg_test_sharpe=0.10, oos_alpha=3.0))
    assert card['passed'] is False
    assert card['checks']['clears_sharpe'] is False
    assert any('Sharpe' in r for r in card['reasons'])


def test_scorecard_fails_on_overfitting():
    card = summarize_scorecard(_wf(avg_test_sharpe=0.5, oos_alpha=3.0, overfit_ratio=3.5))
    assert card['passed'] is False
    assert card['checks']['not_overfit'] is False
    assert any('Overfit' in r for r in card['reasons'])


def test_scorecard_passes_through_no_data_errors():
    card = summarize_scorecard({'error': 'No data for SPY'})
    assert card['passed'] is False
    assert card['verdict'] == 'NO DATA'
    assert card['reasons'] == ['No data for SPY']


def test_scorecard_thresholds_are_configurable():
    wf = _wf(avg_test_sharpe=0.2, oos_alpha=0.5, overfit_ratio=1.2)
    # Default Sharpe floor (0.30) fails this; a looser floor passes it.
    assert summarize_scorecard(wf)['passed'] is False
    assert summarize_scorecard(wf, min_oos_sharpe=0.15)['passed'] is True
