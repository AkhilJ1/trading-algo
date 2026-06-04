"""
Walk-Forward Validation — Anchored expanding-window optimizer.
--------------------------------------------------------------
Splits historical data into expanding train + fixed test windows.
On each fold: sweeps parameters, picks best in-sample Sharpe,
then evaluates out-of-sample on the test window.

Detects overfitting: if train >> test performance, params are overfit.

Usage:
    python walk_forward.py SPY Fractal
"""

import sys
from itertools import product

import numpy as np
import pandas as pd

from config import (
    DATA_PERIOD,
    DEFAULT_COMMISSION_PCT,
    SCORECARD_MAX_OVERFIT,
    SCORECARD_MIN_OOS_ALPHA,
    SCORECARD_MIN_OOS_SHARPE,
)
from data_fetcher import fetch_stock_data
from backtest import _run_backtest_on_df


# ---------------------------------------------------------------------------
# Parameter grids per strategy
# ---------------------------------------------------------------------------

PARAM_GRIDS = {
    'Fractal': {
        'fd_trending_threshold': [1.30, 1.35, 1.40, 1.45],
        'fd_choppy_threshold':   [1.50, 1.55, 1.60],
        'sma_filter':            [20, 50, 100],
        'lookback':              [3, 5, 7],
    },
    'MA Crossover': {
        'short': [20, 50],
        'long':  [100, 200],
    },
    'Ensemble': {
        'threshold': [2, 3, 4],
    },
}


def _get_signal_fn(strategy: str):
    """Get the generate_signals function and signal column for a strategy."""
    if strategy == 'Fractal':
        from strategies.fractal_signals import generate_signals
        return generate_signals, 'strategy_signal'
    elif strategy == 'MA Crossover':
        from strategies.ma_crossover import generate_signals
        return generate_signals, 'ma_signal'
    elif strategy == 'MACD + RSI':
        from strategies.macd_rsi import generate_signals
        return generate_signals, 'strategy_signal'
    elif strategy == 'BB Squeeze':
        from strategies.bb_squeeze import generate_signals
        return generate_signals, 'strategy_signal'
    elif strategy == 'TSMOM':
        from strategies.tsmom import generate_signals
        return generate_signals, 'strategy_signal'
    elif strategy == 'Turtle':
        from strategies.turtle import generate_signals
        return generate_signals, 'strategy_signal'
    elif strategy == 'Ensemble':
        from strategies.ensemble import generate_signals
        return generate_signals, 'strategy_signal'
    else:
        raise ValueError(f'Unknown strategy: {strategy}')


def _backtest_with_params(df, strategy, params, sig_fn, sig_col, commission_pct=0.0):
    """Run a single backtest with given params, return Sharpe and results."""
    try:
        df_sig = sig_fn(df.copy(), **params)
        if sig_col == 'ma_signal' and 'ma_signal' in df_sig.columns:
            df_sig['strategy_signal'] = df_sig['ma_signal']
            sig_col_use = 'strategy_signal'
        else:
            sig_col_use = sig_col
        result = _run_backtest_on_df(
            df_sig, 10_000.0, sig_col_use, commission_pct=commission_pct)
        return result.get('sharpe_ratio', -999), result
    except Exception:
        return -999, {}


def walk_forward_test(
    ticker: str,
    strategy: str = 'Fractal',
    n_splits: int = 8,
    test_months: int = 3,
    param_grid: dict = None,
    commission_pct: float = DEFAULT_COMMISSION_PCT,
) -> dict:
    """
    Anchored walk-forward optimization.

    Parameters
    ----------
    ticker : str
    strategy : str
    n_splits : int, number of train/test folds
    test_months : int, length of each test window in months
    param_grid : dict, parameter grid to sweep (default from PARAM_GRIDS)
    commission_pct : float, per-side transaction cost applied to every fill so
        out-of-sample results are net of costs (defaults to the conservative
        project-wide DEFAULT_COMMISSION_PCT). Pass 0.0 for a gross run.

    Returns
    -------
    dict with:
        - folds: list of per-fold results (train/test Sharpe, OOS alpha vs
          buy & hold, best params, etc.)
        - avg_test_sharpe: aggregate out-of-sample Sharpe (net of costs)
        - oos_win_rate: aggregate OOS win rate
        - oos_return / oos_buy_hold / oos_alpha: aggregate OOS performance vs
          the buy-and-hold benchmark over the same test windows
        - overfit_ratio: avg(train_sharpe) / avg(test_sharpe) — >2 = likely overfit
        - best_params: most frequently selected parameter set
    """
    df = fetch_stock_data(ticker, period='5y')
    if df.empty:
        return {'error': f'No data for {ticker}'}
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    if param_grid is None:
        param_grid = PARAM_GRIDS.get(strategy, {})

    sig_fn, sig_col = _get_signal_fn(strategy)

    # Generate all parameter combinations
    if param_grid:
        keys = list(param_grid.keys())
        combos = [dict(zip(keys, vals)) for vals in product(*param_grid.values())]
    else:
        combos = [{}]

    # Create fold boundaries
    total_days = len(df)
    test_days = test_months * 21  # ~21 trading days per month
    min_train = max(252, total_days // 4)  # at least 1 year or 25% of data

    folds = []
    for fold_idx in range(n_splits):
        test_end_idx = total_days - fold_idx * test_days
        test_start_idx = test_end_idx - test_days
        train_end_idx = test_start_idx

        if train_end_idx < min_train or test_start_idx < 0 or test_end_idx <= test_start_idx:
            continue

        train_df = df.iloc[:train_end_idx]
        test_df  = df.iloc[test_start_idx:test_end_idx]

        if len(train_df) < min_train or len(test_df) < 20:
            continue

        # Sweep parameters on train set (net of costs, same as OOS eval)
        best_sharpe = -999
        best_params = {}
        for params in combos:
            sharpe, _ = _backtest_with_params(
                train_df, strategy, params, sig_fn, sig_col,
                commission_pct=commission_pct)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params

        # Evaluate best params on test set
        # Use full data up to test_end for signal warmup, then measure only test period
        warmup_df = df.iloc[:test_end_idx]
        try:
            warmup_sig = sig_fn(warmup_df.copy(), **best_params)
            if sig_col == 'ma_signal' and 'ma_signal' in warmup_sig.columns:
                warmup_sig['strategy_signal'] = warmup_sig['ma_signal']
                test_sig_col = 'strategy_signal'
            else:
                test_sig_col = sig_col
            test_slice = warmup_sig.iloc[test_start_idx:test_end_idx]
            test_result = _run_backtest_on_df(
                test_slice, 10_000.0, test_sig_col, commission_pct=commission_pct)
            test_sharpe = test_result.get('sharpe_ratio', 0)
            test_win_rate = test_result.get('win_rate_pct', 0)
            test_return = test_result.get('total_return_pct', 0)
            test_bh = test_result.get('buy_hold_return_pct', 0)
            test_alpha = test_result.get('alpha_pct', 0)
            test_trades = test_result.get('num_trades', 0)
        except Exception:
            test_sharpe = test_win_rate = test_return = 0
            test_bh = test_alpha = 0
            test_trades = 0

        folds.append({
            'fold': fold_idx + 1,
            'train_start': str(train_df.index[0].date()),
            'train_end':   str(train_df.index[-1].date()),
            'test_start':  str(test_df.index[0].date()),
            'test_end':    str(test_df.index[-1].date()),
            'best_params': best_params,
            'train_sharpe': round(best_sharpe, 3),
            'test_sharpe':  round(test_sharpe, 3),
            'test_win_rate': round(test_win_rate, 1),
            'test_return':  round(test_return, 2),
            'test_buy_hold': round(test_bh, 2),
            'test_alpha':   round(test_alpha, 2),
            'test_trades':  test_trades,
        })

    if not folds:
        return {'error': 'Insufficient data for walk-forward splits'}

    # Aggregate metrics
    train_sharpes = [f['train_sharpe'] for f in folds if f['train_sharpe'] > -999]
    test_sharpes  = [f['test_sharpe'] for f in folds]
    test_win_rates = [f['test_win_rate'] for f in folds if f['test_trades'] > 0]

    avg_train = np.mean(train_sharpes) if train_sharpes else 0
    avg_test  = np.mean(test_sharpes) if test_sharpes else 0
    overfit_ratio = (avg_train / avg_test) if avg_test != 0 else float('inf')

    # Aggregate OOS performance vs the buy-and-hold benchmark (net of costs).
    oos_return    = float(np.mean([f['test_return'] for f in folds]))
    oos_buy_hold  = float(np.mean([f['test_buy_hold'] for f in folds]))
    oos_alpha     = float(np.mean([f['test_alpha'] for f in folds]))

    # Most common best params
    from collections import Counter
    param_strs = [str(sorted(f['best_params'].items())) for f in folds]
    most_common = Counter(param_strs).most_common(1)
    if most_common:
        # Find the actual dict for the most common params
        target = most_common[0][0]
        for f in folds:
            if str(sorted(f['best_params'].items())) == target:
                robust_params = f['best_params']
                break
        else:
            robust_params = folds[0]['best_params']
    else:
        robust_params = {}

    return {
        'ticker': ticker,
        'strategy': strategy,
        'n_folds': len(folds),
        'folds': folds,
        'commission_pct':   commission_pct,
        'avg_train_sharpe': round(avg_train, 3),
        'avg_test_sharpe':  round(avg_test, 3),
        'oos_win_rate':     round(np.mean(test_win_rates), 1) if test_win_rates else 0,
        'oos_return':       round(oos_return, 2),
        'oos_buy_hold':     round(oos_buy_hold, 2),
        'oos_alpha':        round(oos_alpha, 2),
        'overfit_ratio':    round(overfit_ratio, 2),
        'robust_params':    robust_params,
    }


def summarize_scorecard(
    wf_result: dict,
    min_oos_sharpe: float = SCORECARD_MIN_OOS_SHARPE,
    min_oos_alpha: float = SCORECARD_MIN_OOS_ALPHA,
    max_overfit: float = SCORECARD_MAX_OVERFIT,
) -> dict:
    """
    Reduce a walk-forward result to a single pass/fail "is the alpha real?"
    scorecard. Pure function — no I/O — so the gate logic is unit-testable.

    The strategy PASSES only if, net of transaction costs and out-of-sample:
      1. it beats buy & hold (mean OOS alpha > `min_oos_alpha`),
      2. its OOS Sharpe clears `min_oos_sharpe`, and
      3. it is not overfit (train/test Sharpe ratio <= `max_overfit`).

    Returns the three checks, the headline OOS numbers, a boolean `passed`,
    and a human-readable `verdict` + list of failure `reasons`.
    """
    if 'error' in wf_result:
        return {'passed': False, 'verdict': 'NO DATA', 'reasons': [wf_result['error']]}

    oos_sharpe = wf_result.get('avg_test_sharpe', 0.0)
    oos_alpha  = wf_result.get('oos_alpha', 0.0)
    overfit    = wf_result.get('overfit_ratio', float('inf'))

    # An infinite/extreme overfit ratio just means avg_test_sharpe ~ 0; treat
    # the Sharpe gate as the binding constraint there rather than auto-failing
    # solely on the ratio.
    beats_benchmark = oos_alpha > min_oos_alpha
    clears_sharpe   = oos_sharpe >= min_oos_sharpe
    not_overfit     = overfit <= max_overfit

    reasons = []
    if not beats_benchmark:
        reasons.append(
            f"OOS alpha {oos_alpha:+.2f}% does not beat buy & hold "
            f"(need > {min_oos_alpha:.2f}%)")
    if not clears_sharpe:
        reasons.append(
            f"OOS Sharpe {oos_sharpe:.2f} below floor {min_oos_sharpe:.2f}")
    if not not_overfit:
        reasons.append(
            f"Overfit ratio {overfit:.2f}x exceeds {max_overfit:.2f}x "
            "(train >> test)")

    passed = beats_benchmark and clears_sharpe and not_overfit
    return {
        'ticker': wf_result.get('ticker'),
        'strategy': wf_result.get('strategy'),
        'passed': passed,
        'verdict': 'ALPHA CONFIRMED' if passed else 'NOT PROVEN',
        'checks': {
            'beats_benchmark': beats_benchmark,
            'clears_sharpe': clears_sharpe,
            'not_overfit': not_overfit,
        },
        'oos_sharpe': round(oos_sharpe, 3),
        'oos_return': wf_result.get('oos_return', 0.0),
        'oos_buy_hold': wf_result.get('oos_buy_hold', 0.0),
        'oos_alpha': round(oos_alpha, 2),
        'overfit_ratio': overfit,
        'oos_win_rate': wf_result.get('oos_win_rate', 0.0),
        'n_folds': wf_result.get('n_folds', 0),
        'commission_pct': wf_result.get('commission_pct', 0.0),
        'reasons': reasons,
    }


def build_scorecard(
    ticker: str = 'SPY',
    strategy: str = 'Fractal',
    commission_pct: float = DEFAULT_COMMISSION_PCT,
    **wf_kwargs,
) -> dict:
    """
    Run an out-of-sample walk-forward (net of costs) and reduce it to the
    alpha scorecard. This is the "backtested proof" entry point: it answers,
    for a given ticker/strategy, whether the edge survives costs and holds up
    out-of-sample against buy & hold.
    """
    wf = walk_forward_test(
        ticker, strategy, commission_pct=commission_pct, **wf_kwargs)
    card = summarize_scorecard(wf)
    card['walk_forward'] = wf
    return card


def print_scorecard(card: dict) -> None:
    """Pretty-print the alpha scorecard."""
    w = 60
    print(f"\n{'='*w}")
    print(f"  ALPHA SCORECARD — {card.get('ticker')} / {card.get('strategy')}")
    print(f"{'='*w}")
    if card.get('verdict') == 'NO DATA':
        print(f"  {card['reasons'][0]}")
        print(f"{'='*w}")
        return
    cost_bps = card.get('commission_pct', 0.0) * 10_000
    print(f"  Verdict          :  {card['verdict']}")
    print(f"  Costs (per side) :  {cost_bps:.1f} bps")
    print(f"  OOS folds        :  {card.get('n_folds')}")
    print(f"{'─'*w}")
    print(f"  OOS Return       :  {card.get('oos_return', 0):>+8.2f}%")
    print(f"  Buy & Hold       :  {card.get('oos_buy_hold', 0):>+8.2f}%")
    print(f"  OOS Alpha        :  {card.get('oos_alpha', 0):>+8.2f}%   "
          f"[{'PASS' if card['checks']['beats_benchmark'] else 'FAIL'}]")
    print(f"  OOS Sharpe       :  {card.get('oos_sharpe', 0):>+8.2f}    "
          f"[{'PASS' if card['checks']['clears_sharpe'] else 'FAIL'}]")
    print(f"  Overfit Ratio    :  {card.get('overfit_ratio', 0):>8.2f}x   "
          f"[{'PASS' if card['checks']['not_overfit'] else 'FAIL'}]")
    print(f"  OOS Win Rate     :  {card.get('oos_win_rate', 0):>8.1f}%")
    if card.get('reasons'):
        print(f"{'─'*w}")
        print("  Why not proven:")
        for r in card['reasons']:
            print(f"    - {r}")
    print(f"{'='*w}")


def print_walk_forward_results(r: dict) -> None:
    """Pretty-print walk-forward results."""
    if 'error' in r:
        print(f"Error: {r['error']}")
        return

    w = 60
    print(f"\n{'='*w}")
    print(f"  {r['ticker']} — {r['strategy']} Walk-Forward ({r['n_folds']} folds)")
    print(f"{'='*w}")
    print(f"  Avg Train Sharpe  : {r['avg_train_sharpe']:>+8.3f}")
    print(f"  Avg Test Sharpe   : {r['avg_test_sharpe']:>+8.3f}")
    print(f"  OOS Win Rate      : {r['oos_win_rate']:>8.1f}%")
    print(f"  Overfit Ratio     : {r['overfit_ratio']:>8.2f}x")
    print(f"  Robust Params     : {r['robust_params']}")
    print(f"{'─'*w}")
    print(f"  {'Fold':<5} {'Train Sharpe':>13} {'Test Sharpe':>12} {'Test WR':>8} {'Test Ret':>9}")
    print(f"  {'─'*48}")
    for f in r['folds']:
        print(f"  {f['fold']:<5} {f['train_sharpe']:>+13.3f} {f['test_sharpe']:>+12.3f} "
              f"{f['test_win_rate']:>7.1f}% {f['test_return']:>+8.2f}%")
    print(f"{'='*w}")


if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'SPY'
    strategy = sys.argv[2] if len(sys.argv) > 2 else 'Fractal'
    card = build_scorecard(ticker, strategy)
    print_walk_forward_results(card.get('walk_forward', {}))
    print_scorecard(card)
