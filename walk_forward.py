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

from config import DATA_PERIOD
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


def _backtest_with_params(df, strategy, params, sig_fn, sig_col):
    """Run a single backtest with given params, return Sharpe and results."""
    try:
        df_sig = sig_fn(df.copy(), **params)
        if sig_col == 'ma_signal' and 'ma_signal' in df_sig.columns:
            df_sig['strategy_signal'] = df_sig['ma_signal']
            sig_col_use = 'strategy_signal'
        else:
            sig_col_use = sig_col
        result = _run_backtest_on_df(df_sig, 10_000.0, sig_col_use)
        return result.get('sharpe_ratio', -999), result
    except Exception:
        return -999, {}


def walk_forward_test(
    ticker: str,
    strategy: str = 'Fractal',
    n_splits: int = 8,
    test_months: int = 3,
    param_grid: dict = None,
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

    Returns
    -------
    dict with:
        - folds: list of per-fold results (train/test Sharpe, best params, etc.)
        - oos_sharpe: aggregate out-of-sample Sharpe
        - oos_win_rate: aggregate OOS win rate
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

        # Sweep parameters on train set
        best_sharpe = -999
        best_params = {}
        for params in combos:
            sharpe, _ = _backtest_with_params(train_df, strategy, params, sig_fn, sig_col)
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
            test_result = _run_backtest_on_df(test_slice, 10_000.0, test_sig_col)
            test_sharpe = test_result.get('sharpe_ratio', 0)
            test_win_rate = test_result.get('win_rate_pct', 0)
            test_return = test_result.get('total_return_pct', 0)
            test_trades = test_result.get('num_trades', 0)
        except Exception:
            test_sharpe = test_win_rate = test_return = 0
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
        'avg_train_sharpe': round(avg_train, 3),
        'avg_test_sharpe':  round(avg_test, 3),
        'oos_win_rate':     round(np.mean(test_win_rates), 1) if test_win_rates else 0,
        'overfit_ratio':    round(overfit_ratio, 2),
        'robust_params':    robust_params,
    }


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
    result = walk_forward_test(ticker, strategy)
    print_walk_forward_results(result)
