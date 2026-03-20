"""
Backtester — MA Crossover (original) + Generic multi-strategy runner.

Usage:
    python backtest.py AAPL NVDA SPY
    python backtest.py          # defaults to SPY, QQQ, AAPL
"""

import sys

import numpy as np
import pandas as pd

from config import DATA_PERIOD, MA_LONG, MA_SHORT
from data_fetcher import fetch_stock_data
from strategies.ma_crossover import generate_signals


# ---------------------------------------------------------------------------
# Generic backtest engine — works with any strategy that produces a
# 'strategy_signal' column (+1 buy, -1 sell, 0 hold).
# ---------------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range for volatility-based position sizing."""
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _run_backtest_on_df(
    df: pd.DataFrame,
    initial_capital: float,
    signal_col: str,
    commission_pct: float = 0.0,
    position_mode: str = 'full',
    risk_pct: float = 0.10,
    atr_multiplier: float = 2.0,
) -> dict:
    """
    Core simulation loop shared by all strategies.
    Long-only: enters on +1, exits on -1.

    Parameters
    ----------
    commission_pct : float
        Round-trip cost per trade as fraction (0.001 = 10 bps).
    position_mode : str
        'full' — all-in (legacy), 'fixed_frac' — risk_pct of capital,
        'volatility' — size inversely to ATR.
    risk_pct : float
        Fraction of capital to risk per trade (used by fixed_frac and volatility modes).
    atr_multiplier : float
        ATR multiplier for volatility sizing stop distance.
    """
    # Drop rows with NaN Close to prevent NaN propagation
    df = df.dropna(subset=['Close']).copy()
    if df.empty:
        return {'total_return_pct': 0, 'buy_hold_return_pct': 0, 'alpha_pct': 0,
                'max_drawdown_pct': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
                'calmar_ratio': 0, 'profit_factor': 0, 'num_trades': 0,
                'win_rate_pct': 0, 'avg_trade_duration': 0, 'trades': [],
                'equity_dates': [], 'equity_values': [], 'bh_values': [],
                'initial_capital': initial_capital, 'final_capital': initial_capital,
                'max_consec_wins': 0, 'max_consec_losses': 0,
                'commission_pct': commission_pct, 'position_mode': position_mode}

    # Pre-compute ATR if needed
    atr = _compute_atr(df) if position_mode == 'volatility' else None

    capital     = initial_capital
    shares      = 0.0
    position    = False
    entry_price = 0.0
    entry_date  = None
    trades      = []

    for idx, row in df.iterrows():
        sig = row.get(signal_col, 0)
        if sig == 1 and not position:
            # Position sizing
            if position_mode == 'fixed_frac':
                alloc = capital * risk_pct
            elif position_mode == 'volatility' and atr is not None:
                atr_val = atr.get(idx, np.nan) if isinstance(atr, pd.Series) else np.nan
                if not np.isnan(atr_val) and atr_val > 0:
                    stop_dist = atr_val * atr_multiplier
                    alloc = min(capital, (capital * risk_pct) / (stop_dist / row['Close']))
                else:
                    alloc = capital * risk_pct
            else:  # 'full'
                alloc = capital

            # Deduct buy commission
            alloc_after_cost = alloc * (1 - commission_pct)
            shares      = alloc_after_cost / row['Close']
            capital    -= alloc  # reserve capital used
            entry_price = row['Close']
            entry_date  = idx
            position    = True
            trades.append({'type': 'buy', 'date': idx,
                           'price': round(float(row['Close']), 2),
                           'shares': round(shares, 4)})

        elif sig == -1 and position:
            proceeds = shares * row['Close'] * (1 - commission_pct)
            capital += proceeds
            pnl_pct = (row['Close'] - entry_price) / entry_price * 100
            duration = (idx - entry_date).days if entry_date is not None else 0
            trades.append({'type': 'sell', 'date': idx,
                           'price': round(float(row['Close']), 2),
                           'shares': round(shares, 4),
                           'pnl_pct': round(pnl_pct, 2),
                           'duration_days': duration})
            shares   = 0.0
            position = False

    if position:
        capital += shares * df['Close'].iloc[-1] * (1 - commission_pct)

    # Equity curve (date-aligned)
    eq_dates, eq_values, bh_values = [], [], []
    first_close = float(df['Close'].iloc[0])
    cap_eq, pos_eq, sh_eq, alloc_eq = initial_capital, False, 0.0, 0.0

    for date, row in df.iterrows():
        sig = row.get(signal_col, 0)
        if sig == 1 and not pos_eq:
            if position_mode == 'fixed_frac':
                a = cap_eq * risk_pct
            elif position_mode == 'volatility' and atr is not None:
                atr_val = atr.get(date, np.nan) if isinstance(atr, pd.Series) else np.nan
                if not np.isnan(atr_val) and atr_val > 0:
                    a = min(cap_eq, (cap_eq * risk_pct) / (atr_val * atr_multiplier / row['Close']))
                else:
                    a = cap_eq * risk_pct
            else:
                a = cap_eq
            alloc_eq = a
            sh_eq  = (a * (1 - commission_pct)) / row['Close']
            cap_eq -= a
            pos_eq = True
        elif sig == -1 and pos_eq:
            cap_eq += sh_eq * row['Close'] * (1 - commission_pct)
            sh_eq  = 0.0
            pos_eq = False
        val = cap_eq if not pos_eq else cap_eq + sh_eq * float(row['Close'])
        eq_dates.append(date)
        eq_values.append(val)
        bh_values.append(initial_capital * float(row['Close']) / first_close)

    eq_series = pd.Series(eq_values, dtype=float)
    peak      = eq_series.cummax()
    max_dd    = float(((eq_series - peak) / peak).min() * 100)
    bh_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

    # --- Risk metrics from equity curve ---
    eq_arr = np.array(eq_values, dtype=float)
    daily_returns = np.diff(eq_arr) / eq_arr[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    if len(daily_returns) > 1:
        mean_ret = np.mean(daily_returns)
        std_ret  = np.std(daily_returns, ddof=1)
        sharpe   = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        downside = daily_returns[daily_returns < 0]
        down_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
        sortino  = (mean_ret / down_std * np.sqrt(252)) if down_std > 0 else 0.0

        total_days = (df.index[-1] - df.index[0]).days
        ann_return = ((eq_arr[-1] / eq_arr[0]) ** (365.25 / max(total_days, 1)) - 1) * 100
        calmar     = (ann_return / abs(max_dd)) if max_dd != 0 else 0.0
    else:
        sharpe = sortino = calmar = ann_return = 0.0

    # --- Trade-level metrics ---
    sell_trades = [t for t in trades if t['type'] == 'sell']
    wins        = [t for t in sell_trades if t.get('pnl_pct', 0) > 0]
    losses      = [t for t in sell_trades if t.get('pnl_pct', 0) <= 0]
    win_rate    = len(wins) / len(sell_trades) * 100 if sell_trades else 0.0
    total_ret   = (capital - initial_capital) / initial_capital * 100

    gross_wins   = sum(t['pnl_pct'] for t in wins) if wins else 0.0
    gross_losses = abs(sum(t['pnl_pct'] for t in losses)) if losses else 0.0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else float('inf') if gross_wins > 0 else 0.0

    durations = [t.get('duration_days', 0) for t in sell_trades]
    avg_duration = np.mean(durations) if durations else 0.0

    # Max consecutive wins/losses
    max_consec_wins = max_consec_losses = consec_w = consec_l = 0
    for t in sell_trades:
        if t.get('pnl_pct', 0) > 0:
            consec_w += 1
            consec_l = 0
            max_consec_wins = max(max_consec_wins, consec_w)
        else:
            consec_l += 1
            consec_w = 0
            max_consec_losses = max(max_consec_losses, consec_l)

    return {
        'initial_capital':      initial_capital,
        'final_capital':        round(capital, 2),
        'total_return_pct':     round(total_ret, 2),
        'annualized_return_pct': round(ann_return, 2) if len(daily_returns) > 1 else None,
        'buy_hold_return_pct':  round(bh_return, 2),
        'alpha_pct':            round(total_ret - bh_return, 2),
        'max_drawdown_pct':     round(max_dd, 2),
        'sharpe_ratio':         round(sharpe, 3),
        'sortino_ratio':        round(sortino, 3),
        'calmar_ratio':         round(calmar, 3),
        'profit_factor':        round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
        'num_trades':           len(sell_trades),
        'win_rate_pct':         round(win_rate, 2),
        'avg_trade_duration':   round(avg_duration, 1),
        'max_consec_wins':      max_consec_wins,
        'max_consec_losses':    max_consec_losses,
        'commission_pct':       commission_pct,
        'position_mode':        position_mode,
        'trades':               trades,
        'equity_dates':         eq_dates,
        'equity_values':        eq_values,
        'bh_values':            bh_values,
    }


# ---------------------------------------------------------------------------
# Advanced analytics
# ---------------------------------------------------------------------------

def compute_advanced_metrics(result: dict) -> dict:
    """Compute institutional-grade metrics from a backtest result."""
    sell_trades = [t for t in result.get('trades', []) if t['type'] == 'sell']
    wins = [t for t in sell_trades if t.get('pnl_pct', 0) > 0]
    losses = [t for t in sell_trades if t.get('pnl_pct', 0) <= 0]

    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t['pnl_pct'] for t in losses])) if losses else 0
    win_rate = len(wins) / len(sell_trades) if sell_trades else 0

    # Expectancy: expected profit per trade
    expectancy = avg_win * win_rate - avg_loss * (1 - win_rate)

    # Payoff ratio
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # Half-Kelly fraction
    kelly = 0.0
    if payoff_ratio > 0:
        kelly = max(0, 0.5 * (win_rate - (1 - win_rate) / payoff_ratio))

    # Recovery factor
    max_dd = abs(result.get('max_drawdown_pct', 0))
    total_ret = abs(result.get('total_return_pct', 0))
    recovery_factor = total_ret / max_dd if max_dd > 0 else 0

    # Max drawdown duration (days)
    eq = np.array(result.get('equity_values', []), dtype=float)
    max_dd_duration = 0
    if len(eq) > 1:
        peak = np.maximum.accumulate(eq)
        in_dd = eq < peak
        current_streak = 0
        for v in in_dd:
            if v:
                current_streak += 1
                max_dd_duration = max(max_dd_duration, current_streak)
            else:
                current_streak = 0

    return {
        'expectancy': round(expectancy, 2),
        'recovery_factor': round(recovery_factor, 2),
        'max_dd_duration_days': max_dd_duration,
        'kelly_fraction': round(kelly, 3),
        'payoff_ratio': round(payoff_ratio, 2),
        'avg_win_pct': round(avg_win, 2),
        'avg_loss_pct': round(avg_loss, 2),
    }


def monte_carlo_simulation(
    trades: list,
    n_simulations: int = 1000,
    initial_capital: float = 10000,
) -> dict:
    """
    Monte Carlo: reshuffle trade P&L order to build confidence intervals.
    Fast: ~50ms for 1000 sims using numpy vectorization.
    """
    pnl_pcts = np.array([t['pnl_pct'] / 100 for t in trades if t['type'] == 'sell'])
    if len(pnl_pcts) < 3:
        return {}

    final_values = np.zeros(n_simulations)
    max_dds = np.zeros(n_simulations)

    for i in range(n_simulations):
        shuffled = np.random.permutation(pnl_pcts)
        equity = initial_capital * np.cumprod(1 + shuffled)
        final_values[i] = equity[-1]
        peak = np.maximum.accumulate(equity)
        max_dds[i] = ((equity - peak) / peak).min() * 100

    return {
        'median_return': round((np.median(final_values) / initial_capital - 1) * 100, 2),
        'p5_return': round((np.percentile(final_values, 5) / initial_capital - 1) * 100, 2),
        'p95_return': round((np.percentile(final_values, 95) / initial_capital - 1) * 100, 2),
        'median_max_dd': round(np.median(max_dds), 2),
        'p95_max_dd': round(np.percentile(max_dds, 5), 2),  # worst-case DD (5th percentile = most negative)
        'final_values': final_values.tolist(),
    }


# Strategy registry — maps strategy name → (module.generate_signals, signal_col, default_kwargs)
def _get_strategy_fn(strategy: str):
    if strategy == 'MA Crossover':
        from strategies.ma_crossover import generate_signals as fn
        return fn, 'ma_signal', {}
    elif strategy == 'MACD + RSI':
        from strategies.macd_rsi import generate_signals as fn
        return fn, 'strategy_signal', {}
    elif strategy == 'BB Squeeze':
        from strategies.bb_squeeze import generate_signals as fn
        return fn, 'strategy_signal', {}
    elif strategy == 'TSMOM':
        from strategies.tsmom import generate_signals as fn
        return fn, 'strategy_signal', {}
    elif strategy == 'Turtle':
        from strategies.turtle import generate_signals as fn
        return fn, 'strategy_signal', {}
    elif strategy == 'Fractal':
        from strategies.fractal_signals import generate_signals as fn
        return fn, 'strategy_signal', {}
    elif strategy == 'Ensemble':
        from strategies.ensemble import generate_signals as fn
        return fn, 'strategy_signal', {}
    else:
        raise ValueError(f'Unknown strategy: {strategy}')


def backtest_strategy(
    ticker: str,
    strategy: str = 'MA Crossover',
    initial_capital: float = 10_000.0,
    start_date=None,
    end_date=None,
    strategy_kwargs: dict = None,
    commission_pct: float = 0.0,
    position_mode: str = 'full',
    risk_pct: float = 0.10,
) -> dict:
    """
    Universal backtest entry point. Fetches data, applies any registered
    strategy, and returns a standardised results dict.

    strategy: one of 'MA Crossover', 'MACD + RSI', 'BB Squeeze', 'TSMOM', 'Turtle',
              'Fractal', 'Ensemble'
    strategy_kwargs: override default indicator parameters for the chosen strategy.
    commission_pct: transaction cost per trade as fraction (0.001 = 10 bps).
    position_mode: 'full', 'fixed_frac', or 'volatility'.
    risk_pct: fraction of capital to risk per trade (for fixed_frac / volatility).
    """
    period = '5y' if (start_date or end_date) else DATA_PERIOD
    df = fetch_stock_data(ticker, period=period)
    if df.empty:
        return {}
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    sig_fn, sig_col, defaults = _get_strategy_fn(strategy)
    params = {**defaults, **(strategy_kwargs or {})}
    df = sig_fn(df, **params)

    # MA Crossover uses 'ma_signal'; normalise to +1/-1 for the engine
    if strategy == 'MA Crossover' and sig_col == 'ma_signal':
        df['strategy_signal'] = df['ma_signal']
        sig_col = 'strategy_signal'

    min_bars = 252 if strategy in ('TSMOM', 'MA Crossover') else 60
    if len(df) < min_bars:
        print(f'  [{ticker}] Insufficient data for {strategy} (need {min_bars} bars).')
        return {}

    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date)]
    if df.empty:
        return {}

    result = _run_backtest_on_df(
        df, initial_capital, sig_col,
        commission_pct=commission_pct,
        position_mode=position_mode,
        risk_pct=risk_pct,
    )
    result['ticker']   = ticker
    result['strategy'] = strategy
    return result


def backtest_ma_crossover(
    ticker: str,
    initial_capital: float = 10_000.0,
    short: int = MA_SHORT,
    long: int = MA_LONG,
    start_date=None,
    end_date=None,
) -> dict:
    """
    Simulate trading the MA crossover strategy.
    Enters on golden cross, exits on death cross.

    start_date / end_date: optional date strings or date objects (e.g. '2022-01-01').
    When provided, fetches 5Y of data so SMA has a full 200-bar warmup before
    the range starts, then filters trades to the specified window.
    """
    period = '5y' if (start_date or end_date) else DATA_PERIOD
    df = fetch_stock_data(ticker, period=period)
    if df.empty or len(df) < long:
        print(f"  [{ticker}] Insufficient data (need {long} bars, got {len(df)})")
        return {}

    # Normalise timezone so date comparisons work cleanly
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = generate_signals(df, short, long)

    # Filter to date range AFTER computing signals (SMA needs the warmup data)
    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date)]

    if df.empty:
        return {}

    capital = initial_capital
    shares = 0.0
    position = False
    entry_price = 0.0
    trades = []

    for idx, row in df.iterrows():
        if row['ma_signal'] == 1 and not position:
            shares = capital / row['Close']
            entry_price = row['Close']
            position = True
            trades.append({
                'type': 'buy',
                'date': idx,
                'price': round(float(row['Close']), 2),
                'shares': round(shares, 4),
            })

        elif row['ma_signal'] == -1 and position:
            capital = shares * row['Close']
            pnl_pct = (row['Close'] - entry_price) / entry_price * 100
            trades.append({
                'type': 'sell',
                'date': idx,
                'price': round(float(row['Close']), 2),
                'shares': round(shares, 4),
                'pnl_pct': round(pnl_pct, 2),
            })
            shares = 0.0
            position = False

    # Close any open position at last price
    if position:
        capital = shares * df['Close'].iloc[-1]

    # Equity curve (date-aligned, for charting and drawdown)
    eq_dates = []
    eq_values = []
    bh_values = []
    first_close = float(df['Close'].iloc[0])
    cap_eq = initial_capital
    pos_eq = False
    sh_eq = 0.0

    for date, row in df.iterrows():
        if row['ma_signal'] == 1 and not pos_eq:
            sh_eq = cap_eq / row['Close']
            pos_eq = True
        elif row['ma_signal'] == -1 and pos_eq:
            cap_eq = sh_eq * row['Close']
            sh_eq = 0.0
            pos_eq = False
        val = cap_eq if not pos_eq else sh_eq * float(row['Close'])
        eq_dates.append(date)
        eq_values.append(val)
        bh_values.append(initial_capital * float(row['Close']) / first_close)

    equity_series = pd.Series(eq_values)
    peak = equity_series.cummax()
    max_drawdown = float(((equity_series - peak) / peak).min() * 100)

    bh_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100

    sell_trades = [t for t in trades if t['type'] == 'sell']
    wins = [t for t in sell_trades if t.get('pnl_pct', 0) > 0]
    win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0.0
    total_return = (capital - initial_capital) / initial_capital * 100

    return {
        'ticker': ticker,
        'initial_capital': initial_capital,
        'final_capital': round(capital, 2),
        'total_return_pct': round(total_return, 2),
        'buy_hold_return_pct': round(bh_return, 2),
        'alpha_pct': round(total_return - bh_return, 2),
        'max_drawdown_pct': round(max_drawdown, 2),
        'num_trades': len(sell_trades),
        'win_rate_pct': round(win_rate, 2),
        'trades': trades,
        'equity_dates': eq_dates,
        'equity_values': eq_values,
        'bh_values': bh_values,
    }


def print_backtest_results(r: dict) -> None:
    w = 52
    print(f"\n{'='*w}")
    print(f"  {r['ticker']} — MA({MA_SHORT}/{MA_LONG}) Crossover Backtest")
    print(f"{'='*w}")
    print(f"  Initial Capital :  ${r['initial_capital']:>10,.2f}")
    print(f"  Final Capital   :  ${r['final_capital']:>10,.2f}")
    print(f"  Strategy Return :  {r['total_return_pct']:>+10.2f}%")
    print(f"  Buy & Hold      :  {r['buy_hold_return_pct']:>+10.2f}%")
    print(f"  Alpha           :  {r['alpha_pct']:>+10.2f}%")
    print(f"  Max Drawdown    :  {r['max_drawdown_pct']:>10.2f}%")
    print(f"  Trades (closed) :  {r['num_trades']:>10}")
    print(f"  Win Rate        :  {r['win_rate_pct']:>10.1f}%")
    print(f"{'─'*w}")
    if r['trades']:
        print("  Trade Log:")
        for t in r['trades']:
            date_str = t['date'].strftime('%Y-%m-%d') if hasattr(t['date'], 'strftime') else str(t['date'])[:10]
            if t['type'] == 'buy':
                print(f"    BUY   {date_str}  @ ${t['price']:.2f}")
            else:
                sign = '+' if t['pnl_pct'] >= 0 else ''
                print(f"    SELL  {date_str}  @ ${t['price']:.2f}  ({sign}{t['pnl_pct']:.2f}%)")
    print(f"{'='*w}")


if __name__ == '__main__':
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ['SPY', 'AAPL', 'NVDA']
    for ticker in tickers:
        results = backtest_ma_crossover(ticker.upper())
        if results:
            print_backtest_results(results)
