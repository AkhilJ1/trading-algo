"""
Ensemble Meta-Strategy — Signal Voting Across All Strategies
-------------------------------------------------------------
Runs multiple independent strategies on the same DataFrame,
counts agreement, and generates consensus buy/sell signals.

Entry: when >= threshold strategies agree on buy
Exit:  when >= threshold strategies agree on sell
"""

import numpy as np
import pandas as pd


# Strategy runners: each returns (signal_series, name)
# We import lazily inside the function to avoid circular imports.

_STRATEGY_REGISTRY = [
    ('MA Crossover',  'ma_signal'),
    ('MACD + RSI',    'strategy_signal'),
    ('BB Squeeze',    'strategy_signal'),
    ('TSMOM',         'strategy_signal'),
    ('Turtle',        'strategy_signal'),
    ('Fractal',       'strategy_signal'),
]


def _run_single_strategy(df: pd.DataFrame, name: str, sig_col: str) -> pd.Series:
    """Run one strategy and return its signal series, or zeros on failure."""
    try:
        if name == 'MA Crossover':
            from strategies.ma_crossover import generate_signals
            out = generate_signals(df.copy())
            return out[sig_col]
        elif name == 'MACD + RSI':
            from strategies.macd_rsi import generate_signals
            out = generate_signals(df.copy())
            return out[sig_col]
        elif name == 'BB Squeeze':
            from strategies.bb_squeeze import generate_signals
            out = generate_signals(df.copy())
            return out[sig_col]
        elif name == 'TSMOM':
            from strategies.tsmom import generate_signals
            out = generate_signals(df.copy())
            return out[sig_col]
        elif name == 'Turtle':
            from strategies.turtle import generate_signals
            out = generate_signals(df.copy())
            return out[sig_col]
        elif name == 'Fractal':
            from strategies.fractal_signals import generate_signals
            out = generate_signals(df.copy())
            return out[sig_col]
    except Exception:
        pass
    return pd.Series(0, index=df.index, dtype=int)


def generate_signals(
    df: pd.DataFrame,
    threshold: int = 3,
    strategies: list = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate ensemble consensus signals via majority voting.

    Parameters
    ----------
    df : DataFrame with OHLCV columns
    threshold : int, minimum strategies that must agree for a signal
    strategies : list of strategy names to include (default: all 6)

    Returns
    -------
    DataFrame with added columns:
        - ensemble_votes_buy: count of strategies signaling buy at each bar
        - ensemble_votes_sell: count of strategies signaling sell at each bar
        - ensemble_confidence: agreement ratio (0-1)
        - strategy_signal: +1 buy / -1 sell / 0 hold (consensus)
    """
    df = df.copy()

    if strategies is None:
        registry = _STRATEGY_REGISTRY
    else:
        registry = [(n, c) for n, c in _STRATEGY_REGISTRY if n in strategies]

    # Collect individual strategy signals
    # Each strategy produces +1 (buy), -1 (sell), 0 (hold)
    # We need to track position state to know when each strategy is "in" vs "out"
    all_positions = {}
    for name, sig_col in registry:
        sig = _run_single_strategy(df, name, sig_col)
        # Convert signal to position state: 1 = in position, 0 = out
        pos = np.zeros(len(sig), dtype=int)
        in_pos = False
        for i in range(len(sig)):
            val = sig.iloc[i] if hasattr(sig, 'iloc') else sig[i]
            if val == 1 and not in_pos:
                in_pos = True
            elif val == -1 and in_pos:
                in_pos = False
            pos[i] = 1 if in_pos else 0
        all_positions[name] = pos

    n = len(df)
    n_strats = len(registry)

    # At each bar, count how many strategies are in a position (bullish)
    bull_count = np.zeros(n, dtype=int)
    for pos_arr in all_positions.values():
        bull_count += pos_arr

    # Generate consensus signals
    signals = np.zeros(n, dtype=int)
    in_position = False
    for i in range(n):
        if not in_position:
            if bull_count[i] >= threshold:
                signals[i] = 1
                in_position = True
        else:
            # Exit when fewer than (threshold - 1) are still bullish
            # This creates slight hysteresis to avoid whipsaws
            exit_threshold = max(1, threshold - 1)
            if bull_count[i] < exit_threshold:
                signals[i] = -1
                in_position = False

    df['ensemble_votes_buy'] = bull_count
    df['ensemble_votes_sell'] = n_strats - bull_count
    df['ensemble_confidence'] = bull_count / max(n_strats, 1)
    df['strategy_signal'] = signals

    return df
