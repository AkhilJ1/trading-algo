"""
Dynamic Candidate Screener
---------------------------
Scans the S&P 500 weekly charts to surface the highest-conviction
mean-reversion setups based on:

  Phase 1 — batch weekly download + RSI pre-filter (fast, ~15-25s)
  Phase 2 — full RSI/BB signal + MA crossover backtest on survivors (~30-90s)

Ranking: composite score = signal_strength * 0.6 + backtest_win_rate * 0.4

Usage:
    from screener import discover_candidates
    results = discover_candidates(progress_callback=lambda pct, msg: ...)
"""

import pandas as pd
import yfinance as yf

from config import WATCHLIST
from strategies.rsi_bollinger import add_indicators, get_buy_signal
from backtest import backtest_ma_crossover

# Pre-filter threshold — wider than the strict RSI_OVERSOLD=30 so we catch
# stocks that are approaching oversold but haven't crossed yet.
_RSI_PREFILTER = 38

# Module-level cache so we don't hit Wikipedia on every button click.
_universe_cache: list = []


def get_sp500_tickers() -> list:
    """
    Fetch the S&P 500 component list from Wikipedia.
    Falls back to the local WATCHLIST if the request fails.
    """
    global _universe_cache
    if _universe_cache:
        return _universe_cache
    try:
        df = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        )[0]
        tickers = [str(t).replace('.', '-') for t in df['Symbol'].tolist()]
        _universe_cache = tickers
        return tickers
    except Exception:
        return list(WATCHLIST)


def _normalise_df(raw, ticker: str, n_tickers: int) -> pd.DataFrame:
    """
    Extract and normalise a per-ticker DataFrame from a yf.download batch result.
    Returns an empty DataFrame on any failure.
    """
    try:
        df = raw.copy() if n_tickers == 1 else raw[ticker].copy()
        df = df.dropna(how='all')
        df.columns = [str(c).strip().title() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def discover_candidates(
    progress_callback=None,
    rsi_prefilter: float = _RSI_PREFILTER,
    top_n: int = 15,
) -> list:
    """
    Two-phase dynamic screen of the S&P 500 on weekly charts.

    progress_callback(pct: float, message: str) is called with values 0.0–1.0
    so callers (e.g. Streamlit) can render a live progress bar.

    Returns a list of result dicts sorted by composite score (descending),
    capped at top_n entries.
    """
    def _cb(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    # ── Phase 1: universe + batch download ────────────────────────────────
    _cb(0.00, 'Fetching S&P 500 ticker list from Wikipedia…')
    universe = get_sp500_tickers()

    _cb(0.03, f'Batch-downloading weekly charts for {len(universe)} stocks…')
    try:
        raw = yf.download(
            universe,
            period='1y',
            interval='1wk',
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception:
        return []

    # ── Phase 1: RSI pre-filter ───────────────────────────────────────────
    _cb(0.35, f'Applying weekly RSI < {rsi_prefilter:.0f} pre-filter…')
    survivors = []
    for ticker in universe:
        df = _normalise_df(raw, ticker, len(universe))
        if df.empty or len(df) < 20 or 'Close' not in df.columns:
            continue
        try:
            df_ind = add_indicators(df)
            rsi_val = float(df_ind['RSI'].iloc[-1])
            if pd.isna(rsi_val) or rsi_val >= rsi_prefilter:
                continue
            survivors.append({'ticker': ticker, 'df': df_ind, 'rsi': rsi_val})
        except Exception:
            continue

    if not survivors:
        _cb(1.0, 'No candidates found meeting the weekly RSI threshold.')
        return []

    _cb(0.40, f'{len(survivors)} candidates passed pre-filter — running full analysis…')

    # ── Phase 2: full signal + backtest ──────────────────────────────────
    scored = []
    phase2_range = 0.95 - 0.40
    step = phase2_range / max(len(survivors), 1)

    for i, s in enumerate(survivors):
        ticker = s['ticker']
        _cb(0.40 + i * step, f'Analysing {ticker} ({i+1}/{len(survivors)})…')
        try:
            signal = get_buy_signal(s['df'])
            bt = backtest_ma_crossover(ticker, initial_capital=10_000)
            win_rate  = bt.get('win_rate_pct', 0.0)  if bt else 0.0
            num_trades = bt.get('num_trades', 0)      if bt else 0
            bt_return  = bt.get('total_return_pct', 0.0) if bt else 0.0

            # Composite score (both components already 0–100 range)
            score = signal['strength'] * 0.6 + win_rate * 0.4

            scored.append({
                'ticker':     ticker,
                'score':      round(score, 1),
                'price':      signal['close'],
                'rsi':        signal['rsi'],
                'strength':   signal['strength'],
                'buy_signal': signal['buy_signal'],
                'rsi_oversold': signal['rsi_oversold'],
                'wick_touches': signal['wick_touches'],
                'bb_lower':   signal['bb_lower'],
                'bb_mid':     signal['bb_mid'],
                'bb_upper':   signal['bb_upper'],
                'win_rate':   round(win_rate, 1),
                'num_trades': num_trades,
                'bt_return':  round(bt_return, 2),
            })
        except Exception:
            continue

    _cb(1.0, f'Done — {len(scored)} candidates scored.')
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:top_n]
