"""
Daily Trade Scanner
--------------------
Scans the watchlist for stocks meeting the RSI + Bollinger Band buy criteria:
  - RSI(14) < 30 on the 1-year daily chart  (oversold)
  - At least one of the last 5 candles has a wick (low) at or below the lower BB

Also reports the current MA(50/200) trend context for each signal.

Usage:
    python scanner.py                        # scan full watchlist from config
    python scanner.py AAPL TSLA NVDA        # scan custom tickers
"""

import sys
from datetime import datetime
from typing import Optional

from config import (
    BB_WICK_LOOKBACK,
    MA_LONG,
    MA_SHORT,
    RSI_OVERSOLD,
    WATCHLIST,
)
from data_fetcher import fetch_stock_data
from strategies.ma_crossover import current_signal as ma_current_signal
from strategies.rsi_bollinger import get_buy_signal


def scan_ticker(ticker: str) -> Optional[dict]:
    """
    Fetch data and evaluate both strategies for one ticker.
    Returns None if data is unavailable or insufficient.
    """
    df = fetch_stock_data(ticker)
    if df.empty or len(df) < MA_LONG:
        return None

    try:
        rsi_bb = get_buy_signal(df)
        ma_info = ma_current_signal(df)
    except Exception as e:
        print(f"  [{ticker}] Error during analysis: {e}")
        return None

    return {
        'ticker': ticker,
        'price': rsi_bb['close'],
        'rsi': rsi_bb['rsi'],
        'buy_signal': rsi_bb['buy_signal'],
        'strength': rsi_bb['strength'],
        'rsi_oversold': rsi_bb['rsi_oversold'],
        'wick_touches': rsi_bb['wick_touches'],
        'wick_touched_bb': rsi_bb['wick_touched_bb'],
        'wick_dates': rsi_bb['wick_dates'],
        'bb_lower': rsi_bb['bb_lower'],
        'bb_mid': rsi_bb['bb_mid'],
        'bb_upper': rsi_bb['bb_upper'],
        'ma_trend': ma_info['trend'],
        'ma_last_crossover': ma_info['last_crossover'],
    }


def run_scanner(watchlist: list = WATCHLIST) -> list:
    width = 62
    print(f"\n{'='*width}")
    print(f"  DAILY TRADE SCANNER  —  {datetime.now().strftime('%Y-%m-%d  %H:%M')}")
    print(f"  RSI < {RSI_OVERSOLD}  +  wick(s) touching lower Bollinger Band")
    print(f"{'='*width}")
    print(f"  Scanning {len(watchlist)} tickers...\n")

    buy_signals = []
    watch_only = []   # RSI oversold but no wick touch yet
    errors = []

    for ticker in watchlist:
        try:
            result = scan_ticker(ticker)
            if result is None:
                continue
            if result['buy_signal']:
                buy_signals.append(result)
            elif result['rsi_oversold']:
                watch_only.append(result)
        except Exception as e:
            errors.append((ticker, str(e)))

    # Sort by signal strength descending
    buy_signals.sort(key=lambda x: x['strength'], reverse=True)
    watch_only.sort(key=lambda x: x['rsi'])

    # --- BUY SIGNALS ---
    if buy_signals:
        print(f"  {'BUY SIGNALS':^{width-4}}")
        print(f"  {'─'*(width-4)}")
        for sig in buy_signals:
            _print_signal(sig, label='BUY')
    else:
        print("  No full buy signals today.\n")

    # --- WATCHLIST: oversold RSI but waiting on BB wick ---
    if watch_only:
        print(f"\n  {'WATCH — RSI oversold, awaiting BB wick touch':^{width-4}}")
        print(f"  {'─'*(width-4)}")
        for sig in watch_only:
            _print_signal(sig, label='WATCH')

    # --- Errors ---
    if errors:
        print(f"\n  {'─'*(width-4)}")
        print("  Fetch errors:")
        for ticker, err in errors:
            print(f"    {ticker}: {err}")

    print(f"\n{'='*width}\n")
    return buy_signals


def _print_signal(sig: dict, label: str = 'BUY') -> None:
    xo = sig['ma_last_crossover']
    xo_str = ''
    if xo:
        date_str = xo['date'].strftime('%Y-%m-%d') if hasattr(xo['date'], 'strftime') else str(xo['date'])[:10]
        xo_str = f"  last {xo['type'].replace('_', ' ')} {date_str} @ ${xo['price']:.2f}"

    wick_str = ', '.join(sig['wick_dates'][-3:]) if sig['wick_dates'] else 'none'

    print(f"\n  [{label}] {sig['ticker']:<6}  ${sig['price']:.2f}   strength {sig['strength']:.0f}/100")
    print(f"    RSI:     {sig['rsi']:.1f}  (threshold < {RSI_OVERSOLD})")
    print(f"    BB:      lower ${sig['bb_lower']:.2f}  |  mid ${sig['bb_mid']:.2f}  |  upper ${sig['bb_upper']:.2f}")
    print(f"    Wicks at/below lower BB (last {BB_WICK_LOOKBACK} candles): {sig['wick_touches']}  → {wick_str}")
    print(f"    MA({MA_SHORT}/{MA_LONG}) trend: {sig['ma_trend'].upper()}{xo_str}")


if __name__ == '__main__':
    custom = [t.upper() for t in sys.argv[1:]] if len(sys.argv) > 1 else None
    run_scanner(custom or WATCHLIST)
