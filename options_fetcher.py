"""
Options chain data fetcher with daily caching.
Uses yfinance for options chains (calls/puts with strike, OI, volume, IV).
Handles futures → ETF proxy mapping (e.g., ES=F → SPY).
"""

import os
import json
from datetime import date
from typing import Optional, Tuple, List

import pandas as pd
import yfinance as yf

from config import CACHE_DIR, FUTURES_PROXY


def _resolve_ticker(ticker: str) -> Tuple[str, bool]:
    """Map futures tickers to their options-tradeable proxy.
    Returns (resolved_ticker, proxy_used)."""
    upper = ticker.upper()
    if upper in FUTURES_PROXY:
        return FUTURES_PROXY[upper], True
    return upper, False


def _cache_path(ticker: str, expiry: str) -> str:
    today = date.today().isoformat()
    safe = ticker.replace('=', '_').replace('/', '_')
    return os.path.join(CACHE_DIR, f"opts_{safe}_{expiry}_{today}.json")


def fetch_expiration_dates(ticker: str) -> List[str]:
    """Return available expiration dates for a ticker's options."""
    resolved, _ = _resolve_ticker(ticker)
    try:
        return list(yf.Ticker(resolved).options)
    except Exception:
        return []


def _load_latest_cache(resolved: str, original: str, proxy_used: bool):
    """Find and load the most recent cached options data for a ticker."""
    import glob
    safe = resolved.replace('=', '_').replace('/', '_')
    pattern = os.path.join(CACHE_DIR, f"opts_{safe}_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    for f in files:
        try:
            with open(f, 'r') as fh:
                cached = json.load(fh)
            calls = pd.DataFrame(cached['calls'])
            puts = pd.DataFrame(cached['puts'])
            if calls.empty:
                continue
            meta = cached['meta']
            meta['original_ticker'] = original
            meta['proxy_used'] = proxy_used
            return calls, puts, meta
        except Exception:
            continue
    return None


def fetch_options_chain(
    ticker: str,
    expiry: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Fetch calls and puts DataFrames for a given expiry.

    Returns (calls_df, puts_df, meta_dict).
    meta_dict contains: resolved_ticker, original_ticker, expiry,
                        spot_price, proxy_used.
    """
    resolved, proxy_used = _resolve_ticker(ticker)
    t = yf.Ticker(resolved)

    available = []
    try:
        available = list(t.options)
    except Exception:
        pass

    os.makedirs(CACHE_DIR, exist_ok=True)

    if not available:
        # After hours: try to load most recent cached data for this ticker
        cached_result = _load_latest_cache(resolved, ticker.upper(), proxy_used)
        if cached_result is not None:
            return cached_result
        return pd.DataFrame(), pd.DataFrame(), {}

    if expiry is None or expiry not in available:
        expiry = available[0]

    cache_file = _cache_path(resolved, expiry)

    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached = json.load(f)
        calls = pd.DataFrame(cached['calls'])
        puts = pd.DataFrame(cached['puts'])
        # Update meta to reflect current caller's proxy status
        meta = cached['meta']
        meta['original_ticker'] = ticker.upper()
        meta['proxy_used'] = proxy_used
        return calls, puts, meta

    try:
        chain = t.option_chain(expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
    except Exception:
        # Fetch failed — try cached data
        cached_result = _load_latest_cache(resolved, ticker.upper(), proxy_used)
        if cached_result is not None:
            return cached_result
        return pd.DataFrame(), pd.DataFrame(), {}

    # Fill NaN in volume/OI with 0
    for col in ('volume', 'openInterest'):
        if col in calls.columns:
            calls[col] = calls[col].fillna(0).astype(int)
        if col in puts.columns:
            puts[col] = puts[col].fillna(0).astype(int)

    # Get spot price
    hist = t.history(period='2d')
    spot = float(hist['Close'].iloc[-1]) if not hist.empty else 0.0

    # For futures tickers, also grab the actual futures price
    futures_spot = spot
    if proxy_used:
        try:
            fhist = yf.Ticker(ticker.upper()).history(period='2d')
            if not fhist.empty:
                futures_spot = float(fhist['Close'].iloc[-1])
        except Exception:
            pass

    meta = {
        'resolved_ticker': resolved,
        'original_ticker': ticker.upper(),
        'expiry': expiry,
        'spot_price': spot,
        'futures_spot': futures_spot,
        'proxy_used': proxy_used,
        'price_ratio': futures_spot / spot if spot > 0 else 1.0,
    }

    cache_data = {
        'calls': calls.to_dict(orient='records'),
        'puts': puts.to_dict(orient='records'),
        'meta': meta,
    }
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, default=str)
    except Exception:
        pass

    return calls, puts, meta


def fetch_multiple_expiries(
    ticker: str,
    n_expiries: int = 4,
    use_cache: bool = True,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Fetch chains for the nearest N expiration dates."""
    dates = fetch_expiration_dates(ticker)[:n_expiries]
    results = []
    for exp in dates:
        calls, puts, meta = fetch_options_chain(ticker, exp, use_cache)
        if not calls.empty:
            results.append((calls, puts, meta))
    return results
