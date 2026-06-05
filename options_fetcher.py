"""
Options chain data fetcher with daily caching.

Delegates the raw market-data fetch to the active provider
(providers.get_provider) — yfinance by default, or Schwab-with-yfinance-fallback
when DATA_PROVIDER=schwab. Everything provider-agnostic lives here: futures→ETF
proxy resolution, NaN handling, quality validation, staleness tagging, source
stamping, and meta construction.

After-hours / post-midnight resilience (Requirement 3):
  - Validates every live fetch with providers.quality.chain_is_usable() before
    trusting it.
  - On an unusable fetch, falls back to the most recent GOOD cache and tags
    meta['stale'] = True + meta['as_of'] = <iso date> so the dashboard can show
    when the data was actually captured.
  - NEVER overwrites a good cache with a bad/empty fetch (this was the bug that
    caused Total GEX = 0 after midnight EST).
  - Stamps every successful cache write with meta['as_of'] and meta['source']
    (which backend actually served the chain).
"""

import os
import re
import json
import glob
from datetime import date
from typing import Optional, Tuple, List

import pandas as pd

from config import CACHE_DIR, FUTURES_PROXY, DATA_PROVIDER
from providers import get_provider
from providers.quality import chain_is_usable


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


_DATE_FROM_FILENAME = re.compile(r'_(\d{4}-\d{2}-\d{2})\.json$')


def _as_of_from_meta_or_filename(filename: str, meta: dict) -> Optional[str]:
    """Best-effort 'as of' ISO date for a cached file."""
    if isinstance(meta, dict) and meta.get('as_of'):
        return str(meta['as_of'])
    m = _DATE_FROM_FILENAME.search(filename)
    if m:
        return m.group(1)
    return None


def _chain_source(provider) -> str:
    """Which backend served the most recent chain, for monitoring/meta.

    FallbackProvider records this on `.last_source['chain']`; a plain provider
    doesn't, so we fall back to the configured DATA_PROVIDER name.
    """
    ls = getattr(provider, 'last_source', None)
    if isinstance(ls, dict) and ls.get('chain'):
        return ls['chain']
    return getattr(provider, 'name', DATA_PROVIDER)


def fetch_expiration_dates(ticker: str) -> List[str]:
    """Return available expiration dates for a ticker's options."""
    resolved, _ = _resolve_ticker(ticker)
    return get_provider().get_expirations(resolved)


def fetch_live_spot(ticker: str, provider=None) -> Optional[float]:
    """
    Freshest available spot for `ticker`, including pre/post-market, or None.

    This is the pre-open spot fix: before the 9:30 ET open, the daily bar for
    today does not exist yet, so the normal spot path (last daily Close) returns
    *yesterday's* settled close. A live quote (pre-market / overnight) is a far
    better anchor for a pre-open dealer-pin estimate. Futures tickers resolve to
    their options-tradeable ETF proxy first, so the price is in the same units
    the analysis works in. Returns None when no live quote is available — the
    caller then degrades to the daily-close spot, so the estimate is never
    blocked, only sharpened when a quote exists.
    """
    resolved, _ = _resolve_ticker(ticker)
    prov = provider if provider is not None else get_provider()
    try:
        px = prov.get_quote(resolved)
    except Exception:
        return None
    try:
        px = float(px)
    except (TypeError, ValueError):
        return None
    return px if px > 0 else None


def _load_latest_cache(
    resolved: str,
    original: str,
    proxy_used: bool,
    require_usable: bool = True,
):
    """
    Find and load the most recent cached options data for a ticker.

    When `require_usable=True` (default), skips any cached file whose chain
    fails chain_is_usable() — this protects against propagating bad snapshots
    that may have been written by older code paths.

    Returns (calls, puts, meta) where meta includes:
        meta['stale'] = True
        meta['as_of'] = '<YYYY-MM-DD>'  (when the data was originally captured)
    """
    safe = resolved.replace('=', '_').replace('/', '_')
    pattern = os.path.join(CACHE_DIR, f"opts_{safe}_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    for f in files:
        try:
            with open(f, 'r') as fh:
                cached = json.load(fh)
            calls = pd.DataFrame(cached.get('calls', []))
            puts = pd.DataFrame(cached.get('puts', []))
            if calls.empty and puts.empty:
                continue
            if require_usable and not chain_is_usable(calls, puts):
                continue
            meta = dict(cached.get('meta', {}))
            meta['original_ticker'] = original
            meta['proxy_used'] = proxy_used
            meta['stale'] = True
            meta['as_of'] = _as_of_from_meta_or_filename(f, meta)
            meta['cache_file'] = os.path.basename(f)
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
    meta_dict contains: resolved_ticker, original_ticker, expiry, spot_price,
                        futures_spot, proxy_used, price_ratio, source, as_of,
                        stale.
    """
    resolved, proxy_used = _resolve_ticker(ticker)
    provider = get_provider()

    available = provider.get_expirations(resolved)

    os.makedirs(CACHE_DIR, exist_ok=True)

    if not available:
        # After hours / source down: load most recent GOOD cache for this ticker.
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
        # Today's cache is fresh (not stale). Update meta for current caller.
        meta = dict(cached['meta'])
        meta['original_ticker'] = ticker.upper()
        meta['proxy_used'] = proxy_used
        meta.setdefault('stale', False)
        return calls, puts, meta

    try:
        calls, puts = provider.get_option_chain(resolved, expiry)
    except Exception:
        # Fetch failed — try most recent good cache.
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

    # QUALITY GATE: if the live chain is unusable, prefer the last good cache
    # over a bad/empty snapshot — and crucially, do NOT cache the bad data.
    usable = chain_is_usable(calls, puts)
    if not usable:
        cached_result = _load_latest_cache(resolved, ticker.upper(), proxy_used)
        if cached_result is not None:
            return cached_result
        # No good cache exists — return best-effort live data but don't persist it.

    # Get spot price
    hist = provider.get_price_history(resolved, period='2d', interval='1d')
    spot = float(hist['Close'].iloc[-1]) if not hist.empty else 0.0

    # For futures tickers, also grab the actual futures price
    futures_spot = spot
    if proxy_used:
        try:
            fhist = provider.get_price_history(ticker.upper(), period='2d', interval='1d')
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
        'source': _chain_source(provider),
        'as_of': date.today().isoformat(),
        'stale': False,
    }

    # Only persist GOOD snapshots — never overwrite a good cache with bad data.
    if usable:
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
