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
from datetime import date, datetime, time as dtime, timezone
from typing import Optional, Tuple, List

import pandas as pd

from config import CACHE_DIR, FUTURES_PROXY, DATA_PROVIDER
from providers import get_provider
from providers.quality import chain_is_usable

# ── Intraday cache bucketing ───────────────────────────────────────────────
# WHY: the options cache used to be keyed by calendar DATE only, so the first
# Analyze of the day wrote opts_<tkr>_<exp>_<date>.json and every later Analyze
# re-read that same file — floor / ceiling / max-pain / dealer-pin were frozen
# at the day's first snapshot no matter how many times you re-ran. During market
# hours we now add an HHMM bucket to the cache key so the chain is re-fetched
# live every few minutes (levels actually move through the day, like the Fractal
# Exchange / Milk RCG graphs). Outside market hours we keep the date-only key so
# the after-close protection (serve last-known-good, never overwrite with a
# thin/closed-market chain) is completely unchanged.
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - zoneinfo ships with the py3.9+ runtime
    _ET = None

_INTRADAY_BUCKET_MINUTES = 5
_MARKET_OPEN = dtime(9, 30)
_MARKET_CLOSE = dtime(16, 0)


def _now_et() -> datetime:
    return datetime.now(_ET) if _ET is not None else datetime.now()


def _is_market_hours(now_et: Optional[datetime] = None) -> bool:
    """True on a weekday between 9:30 and 16:00 ET (regular session)."""
    now = now_et or _now_et()
    if now.weekday() >= 5:   # Sat / Sun
        return False
    return _MARKET_OPEN <= now.time() <= _MARKET_CLOSE


def _intraday_bucket(now_et: Optional[datetime] = None) -> str:
    """HHMM rounded down to the bucket size, e.g. 14:32 -> '1430'."""
    now = now_et or _now_et()
    minute = (now.minute // _INTRADAY_BUCKET_MINUTES) * _INTRADAY_BUCKET_MINUTES
    return f"{now.hour:02d}{minute:02d}"


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
    now_et = _now_et()
    if _is_market_hours(now_et):
        # Bucketed key rotates every _INTRADAY_BUCKET_MINUTES, so each Analyze in
        # a new bucket triggers a fresh live fetch instead of replaying the day's
        # first snapshot. Within a bucket, repeated Analyze reuses the file (no
        # hammering the Schwab API).
        return os.path.join(
            CACHE_DIR,
            f"opts_{safe}_{expiry}_{today}_{_intraday_bucket(now_et)}.json",
        )
    return os.path.join(CACHE_DIR, f"opts_{safe}_{expiry}_{today}.json")


# Matches the date in both the date-only key and the intraday-bucketed key
# (..._YYYY-MM-DD.json and ..._YYYY-MM-DD_HHMM.json).
_DATE_FROM_FILENAME = re.compile(r'_(\d{4}-\d{2}-\d{2})(?:_\d{4})?\.json$')


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


def _drop_expired(expiries: List[str]) -> List[str]:
    """Remove expirations that have already passed.

    Schwab's expiration list INCLUDES the most-recently-expired date and lists
    it FIRST (e.g. on a weekend, the prior Friday; after the close, that day's
    0DTE). Because fetch_options_chain picks ``available[0]`` when no expiry is
    requested, that stale date was being sent to get_option_chain, which returns
    an EMPTY map with NO exception — so FallbackProvider silently served
    yfinance instead of Schwab (proven via data_health probe: exps[0]=expired
    Friday → 0/0, nearest live expiry → 171/171 usable). Keeping only expiries
    >= today fixes the selection while preserving today's still-live 0DTE for
    the 9:25am pre-open recorder. Unparseable entries are kept, and if filtering
    would empty the list we return it unchanged (defensive)."""
    today = date.today()
    kept = []
    for e in expiries:
        try:
            d = datetime.strptime(str(e)[:10], "%Y-%m-%d").date()
        except Exception:
            kept.append(e)   # unparseable — keep rather than silently drop
            continue
        if d >= today:
            kept.append(e)
    return kept or list(expiries)


def fetch_expiration_dates(ticker: str) -> List[str]:
    """Return available (non-expired) expiration dates for a ticker's options."""
    resolved, _ = _resolve_ticker(ticker)
    return _drop_expired(get_provider().get_expirations(resolved))


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
    # mtime (not filename) so the newest snapshot wins regardless of whether it
    # was written with a date-only or an intraday-bucketed key.
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
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
    available = _drop_expired(available)   # never request an already-expired chain

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
        # Only serve this bucket's cache if the chain is still usable. Re-running
        # within the same bucket replays a known-good snapshot (cheap); a
        # thin/empty file never freezes the bucket — we fall through to a fresh
        # live fetch below.
        if chain_is_usable(calls, puts):
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
