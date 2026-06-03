"""
Shared data-quality validators (Requirement 3: "confirm data is accurate").

Centralizes the single definition of "is this data good enough to trust" so the
fallback provider and the options fetcher can't drift apart. After-hours feeds
frequently return chains whose rows have NaN/0 IV — they pass a naive row-count
check but produce Total GEX = 0 and empty walls. These gates catch that.
"""

import pandas as pd

# A chain needs at least this many strikes with real IV *and* open interest to
# be considered usable. Below this we treat the fetch as a failure.
MIN_USABLE_STRIKES = 20
MIN_USABLE_IV = 0.05  # 5% — below this is almost certainly stale/missing


def price_is_usable(df) -> bool:
    """True if a price-history frame has a real, positive most-recent close."""
    if df is None or getattr(df, "empty", True):
        return False
    if "Close" not in df.columns:
        return False
    try:
        return float(df["Close"].iloc[-1]) > 0
    except Exception:
        return False


def _usable_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    if "impliedVolatility" not in df.columns or "openInterest" not in df.columns:
        return 0
    iv = pd.to_numeric(df["impliedVolatility"], errors="coerce").fillna(0)
    oi = pd.to_numeric(df["openInterest"], errors="coerce").fillna(0)
    return int(((iv > MIN_USABLE_IV) & (oi > 0)).sum())


def chain_is_usable(calls, puts) -> bool:
    """
    True if the combined calls+puts chain has enough non-zero IV/OI rows to
    produce meaningful GEX, IV, and walls.
    """
    if calls is None or puts is None:
        return False
    if getattr(calls, "empty", True) and getattr(puts, "empty", True):
        return False
    return (_usable_count(calls) + _usable_count(puts)) >= MIN_USABLE_STRIKES
