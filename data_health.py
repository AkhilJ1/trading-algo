"""
Data-health check (Requirement 3: "confirm data is accurate all the time").

Runs in the cloud (GitHub Actions) once a day. It does two things:

  1. PIPELINE CHECK — fetches the live SPY option chain through the normal path
     and asserts the result is trustworthy: non-empty, passes the usability gate,
     has a real spot price, and is NOT being served from stale cache.

  2. CROSS-SOURCE RECONCILIATION — when Schwab credentials are present, pulls the
     SPY spot from BOTH Schwab and yfinance independently and flags drift beyond
     a tolerance. This is how we catch one source silently going wrong.

Exit code 0 = healthy. Exit code 1 = a real problem (the workflow then fails,
which emails the repo owner). "Schwab not configured" is NOT a failure — it just
skips reconciliation, because yfinance-only is a valid operating mode.

Usage:
    python data_health.py            # checks SPY
    python data_health.py SPY QQQ    # checks a custom set
"""

import sys

from options_fetcher import fetch_options_chain
from providers.quality import chain_is_usable

# Spot prices from two independent sources should agree closely intraday/at
# close. More than this fraction apart means something is wrong.
RECONCILE_TOLERANCE = 0.01  # 1%


def check_pipeline(ticker: str):
    """Returns (meta, problems[])."""
    problems = []
    calls, puts, meta = fetch_options_chain(ticker, use_cache=False)

    if (calls is None or calls.empty) and (puts is None or puts.empty):
        problems.append("empty chain (no calls and no puts)")
        return meta, problems

    if not chain_is_usable(calls, puts):
        problems.append("chain failed usability gate (too few real IV/OI strikes)")

    spot = meta.get("spot_price", 0) or 0
    if spot <= 0:
        problems.append(f"non-positive spot price ({spot})")

    if meta.get("stale"):
        problems.append(
            f"served STALE cache (as_of {meta.get('as_of')}, "
            f"file {meta.get('cache_file')}) — live fetch was unusable"
        )

    return meta, problems


def reconcile_sources(ticker: str):
    """
    Compare SPY spot from Schwab vs yfinance directly. Returns a list of
    problems. Skips cleanly (no problems) when Schwab isn't configured.
    """
    try:
        from providers.schwab_provider import SchwabProvider
        from providers.yfinance_provider import YFinanceProvider
    except Exception:
        return []

    schwab = SchwabProvider()
    yfin = YFinanceProvider()

    # Schwab side — any failure here (no creds/token/expired) means "not
    # configured / unavailable", which is not a data-accuracy problem.
    try:
        s_hist = schwab.get_price_history(ticker, period="5d", interval="1d")
        s_spot = float(s_hist["Close"].iloc[-1]) if not s_hist.empty else 0.0
    except Exception:
        return []  # Schwab unavailable — nothing to reconcile against.
    if s_spot <= 0:
        return []

    try:
        y_hist = yfin.get_price_history(ticker, period="5d", interval="1d")
        y_spot = float(y_hist["Close"].iloc[-1]) if not y_hist.empty else 0.0
    except Exception:
        return [f"yfinance reconciliation source unavailable for {ticker}"]
    if y_spot <= 0:
        return [f"yfinance spot non-positive for {ticker}"]

    drift = abs(s_spot - y_spot) / y_spot
    print(
        f"  [{ticker}] reconcile: schwab={s_spot:.2f} yfinance={y_spot:.2f} "
        f"drift={drift*100:.2f}%"
    )
    if drift > RECONCILE_TOLERANCE:
        return [
            f"spot drift {drift*100:.2f}% between Schwab ({s_spot:.2f}) and "
            f"yfinance ({y_spot:.2f}) exceeds {RECONCILE_TOLERANCE*100:.0f}%"
        ]
    return []


def main(tickers):
    all_problems = []
    for t in tickers:
        t = t.upper()
        meta, problems = check_pipeline(t)
        src = meta.get("source", "?") if meta else "?"
        print(f"  [{t}] source={src} spot={meta.get('spot_price') if meta else None}")
        for p in problems:
            print(f"    ✗ {p}")
        all_problems += [f"[{t}] {p}" for p in problems]

        for p in reconcile_sources(t):
            print(f"    ✗ {p}")
            all_problems.append(f"[{t}] {p}")

    if all_problems:
        print(f"\n  DATA HEALTH: FAIL ({len(all_problems)} problem(s))")
        return 1
    print("\n  DATA HEALTH: OK")
    return 0


if __name__ == "__main__":
    custom = [t.upper() for t in sys.argv[1:]]
    sys.exit(main(custom or ["SPY"]))
