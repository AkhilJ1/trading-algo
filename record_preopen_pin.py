"""
Pre-Open Dealer-Pin Recorder
----------------------------
Records one *pre-open* dealer-pin close forecast per scheduled run, the way a
0DTE pin trader does it: before the 9:30 ET open, read the settled option open
interest (max-pain + dominant-gamma strike are already fixed overnight), anchor
on the live pre-market spot, and estimate where dealer hedging will pin SPY into
that day's expiry.

Two things make this *pre-open* honest rather than a replay of the after-close
recorder:

  1. Spot anchor — before the open the daily bar for today does not exist, so the
     normal spot is *yesterday's* settled close. We instead pull a live quote
     (pre-market / overnight via the provider's get_quote) and feed it to the
     analysis as `spot_override`, re-centering max-pain side, GEX, the IV
     envelope and the pin pull on where the market actually is. If no live quote
     is available we degrade to the daily close — the pin is still recorded.

  2. Expiry — the nearest SPY expiry pre-open is *today* (0DTE), which is exactly
     the contract whose pin matters for the day. compute_composite_analysis picks
     the nearest expiry automatically (and only rolls forward if 0DTE OI is thin,
     which it never is pre-open).

The forecast is written to its own durable ledger (Google Sheet "PinForecasts",
local data/pin_forecasts.csv fallback) — separate from the after-close
Predictions store so the two never collide on the same SPY date. grade_pin_
forecasts.py fills in the realized close that evening, producing a clean,
point-in-time-honest pre-open pin scorecard.

Everything is autonomous: a GitHub Action (.github/workflows/preopen-pin.yml)
runs this before the open daily, with no website button and no local machine.

Usage:
    python record_preopen_pin.py            # SPY (default)
    python record_preopen_pin.py SPY QQQ    # custom set of tickers
"""

import sys

from strategies.fractal_options import compute_composite_analysis
from options_fetcher import fetch_live_spot
from sheets_logger import (
    is_sheets_available,
    log_pin_forecast,
    log_pin_forecast_csv,
    log_levels_snapshot,
    log_levels_snapshot_csv,
    pacific_now,
)

DEFAULT_TICKERS = ["SPY"]


def _net_gex(result: dict):
    """Sum net gamma exposure from the analysis result, if present."""
    gex_df = result.get("gex_df")
    try:
        if gex_df is not None and not gex_df.empty and "net_gex" in gex_df:
            return float(gex_df["net_gex"].sum())
    except Exception:
        pass
    return None


def _fetch_vix():
    """Latest VIX close + regime label, mirroring the daily recorder."""
    try:
        from strategies.vix_filter import fetch_vix, classify_vix_regime
        vix_s = fetch_vix(period="1mo")
        if not vix_s.empty:
            val = float(vix_s.iloc[-1])
            return val, classify_vix_regime(val)
    except Exception:
        pass
    return None, None


def build_pin_forecast_kwargs(result: dict, *, ticker: str,
                              vix_val=None, regime=None, gex_net=None) -> dict:
    """
    Flatten a compute_composite_analysis result into log_pin_forecast kwargs.

    Pure: dict in, dict out — unit-testable offline with no network. The pin
    scalars we later grade (estimated_close, pin_target, max_pain) are pulled
    from the dealer-pin sub-result; spot_price carries the *anchor* spot used
    for the estimate (the live pre-market quote when one was available), which
    is exactly the naive baseline the grader scores the pin against.
    """
    pin = result.get("estimated_close")
    if isinstance(pin, dict):
        est_close = pin.get("estimated_close")
        pin_target = pin.get("pin_target")
        max_pain = pin.get("max_pain", result.get("max_pain"))
    else:
        est_close, pin_target, max_pain = None, None, result.get("max_pain")

    # Prefer the gamma regime from the pin engine (positive/negative gamma is the
    # single biggest driver of whether the pin holds); fall back to VIX regime.
    pin_regime = None
    if isinstance(pin, dict):
        pin_regime = pin.get("gamma_regime")
    regime_label = pin_regime or regime or result.get("market_regime")

    return dict(
        date_str=result.get("timestamp", "")[:10],
        ticker=result.get("ticker", ticker),
        spot_price=result.get("spot_price", 0),
        floor=result.get("floor", 0),
        ceiling=result.get("ceiling", 0),
        bias=result.get("bias", ""),
        confidence=result.get("confidence", 0),
        expiry=result.get("expiry", ""),
        vix=vix_val,
        gex_net=gex_net,
        regime=regime_label,
        estimated_close=est_close,
        pin_target=pin_target,
        max_pain=max_pain,
        # Provenance for data-accuracy auditing: was the pin anchored on a LIVE
        # pre-market quote ('live_override') or a stale settled close
        # ('daily_close'), and which backend served the option chain it was
        # built from ('schwab' / 'yfinance').
        spot_source=result.get("spot_source", "daily_close"),
        chain_source=result.get("source", ""),
    )


def record_ticker(ticker: str, sheets_ok: bool) -> bool:
    """Run the pre-open pin analysis for one ticker and log a single forecast row."""
    ticker = ticker.upper()

    # 1. Live pre-market spot (None ⇒ degrade to the daily close inside compute).
    live_spot = fetch_live_spot(ticker)

    # 2. Same analysis the dashboard runs, re-centered on the live spot. Nearest
    #    expiry pre-open = today's 0DTE — the pin that matters for the session.
    result = compute_composite_analysis(ticker, spot_override=live_spot)
    if not result or "error" in result:
        err = result.get("error", "no result") if result else "no result"
        print(f"  [{ticker}] skipped — {err}")
        return False

    vix_val, regime = _fetch_vix()
    gex_net = _net_gex(result)
    kwargs = build_pin_forecast_kwargs(
        result, ticker=ticker, vix_val=vix_val, regime=regime, gex_net=gex_net)

    logged = False
    if sheets_ok:
        try:
            logged = log_pin_forecast(**kwargs)
        except Exception as e:
            print(f"  [{ticker}] sheets log failed ({e}); using CSV fallback")
    if not logged:
        logged = log_pin_forecast_csv(**kwargs)

    # One levels-history row per scheduled run, so the chart can show this
    # run's floor/ceiling as its own line alongside the intraday ones.
    try:
        _r2 = (result.get("ranges") or {}).get("2sigma", {}) or {}
        _lvl_kwargs = dict(
            ticker=ticker, spot=result.get("spot_price"),
            floor=result.get("floor"), ceiling=result.get("ceiling"),
            floor2=_r2.get("floor"), ceiling2=_r2.get("ceiling"),
            est_close=kwargs["estimated_close"], source="pre_open",
        )
        if not (sheets_ok and log_levels_snapshot(**_lvl_kwargs)):
            log_levels_snapshot_csv(**_lvl_kwargs)
    except Exception as e:
        print(f"  [{ticker}] levels snapshot failed ({e}) — forecast still recorded")

    dest = "Google Sheets" if (sheets_ok and logged) else "local CSV"
    spot_src = result.get("spot_source", "daily_close")
    est = kwargs["estimated_close"]
    est_str = f"  pin ${est:.2f}" if isinstance(est, (int, float)) else ""
    print(
        f"  [{ticker}] {kwargs['expiry']} (0DTE)  "
        f"spot ${kwargs['spot_price']} [{spot_src}]{est_str}  "
        f"max_pain ${kwargs['max_pain']}  "
        f"→ {dest if logged else 'FAILED'}"
    )
    return logged


def main(tickers):
    sheets_ok = is_sheets_available()
    print(f"\n  PRE-OPEN PIN RECORDER — {pacific_now().strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"  Destination: {'Google Sheets' if sheets_ok else 'local CSV (no Sheets creds)'}")
    print(f"  Recording {len(tickers)} ticker(s): {', '.join(tickers)}\n")

    ok = 0
    for t in tickers:
        try:
            if record_ticker(t, sheets_ok):
                ok += 1
        except Exception as e:
            print(f"  [{t}] error: {e}")

    print(f"\n  Done — {ok}/{len(tickers)} recorded.\n")
    return 0 if ok == len(tickers) else 1


if __name__ == "__main__":
    custom = [t.upper() for t in sys.argv[1:]]
    sys.exit(main(custom or DEFAULT_TICKERS))
