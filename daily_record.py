"""
Daily Prediction Recorder
-------------------------
Records one prediction data point per ticker per day (floor / ceiling / bias /
confidence + VIX, net GEX, regime) using the same analysis the dashboard runs.

Writes to the Google Sheet ("TradingAlgoPredictions" → "Predictions") when
credentials are available — which is what the website reads — and falls back to
the local data/predictions.csv otherwise.

Intended to be run once per (market) day by a scheduler. Running it more than
once per day is harmless: read_predictions() keeps the latest row per ticker
per day.

Usage:
    python daily_record.py            # records SPY (default)
    python daily_record.py SPY QQQ    # records a custom set of tickers
"""

import sys
from datetime import datetime

from strategies.fractal_options import compute_composite_analysis
from sheets_logger import (
    is_sheets_available,
    log_prediction,
    log_prediction_csv,
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
    """Latest VIX close + regime label, mirroring the dashboard."""
    try:
        from strategies.vix_filter import fetch_vix, classify_vix_regime
        vix_s = fetch_vix(period="1mo")
        if not vix_s.empty:
            val = float(vix_s.iloc[-1])
            return val, classify_vix_regime(val)
    except Exception:
        pass
    return None, None


def _pin_close_fields(result: dict):
    """Pull the dealer-pin forecast scalars we later grade (item 4)."""
    pin = result.get("estimated_close")
    if isinstance(pin, dict):
        return (
            pin.get("estimated_close"),
            pin.get("pin_target"),
            pin.get("max_pain", result.get("max_pain")),
        )
    return None, None, result.get("max_pain")


def record_ticker(ticker: str, sheets_ok: bool) -> bool:
    """Run the analysis for one ticker and log a single prediction row."""
    ticker = ticker.upper()
    result = compute_composite_analysis(ticker)
    if not result or "error" in result:
        err = result.get("error", "no result") if result else "no result"
        print(f"  [{ticker}] skipped — {err}")
        return False

    vix_val, regime = _fetch_vix()
    gex_net = _net_gex(result)
    est_close, pin_target, max_pain = _pin_close_fields(result)

    kwargs = dict(
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
        regime=regime if regime else result.get("market_regime"),
        estimated_close=est_close,
        pin_target=pin_target,
        max_pain=max_pain,
        # Provenance for data-accuracy auditing: which spot the estimate was
        # anchored on ('daily_close' after the close) and which backend served
        # the option chain it was built from ('schwab' / 'yfinance').
        spot_source=result.get("spot_source", "daily_close"),
        chain_source=result.get("source", ""),
    )

    logged = False
    if sheets_ok:
        try:
            logged = log_prediction(**kwargs)
        except Exception as e:
            print(f"  [{ticker}] sheets log failed ({e}); using CSV fallback")
    if not logged:
        logged = log_prediction_csv(**kwargs)

    dest = "Google Sheets" if (sheets_ok and logged) else "local CSV"
    est_str = f"  est-close ${est_close:.2f}" if est_close is not None else ""
    print(
        f"  [{ticker}] {kwargs['bias']} {kwargs['confidence']}%  "
        f"spot ${kwargs['spot_price']}  "
        f"floor ${kwargs['floor']}  ceiling ${kwargs['ceiling']}{est_str}  "
        f"→ {dest if logged else 'FAILED'}"
    )
    return logged


def main(tickers):
    sheets_ok = is_sheets_available()
    print(f"\n  DAILY RECORDER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
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
