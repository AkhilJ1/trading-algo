"""
Calibration Snapshot Recorder
-----------------------------
Records one range-engine calibration data point per scheduled run. Replays the
evidence-based floor/ceiling engine over a trailing window of SPY history (VIX
as the point-in-time IV proxy, range_calibration.py), grades how often each
confidence band actually contained the realized next-session close, and runs the
anchored out-of-sample width sweep — then appends a single snapshot row.

Writes to the Google Sheet ("TradingAlgoPredictions" → "Calibration") when
credentials are available — the same durable store the dashboard reads from —
and falls back to local data/calibration.csv otherwise.

Each row is a point-in-time-honest record of how well-calibrated the range
engine was as of that run, so a human (or the dashboard) can watch for
calibration drift over time without any local machine staying awake.

Usage:
    python record_calibration.py              # SPY, 2y window (defaults)
    python record_calibration.py SPY 5y       # custom ticker / window
"""

import sys
from datetime import datetime

import pandas as pd

from sheets_logger import (
    is_sheets_available,
    log_calibration,
    log_calibration_csv,
)


def build_calibration_snapshot(summary: dict, sweep: dict, *,
                               ticker: str, window: str, run_date: str) -> dict:
    """
    Flatten a calibration summary + out-of-sample sweep into one snapshot row
    (keys = CALIBRATION_HEADERS). Pure: dicts in, dict out — unit-testable
    offline with no network. Missing fields serialize to '' downstream.
    """
    cov = (summary or {}).get("coverage", {}) or {}
    sweep = sweep or {}
    sweep_ok = "error" not in sweep
    return {
        "date": run_date,
        "ticker": ticker,
        "window": window,
        "n_days": (summary or {}).get("n_days", 0),
        "cov_1sigma": cov.get("1sigma"),
        "cov_1_5sigma": cov.get("1_5sigma"),
        "cov_2sigma": cov.get("2sigma"),
        "calibration_error": (summary or {}).get("calibration_error"),
        "mean_width_pct": (summary or {}).get("mean_width_pct"),
        "best_width": sweep.get("best_width") if sweep_ok else None,
        "baseline_test_error": sweep.get("baseline_test_error") if sweep_ok else None,
        "proposed_test_error": sweep.get("proposed_test_error") if sweep_ok else None,
        "improved": bool(sweep.get("improved")) if sweep_ok else False,
    }


def _fetch_inputs(ticker: str, window: str):
    """Pull SPY price history + VIX/VIX3M for the calibration replay."""
    from data_fetcher import fetch_stock_data
    price = fetch_stock_data(ticker, period=window)
    vix = fetch_stock_data("^VIX", period=window)
    vix3m = fetch_stock_data("^VIX3M", period=window)
    vix_s = vix["Close"] if vix is not None and not vix.empty else pd.Series(dtype=float)
    v3_s = vix3m["Close"] if vix3m is not None and not vix3m.empty else None
    return price, vix_s, v3_s


def record(ticker: str, window: str, sheets_ok: bool) -> bool:
    """Run the calibration for one ticker/window and log a single snapshot row."""
    import range_calibration as rc

    ticker = ticker.upper()
    price, vix_s, v3_s = _fetch_inputs(ticker, window)
    if price is None or price.empty:
        print(f"  [{ticker}] skipped — no price data")
        return False

    res = rc.replay_and_summarize(price, vix_s, v3_s)
    summary = res["summary"]
    if summary.get("n_days", 0) == 0:
        print(f"  [{ticker}] skipped — not enough history to calibrate")
        return False
    sweep = rc.sweep_parameters(price, vix_s, v3_s)

    snapshot = build_calibration_snapshot(
        summary, sweep, ticker=ticker, window=window,
        run_date=datetime.now().strftime("%Y-%m-%d"),
    )

    logged = False
    if sheets_ok:
        try:
            logged = log_calibration(snapshot)
        except Exception as e:
            print(f"  [{ticker}] sheets log failed ({e}); using CSV fallback")
    if not logged:
        logged = log_calibration_csv(snapshot)

    dest = "Google Sheets" if (sheets_ok and logged) else "local CSV"
    cov = summary.get("coverage", {})
    verdict = "no error" if not sweep else (
        "widen x%.2f" % sweep["best_width"] if sweep.get("improved") else "keep x1.00"
    )
    print(
        f"  [{ticker}] {window}  n={summary['n_days']}  "
        f"1σ {cov.get('1sigma')}%  2σ {cov.get('2sigma')}%  "
        f"calib_err {summary.get('calibration_error')}  "
        f"OOS {verdict}  → {dest if logged else 'FAILED'}"
    )
    return logged


def main(argv):
    ticker = (argv[0] if len(argv) >= 1 else "SPY").upper()
    window = argv[1] if len(argv) >= 2 else "2y"

    sheets_ok = is_sheets_available()
    print(f"\n  CALIBRATION RECORDER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Destination: {'Google Sheets' if sheets_ok else 'local CSV (no Sheets creds)'}")
    print(f"  Calibrating {ticker} over {window}\n")

    try:
        ok = record(ticker, window, sheets_ok)
    except Exception as e:
        print(f"  [{ticker}] error: {e}")
        ok = False

    print(f"\n  Done — {'recorded' if ok else 'nothing recorded'}.\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
