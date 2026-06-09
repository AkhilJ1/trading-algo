"""
Grade matured forecasts → the durable Outcomes ledger.

This is the deferred half of the track record. The daily recorder writes a
point-in-time forecast (spot, floor/ceiling, dealer-pin estimated close) the
moment it is made. Later — once the forecast's expiry day has actually closed —
this runner looks up the realized close, scores the forecast against it (and
against the naive "price stays at spot" baseline), and appends one row to the
Outcomes sheet the dashboard reads.

It is safe to run repeatedly: track_record.pending_predictions() excludes any
forecast that already has an Outcomes row, so re-runs only fill in newly
matured days. Like the recorder, it prefers Google Sheets (what the website
reads, cloud-durable) and falls back to local CSV.

Usage:
    python grade_predictions.py          # grade every matured, ungraded forecast
"""

import sys

import pandas as pd

from data_fetcher import fetch_stock_data
from track_record import grade_prediction, pending_predictions, prediction_is_post_close
from sheets_logger import (
    is_sheets_available,
    read_predictions, read_predictions_csv,
    read_outcomes, read_outcomes_csv,
    log_outcome, log_outcome_csv,
)


def _realized_close(ticker: str, expiry: str):
    """
    Close of `ticker` on its forecast `expiry` day, or None if that day has not
    closed yet (so the forecast stays pending and is graded on a later run).
    """
    exp = pd.to_datetime(expiry, errors="coerce")
    if pd.isna(exp):
        return None
    # use_cache=False: grading runs the evening of the expiry day, when a
    # same-day cache file may have been written intraday — before the bar
    # settled — and data_fetcher drops the unfinished (NaN-close) row. Reading
    # that stale cache would hide the just-settled close and leave the forecast
    # wrongly "pending" forever. Always pull fresh so the realized close is seen.
    df = fetch_stock_data(ticker, period="3mo", interval="1d", use_cache=False)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    idx = pd.to_datetime(df.index)
    match = df[idx.normalize() == exp.normalize()]
    if match.empty:
        return None
    return float(match["Close"].iloc[-1])


def main(argv=None) -> int:
    sheets_ok = is_sheets_available()
    preds = read_predictions() if sheets_ok else read_predictions_csv()
    outs = read_outcomes() if sheets_ok else read_outcomes_csv()

    print(f"\n  GRADING RUN — source: {'Google Sheets' if sheets_ok else 'local CSV'}")
    if preds is None or preds.empty:
        print("  No predictions on record yet — nothing to grade.\n")
        return 0

    pend = pending_predictions(preds, outs)
    print(f"  {len(preds)} predictions on record, {len(pend)} matured & ungraded.\n")
    if pend.empty:
        print("  Up to date — no matured forecasts awaiting a grade.\n")
        return 0

    graded = 0
    for _, p in pend.iterrows():
        pred = p.to_dict()
        ticker = str(pred.get("ticker", "")).upper()
        expiry = pred.get("expiry", "")
        if prediction_is_post_close(pred):
            print(
                f"  [{ticker} {expiry}] skipped — recorded after its expiry "
                "session closed (outcome was already known; not a forecast)"
            )
            continue
        actual = _realized_close(ticker, expiry)
        if actual is None:
            print(f"  [{ticker} {expiry}] realized close not available yet — still pending")
            continue
        try:
            outcome = grade_prediction(pred, actual)
        except ValueError as e:
            print(f"  [{ticker} {expiry}] skipped — {e}")
            continue

        # When Sheets is the source of truth, require the Sheets write to
        # succeed — do NOT fall back to an ephemeral runner-local CSV that is
        # discarded when the job ends. The old `log_outcome(...) or
        # log_outcome_csv(...)` masked Sheets failures (e.g. a NaN cell) as
        # "graded", which is exactly why the scorecard looked empty while the
        # job reported success. CSV is only the destination when Sheets is absent.
        logged = log_outcome(outcome) if sheets_ok else log_outcome_csv(outcome)
        if logged:
            graded += 1
            print(
                f"  [{ticker} {expiry}] actual ${actual:.2f} vs est "
                f"${outcome['estimated_close']:.2f}  "
                f"err {outcome['close_abs_err']:.2f}  skill {outcome['skill']:+.2f}  "
                f"{'IN' if outcome['in_range'] else 'OUT of'} range"
            )
        else:
            print(f"  [{ticker} {expiry}] FAILED to log outcome")

    print(f"\n  Done — graded {graded} forecast(s).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
