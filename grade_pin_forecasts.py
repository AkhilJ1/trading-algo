"""
Grade matured pre-open pin forecasts → the durable PinOutcomes ledger.

This is the evening half of the pre-open pin track record — the part that "fills
in the realized close that evening." record_preopen_pin.py writes a point-in-time
pin forecast before the open (anchored on the live pre-market spot, against that
day's 0DTE expiry). Once that expiry day has actually closed, this runner looks
up the realized SPY close, scores the pin against it — and against the naive
"price stays at the pre-market spot" baseline — and appends one row to the
PinOutcomes store the scorecard reads.

It reuses the exact same honest grading logic as the after-close track record
(track_record.grade_prediction / pending_predictions and grade_predictions.
_realized_close); only the source/destination ledgers differ. Safe to run
repeatedly: already-graded forecasts are skipped, so re-runs only fill in newly
matured days. Prefers Google Sheets (cloud-durable) and falls back to local CSV.

Usage:
    python grade_pin_forecasts.py     # grade every matured, ungraded pin forecast
"""

import sys

from track_record import grade_prediction, pending_predictions, prediction_is_post_close
from grade_predictions import _GRADE_LOOKBACK, _realized_close
from sheets_logger import (
    is_sheets_available,
    read_pin_forecasts, read_pin_forecasts_csv,
    read_pin_outcomes, read_pin_outcomes_csv,
    log_pin_outcome, log_pin_outcome_csv,
)


def main(argv=None) -> int:
    sheets_ok = is_sheets_available()
    preds = read_pin_forecasts() if sheets_ok else read_pin_forecasts_csv()
    outs = read_pin_outcomes() if sheets_ok else read_pin_outcomes_csv()

    print(f"\n  PIN GRADING RUN — source: {'Google Sheets' if sheets_ok else 'local CSV'}")
    if preds is None or preds.empty:
        print("  No pin forecasts on record yet — nothing to grade.\n")
        return 0

    pend = pending_predictions(preds, outs)
    print(f"  {len(preds)} pin forecasts on record, {len(pend)} matured & ungraded.\n")
    if pend.empty:
        print("  Up to date — no matured pin forecasts awaiting a grade.\n")
        return 0

    graded = 0
    pending = 0
    unavailable = 0
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
        actual, status = _realized_close(ticker, expiry)
        if status == "pending":
            print(f"  [{ticker} {expiry}] realized close not available yet — still pending")
            pending += 1
            continue
        if status == "unavailable":
            print(
                f"  [{ticker} {expiry}] un-gradeable — no close on record within "
                f"the {_GRADE_LOOKBACK} lookback (expiry too old, or not a trading day)"
            )
            unavailable += 1
            continue
        try:
            outcome = grade_prediction(pred, actual)
        except ValueError as e:
            print(f"  [{ticker} {expiry}] skipped — {e}")
            continue

        # When Sheets is the source of truth, require the Sheets write to
        # succeed — do NOT silently fall back to an ephemeral runner-local CSV
        # that is discarded when the job ends. The old `log_pin_outcome(...) or
        # log_pin_outcome_csv(...)` masked Sheets failures (e.g. a NaN cell) as
        # "graded", which left the scorecard empty while the job reported
        # success. CSV is only the destination when Sheets is absent.
        logged = log_pin_outcome(outcome) if sheets_ok else log_pin_outcome_csv(outcome)
        if logged:
            graded += 1
            print(
                f"  [{ticker} {expiry}] actual ${actual:.2f} vs pin "
                f"${outcome['estimated_close']:.2f}  "
                f"err {outcome['close_abs_err']:.2f}  skill {outcome['skill']:+.2f}  "
                f"{'IN' if outcome['in_range'] else 'OUT of'} range"
            )
        else:
            print(f"  [{ticker} {expiry}] FAILED to log pin outcome")

    print(
        f"\n  Done — graded {graded} pin forecast(s); "
        f"{pending} still pending, {unavailable} un-gradeable.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
