"""
Snapshot every Google Sheets ledger to local CSV.

WHY THIS EXISTS
---------------
The forecast ledgers, graded outcomes and weight history live in Google Sheets,
which the GitHub workflows write to but nothing local can read without
credentials. That makes the track record effectively unanalysable offline —
you cannot tune weights against a scorecard you cannot load.

This pulls every configured tab into data/ledgers/*.csv in one shot, so
analysis (and any agent helping with it) works against a plain local snapshot
instead of needing live API access mid-thought. Re-run it whenever you want
fresh data; it is read-only and never writes back to the Sheet.

SETUP (once)
------------
Point GOOGLE_SHEETS_CREDS at a service-account JSON that has been shared on the
spreadsheet:

    export GOOGLE_SHEETS_CREDS=~/.config/trading-algo/gcp_sa.json
    python pull_ledgers.py

USAGE
-----
    python pull_ledgers.py              # all tabs
    python pull_ledgers.py Outcomes     # just the named tab(s)
"""

import os
import sys

import pandas as pd

import config
from sheets_logger import get_spreadsheet

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ledgers")

# Every tab worth pulling, in the order they matter for analysis.
TABS = [
    config.GSHEET_PREDICTIONS_SHEET,
    config.GSHEET_OUTCOMES_SHEET,
    config.GSHEET_PIN_FORECASTS_SHEET,
    config.GSHEET_PIN_OUTCOMES_SHEET,
    config.GSHEET_WEIGHTS_SHEET,
    config.GSHEET_CALIBRATION_SHEET,
    config.GSHEET_LEVELS_SHEET,
    config.GSHEET_FEATURES_SHEET,
]


def _freshness(df: pd.DataFrame) -> str:
    """Newest date-ish value in the frame, so a stale pull is obvious."""
    for col in ("graded_at", "timestamp", "pred_time", "date", "pred_date"):
        if col in df.columns and df[col].notna().any():
            newest = pd.to_datetime(df[col], errors="coerce").max()
            if pd.notna(newest):
                return f"newest {col}: {newest:%Y-%m-%d %H:%M}"
    return "no date column"


def main(argv=None) -> int:
    argv = list(argv or [])
    wanted = argv or TABS

    if not os.environ.get("GOOGLE_SHEETS_CREDS"):
        print(
            "ERROR: GOOGLE_SHEETS_CREDS is not set.\n"
            "       export GOOGLE_SHEETS_CREDS=~/.config/trading-algo/gcp_sa.json"
        )
        return 1

    try:
        ss = get_spreadsheet()
    except Exception as e:
        print(f"ERROR: could not open '{config.GSHEET_SPREADSHEET_NAME}': {e}")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n  LEDGER PULL — {config.GSHEET_SPREADSHEET_NAME} → data/ledgers/\n")

    pulled = 0
    for tab in wanted:
        try:
            ws = ss.worksheet(tab)
        except Exception:
            print(f"  [{tab}] not found — skipping")
            continue

        df = pd.DataFrame(ws.get_all_records())
        path = os.path.join(OUT_DIR, f"{tab}.csv")
        df.to_csv(path, index=False)
        pulled += 1
        detail = _freshness(df) if not df.empty else "empty"
        print(f"  [{tab}] {len(df):>5} rows → {os.path.basename(path)}  ({detail})")

    print(f"\n  Done — {pulled}/{len(wanted)} tab(s) pulled.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
