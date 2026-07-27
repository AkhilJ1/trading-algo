"""
Ledger source alarm — is the track record actually being built from the data
source we think it is?

WHY THIS EXISTS
---------------
Between 2026-06 and 2026-07 every single forecast on record was built from
yfinance while the pipeline believed it was running on Schwab. The GitHub
`SCHWAB_TOKEN` secret had gone stale, every fetch quietly fell back, and nothing
complained: the workflows exited 0, the daily health probe passed, and the
re-auth notifier was watching token age rather than what actually got written.
Two months of track record turned out to be measuring a system nobody intended
to run, and it was only caught by auditing the ledger by hand.

Token age is a proxy. This checks the outcome instead: read back the rows that
were actually recorded and assert their `chain_source` matches the configured
provider. That catches every cause at once — expired token, corrupted secret,
failed quality gate, provider outage, bad deploy — because it asks the only
question that matters: did the data we intended to collect actually land?

Complementary to data_health.py, which probes the LIVE pipeline right now. This
one reads history, so it also catches a source that broke while nobody looked.

Exit codes (so a workflow can branch on them):
    0  = healthy, or nothing to check, or provider-agnostic mode (yfinance).
    10 = the recorded source has diverged from the configured provider.

Usage:
    python ledger_source_alarm.py                 # all ledgers, default window
    python ledger_source_alarm.py --sessions 5    # widen the tolerance
"""

import argparse
import os
import sys

import pandas as pd

import config
from sheets_logger import (
    is_sheets_available,
    read_predictions,
    read_predictions_csv,
)

# How many consecutive recorded sessions may disagree with the configured
# provider before this is treated as a real problem. One session can legitimately
# fall back (a thin pre-open chain failing the usability gate, a transient 5xx).
# Three consecutive sessions is not noise — that is a broken pipeline.
DEFAULT_MAX_FALLBACK_SESSIONS = 3

LEDGERS = [
    ("PinForecasts", config.GSHEET_PIN_FORECASTS_SHEET, "pin_forecasts.csv"),
    ("Predictions", config.GSHEET_PREDICTIONS_SHEET, "predictions.csv"),
]


def _load(sheet_name: str, csv_name: str, sheets_ok: bool) -> pd.DataFrame:
    """Read one ledger, preferring Sheets and falling back to local CSV."""
    try:
        if sheets_ok:
            return read_predictions(sheet=sheet_name)
        return read_predictions_csv(csv_name)
    except Exception as e:
        print(f"  [{sheet_name}] could not be read ({e}) — skipping")
        return pd.DataFrame()


def check_ledger(df: pd.DataFrame, label: str, expected: str, max_sessions: int):
    """
    Compare the most recent recorded sessions against the expected source.

    Returns (problem_or_None, detail_line). Counts distinct SESSIONS rather than
    rows: the recorders write two or three rows a day, so a row-based window
    would silently shrink to a single trading day.
    """
    if df is None or df.empty or "chain_source" not in df.columns:
        return None, f"  [{label}] no rows on record — nothing to check"

    d = df.copy()
    d["date"] = pd.to_datetime(d.get("date"), errors="coerce")
    d = d.dropna(subset=["date"])
    if d.empty:
        return None, f"  [{label}] no dated rows — nothing to check"

    d["chain_source"] = (
        d["chain_source"].fillna("").astype(str).str.strip().str.lower()
    )
    # One verdict per session: a session counts as on-source if ANY row that day
    # came from the expected provider, so a single bad intraday run is tolerated.
    per_session = (
        d.groupby(d["date"].dt.date)["chain_source"]
        .apply(lambda s: expected in set(s))
        .sort_index()
    )
    recent = per_session.tail(max_sessions)
    last_ok = per_session[per_session]
    last_ok_str = f"{last_ok.index[-1]}" if len(last_ok) else "never"

    if len(recent) < max_sessions:
        return None, (
            f"  [{label}] only {len(recent)} session(s) on record "
            f"(need {max_sessions}) — too early to judge"
        )

    if not recent.any():
        span = f"{recent.index[0]} → {recent.index[-1]}"
        return (
            f"[{label}] last {max_sessions} recorded sessions ({span}) carried no "
            f"'{expected}' rows — last good session: {last_ok_str}"
        ), f"  [{label}] ✗ {max_sessions} consecutive sessions off-source"

    return None, (
        f"  [{label}] ok — {int(recent.sum())}/{len(recent)} recent sessions on "
        f"'{expected}' (last good: {last_ok_str})"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sessions", type=int, default=DEFAULT_MAX_FALLBACK_SESSIONS,
        help="consecutive off-source sessions tolerated before alarming",
    )
    args = ap.parse_args(argv)

    expected = (os.environ.get("DATA_PROVIDER") or config.DATA_PROVIDER or "").lower()
    print(f"\n  LEDGER SOURCE ALARM — expected chain_source: '{expected}'\n")

    # yfinance is the provider-agnostic default: there is no fallback to detect,
    # so this check has nothing to say. Never alarm in that mode.
    if expected != "schwab":
        print("  Provider is not 'schwab' — fallback detection does not apply.")
        return 0

    sheets_ok = is_sheets_available()
    print(f"  Source: {'Google Sheets' if sheets_ok else 'local CSV'}")

    problems = []
    for label, sheet_name, csv_name in LEDGERS:
        df = _load(sheet_name, csv_name, sheets_ok)
        problem, detail = check_ledger(df, label, expected, args.sessions)
        print(detail)
        if problem:
            problems.append(problem)

    if problems:
        print("\n  SOURCE ALARM: the ledger is not being built from "
              f"'{expected}'.\n")
        for p in problems:
            print(f"    ✗ {p}")
        print(
            "\n  The pipeline is still recording, so nothing looks broken from "
            "the outside — but the track record being written right now is not "
            "the one you think you are collecting.\n"
        )
        return 10

    print("\n  SOURCE ALARM: OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
