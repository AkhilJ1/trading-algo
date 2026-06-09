"""
One-off ledger migration: convert legacy UTC timestamps to Pacific.

Run on 2026-06-09 alongside the Pacific-timestamps change (PR #27). Before
that change, every ledger stamp was written with datetime.utcnow() (since
commit 0de4da0, 2026-03-19 — "Predictions now include a UTC timestamp"); the
dashboard now labels these columns "(PT)", so the stored values must actually
be Pacific. This script rewrites, in place:

  * Predictions.timestamp, PinForecasts.timestamp  — UTC → America/Los_Angeles
  * Outcomes.graded_at, PinOutcomes.graded_at      — UTC → America/Los_Angeles
  * Outcomes.pred_time, PinOutcomes.pred_time      — backfilled from the
    matching forecast row's (converted) timestamp, joined on
    (date, ticker, expiry); blank when no forecast row matches.

Only rows whose stamp parses AND falls before the 2026-06-10 cutover are
touched — rows written by the new code are already Pacific. Blank stamps stay
blank (no fake times are invented).

!! Run ONCE. The conversion is not idempotent: a second --apply would shift
   the already-converted (pre-cutover-dated) stamps by another -7/-8 hours.

Usage:
    python migrate_timestamps_to_pacific.py            # dry-run: print the diff
    python migrate_timestamps_to_pacific.py --apply    # back up, then write
"""

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from sheets_logger import get_spreadsheet

UTC = ZoneInfo("UTC")
PT = ZoneInfo("America/Los_Angeles")
CUTOVER = datetime(2026, 6, 10)
FMT = "%Y-%m-%d %H:%M:%S"

BACKUP_DIR = (
    "/Users/akhiljoshi/trading-algo-ledger-backups/"
    + datetime.now(PT).strftime("%Y-%m-%d_%H%M%S")
)


def _to_pt(stamp: str):
    """'YYYY-MM-DD HH:MM:SS' UTC → same format in Pacific, or None to skip."""
    try:
        ts = datetime.strptime(stamp.strip(), FMT)
    except (ValueError, AttributeError):
        return None          # blank / unparseable — leave untouched
    if ts >= CUTOVER:
        return None          # written by the new code — already Pacific
    return ts.replace(tzinfo=UTC).astimezone(PT).strftime(FMT)


def _col_letter(idx0: int) -> str:
    letters = ""
    idx = idx0 + 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _load(ws):
    vals = ws.get_all_values()
    return vals[0], vals[1:]


def _backup(title, header, rows):
    import csv
    import os
    os.makedirs(BACKUP_DIR, exist_ok=True)
    path = f"{BACKUP_DIR}/{title}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  backed up {title} → {path}")


def _convert_column(title, header, rows, col_name, changes):
    i = header.index(col_name)
    out = []
    for rn, row in enumerate(rows, start=2):   # sheet row numbers (1 = header)
        cur = row[i] if i < len(row) else ""
        new = _to_pt(cur)
        if new is not None and new != cur:
            changes.append((title, rn, col_name, cur, new))
            out.append([new])
        else:
            out.append([cur])
    return i, out


def _forecast_time_index(header, rows):
    """(date, ticker, expiry) → latest converted-PT timestamp string."""
    di, ti, ei, si = (header.index(c) for c in ("date", "ticker", "expiry", "timestamp"))
    idx = {}
    for row in rows:
        if len(row) <= max(di, ti, ei, si):
            continue
        key = (row[di][:10], row[ti].upper(), row[ei][:10])
        stamp = _to_pt(row[si]) or row[si]
        if stamp and stamp >= idx.get(key, ""):
            idx[key] = stamp
    return idx


def main(apply: bool) -> int:
    ss = get_spreadsheet()
    changes = []
    writes = []   # (worksheet, a1_range, values)

    fc_index = {}
    for title in ("Predictions", "PinForecasts"):
        ws = ss.worksheet(title)
        header, rows = _load(ws)
        _backup(title, header, rows) if apply else None
        i, col = _convert_column(title, header, rows, "timestamp", changes)
        letter = _col_letter(i)
        writes.append((ws, f"{letter}2:{letter}{len(rows) + 1}", col))
        fc_index[title] = _forecast_time_index(header, rows)

    for title, fc_key in (("Outcomes", "Predictions"), ("PinOutcomes", "PinForecasts")):
        ws = ss.worksheet(title)
        header, rows = _load(ws)
        _backup(title, header, rows) if apply else None
        gi, gcol = _convert_column(title, header, rows, "graded_at", changes)
        gl = _col_letter(gi)
        writes.append((ws, f"{gl}2:{gl}{len(rows) + 1}", gcol))

        # Backfill pred_time from the matching forecast row.
        pi = header.index("pred_time")
        di, ti, ei = (header.index(c) for c in ("pred_date", "ticker", "expiry"))
        pcol = []
        for rn, row in enumerate(rows, start=2):
            row = row + [""] * (len(header) - len(row))
            cur = row[pi]
            if cur.strip():
                pcol.append([cur])     # already has a real stamp — keep it
                continue
            key = (row[di][:10], row[ti].upper(), row[ei][:10])
            new = fc_index[fc_key].get(key, "")
            if new:
                changes.append((title, rn, "pred_time", "(blank)", new))
            pcol.append([new])
        pl = _col_letter(pi)
        writes.append((ws, f"{pl}2:{pl}{len(rows) + 1}", pcol))

    print(f"\n  {len(changes)} cell change(s):")
    for title, rn, col, cur, new in changes:
        print(f"    {title:12s} row {rn:3d}  {col:10s}  {cur or '(blank)':>19s} → {new}")

    if not apply:
        print("\n  DRY RUN — nothing written. Re-run with --apply to migrate.\n")
        return 0

    for ws, rng, vals in writes:
        ws.update(rng, vals, value_input_option="RAW")
    print(f"\n  Applied. Backups in {BACKUP_DIR}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(apply="--apply" in sys.argv[1:]))
