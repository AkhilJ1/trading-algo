"""
Regression: NaN/inf must never reach the gspread write boundary.

The production bug — graders reporting success while the Outcomes scorecard
stayed empty — was a serialization failure. gspread writes rows with
value_input_option="RAW", which serializes through JSON, and JSON has no
NaN/Infinity. A single non-finite float raised

    Out of range float values are not JSON compliant

and Google rejected the WHOLE row. A blank `estimated_close` cell (older
predictions predating that column) read back through pandas as NaN, sailed
through the old `is None` guard, and poisoned every derived metric.

These tests pin the durable defense at the write boundary: `_cell_safe` /
`_json_safe` blank out any non-finite number, and the real row-builders
(`_outcome_row`, `_prediction_row`) emit rows that survive a strict,
JSON-compliant encode. Pure and offline — no credentials, no network.
"""
import json
import math

from sheets_logger import (
    _cell_safe,
    _json_safe,
    _num_or_blank,
    _outcome_row,
    _prediction_row,
    OUTCOME_HEADERS,
)


def _json_ok(row):
    """True iff the row encodes under JSON's strict (no-NaN) mode, as gspread needs."""
    json.dumps(row, allow_nan=False)
    return True


# ── _cell_safe / _json_safe ──────────────────────────────────────────────────

def test_cell_safe_blanks_nan_and_inf():
    assert _cell_safe(float("nan")) == ""
    assert _cell_safe(float("inf")) == ""
    assert _cell_safe(float("-inf")) == ""


def test_cell_safe_passes_finite_numbers_and_non_numbers():
    assert _cell_safe(603.21) == 603.21
    assert _cell_safe(0) == 0
    assert _cell_safe("up") == "up"       # strings untouched
    assert _cell_safe("") == ""
    assert _cell_safe(None) is None       # None is a valid blank for gspread


def test_json_safe_sanitizes_a_mixed_row():
    row = ["SPY", 603.0, float("nan"), "up", float("inf"), None, ""]
    safe = _json_safe(row)
    assert safe == ["SPY", 603.0, "", "up", "", None, ""]
    assert _json_ok(safe)


def test_num_or_blank_blanks_non_finite():
    assert _num_or_blank(float("nan")) == ""
    assert _num_or_blank(float("inf")) == ""
    assert _num_or_blank(None) == ""
    assert _num_or_blank("garbage") == ""
    assert _num_or_blank(603.219, 2) == 603.22


# ── the real row-builders survive a strict JSON encode ───────────────────────

def test_outcome_row_with_nan_fields_is_json_compliant():
    # Exactly the shape that broke production: a graded-outcome dict in which a
    # few numeric fields are NaN/inf. The row must still encode under allow_nan
    # =False (i.e. gspread will accept it) with those cells blanked.
    poisoned = {h: float("nan") for h in OUTCOME_HEADERS}
    poisoned["ticker"] = "SPY"
    poisoned["expiry"] = "2026-06-05"
    poisoned["estimated_close"] = float("inf")
    row = _outcome_row(poisoned)

    assert _json_ok(row)
    for v in row:
        assert not (isinstance(v, float) and not math.isfinite(v))
    # The string fields survive; the non-finite numbers became blanks.
    assert "SPY" in row and "2026-06-05" in row


def test_clean_outcome_row_preserves_values():
    clean = {h: "" for h in OUTCOME_HEADERS}
    clean.update(ticker="SPY", actual_close=604.0, estimated_close=603.0,
                 close_abs_err=1.0, skill=3.0)
    row = _outcome_row(clean)
    assert _json_ok(row)
    assert 604.0 in row and 603.0 in row and 3.0 in row


def test_prediction_row_with_nan_estimate_is_json_compliant():
    # A pin forecast whose estimated_close could not be computed (NaN) must not
    # reject the whole Predictions write — the cell blanks, the row survives.
    row = _prediction_row(
        date_str="2026-06-05", ticker="SPY", spot_price=600.0,
        floor=float("nan"), ceiling=float("inf"), bias="neutral",
        confidence=float("nan"), expiry="2026-06-05", vix=14.2, gex_net=None,
        regime="positive_gamma", estimated_close=float("nan"),
        pin_target=601.0, max_pain=600.0,
    )
    assert _json_ok(row)
    assert "SPY" in row


# ── prediction dedupe identity (the shadowed-overnight-forecast bug) ─────────

def test_dedupe_keeps_same_day_rows_with_different_expiries():
    """The 1:16pm recorder's new overnight row (same date, NEXT expiry) must not
    shadow the previous evening's still-ungraded forecast for TODAY's expiry —
    that shadowing left the dealer-pin track record permanently empty."""
    import pandas as pd
    from sheets_logger import _dedupe_latest_predictions

    df = pd.DataFrame([
        # Yesterday-evening overnight forecast for today's expiry (matured,
        # ungraded) — the row that was being silently dropped.
        {"date": pd.Timestamp("2026-06-09"), "ticker": "SPY",
         "expiry": "2026-06-09", "estimated_close": 739.91,
         "timestamp": pd.Timestamp("2026-06-08 23:55:28")},
        # Today's 1:16pm recording for tomorrow's expiry (later timestamp).
        {"date": pd.Timestamp("2026-06-09"), "ticker": "SPY",
         "expiry": "2026-06-10", "estimated_close": 734.68,
         "timestamp": pd.Timestamp("2026-06-09 16:42:15")},
    ])
    out = _dedupe_latest_predictions(df)
    assert len(out) == 2                       # both expiries survive
    assert set(out["expiry"]) == {"2026-06-09", "2026-06-10"}


def test_dedupe_still_collapses_same_day_same_expiry_reruns():
    import pandas as pd
    from sheets_logger import _dedupe_latest_predictions

    df = pd.DataFrame([
        {"date": pd.Timestamp("2026-06-09"), "ticker": "SPY",
         "expiry": "2026-06-10", "estimated_close": 734.10,
         "timestamp": pd.Timestamp("2026-06-09 13:16:00")},
        {"date": pd.Timestamp("2026-06-09"), "ticker": "SPY",
         "expiry": "2026-06-10", "estimated_close": 734.68,
         "timestamp": pd.Timestamp("2026-06-09 16:42:15")},
    ])
    out = _dedupe_latest_predictions(df)
    assert len(out) == 1                       # re-runs still collapse
    assert float(out.iloc[0]["estimated_close"]) == 734.68   # latest wins
