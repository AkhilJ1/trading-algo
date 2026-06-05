"""
Tests for the calibration snapshot recorder (record_calibration.py).

The recorder flattens a range-engine calibration summary + out-of-sample sweep
into a single durable row, mirroring daily_record.py. Everything here is pure
and offline: hand-built summary/sweep dicts in, snapshot dict + serialized row
out — no network, no RNG — matching test_track_record.py's style.
"""
from record_calibration import build_calibration_snapshot
from sheets_logger import CALIBRATION_HEADERS, _calibration_row


def _summary(**over):
    base = {
        "n_days": 435,
        "coverage": {"1sigma": 72.0, "1_5sigma": 86.2, "2sigma": 93.8},
        "targets": {"1sigma": 68.27, "1_5sigma": 86.64, "2sigma": 95.45},
        "calibration_error": 1.923,
        "mean_width_pct": 1.67,
        "primary_label": "1sigma",
    }
    base.update(over)
    return base


def _sweep(**over):
    base = {
        "n_days": 435, "n_train": 304, "n_test": 131,
        "best_width": 1.05,
        "baseline_test": {}, "proposed_test": {},
        "baseline_test_error": 1.283, "proposed_test_error": 1.401,
        "improved": False,
    }
    base.update(over)
    return base


# ── snapshot building ────────────────────────────────────────────────────────

def test_snapshot_carries_coverage_and_oos_verdict():
    snap = build_calibration_snapshot(
        _summary(), _sweep(), ticker="SPY", window="2y", run_date="2026-06-04")
    assert snap["date"] == "2026-06-04"
    assert snap["ticker"] == "SPY" and snap["window"] == "2y"
    assert snap["n_days"] == 435
    assert snap["cov_1sigma"] == 72.0
    assert snap["cov_1_5sigma"] == 86.2
    assert snap["cov_2sigma"] == 93.8
    assert snap["calibration_error"] == 1.923
    assert snap["mean_width_pct"] == 1.67
    assert snap["best_width"] == 1.05
    assert snap["baseline_test_error"] == 1.283
    assert snap["proposed_test_error"] == 1.401
    assert snap["improved"] is False


def test_snapshot_marks_improvement_when_sweep_improves():
    snap = build_calibration_snapshot(
        _summary(), _sweep(best_width=1.20, proposed_test_error=1.10, improved=True),
        ticker="SPY", window="5y", run_date="2026-06-04")
    assert snap["improved"] is True
    assert snap["best_width"] == 1.20


def test_snapshot_handles_sweep_error_gracefully():
    # An insufficient-data sweep returns {"error": ...}; the snapshot should
    # blank the sweep fields rather than raise.
    snap = build_calibration_snapshot(
        _summary(), {"error": "insufficient data for calibration sweep"},
        ticker="SPY", window="2y", run_date="2026-06-04")
    assert snap["best_width"] is None
    assert snap["baseline_test_error"] is None
    assert snap["proposed_test_error"] is None
    assert snap["improved"] is False


def test_snapshot_empty_summary_is_safe():
    snap = build_calibration_snapshot(
        {}, {}, ticker="SPY", window="2y", run_date="2026-06-04")
    assert snap["n_days"] == 0
    assert snap["cov_1sigma"] is None
    assert snap["calibration_error"] is None


# ── row serialization (durable store contract) ───────────────────────────────

def test_row_matches_header_order_and_length():
    snap = build_calibration_snapshot(
        _summary(), _sweep(), ticker="SPY", window="2y", run_date="2026-06-04")
    row = _calibration_row(snap)
    assert len(row) == len(CALIBRATION_HEADERS)
    # First three columns are the identity of the snapshot.
    assert row[:4] == ["2026-06-04", "SPY", "2y", 435]
    # A snapshot missing a header serializes that cell to '' (append-only safe).
    partial = _calibration_row({"date": "2026-06-04", "ticker": "SPY"})
    assert partial[CALIBRATION_HEADERS.index("n_days")] == ""
