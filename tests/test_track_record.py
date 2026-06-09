"""
Track-record grading tests (Step 3).

These prove the durable point-in-time ledger is graded *honestly*:

  * a dealer-pin close forecast is scored against the realized close AND
    against the naive "price stays at spot" null model, so "skill" only shows
    up when the model genuinely beats doing nothing,
  * the [floor, ceiling] band and the directional call are graded too, and
  * maturity/idempotency logic only ever grades a forecast once, and only
    after its expiry day has actually closed.

Everything is pure and offline: dicts and DataFrames, no network, no RNG.
"""
import pandas as pd

from track_record import (
    grade_prediction,
    summarize_track_record,
    pending_predictions,
    join_predictions_outcomes,
)


def _pred(**over):
    base = {
        "date": "2026-06-01", "ticker": "SPY", "expiry": "2026-06-01",
        "spot_price": 600.0, "floor": 592.0, "ceiling": 608.0,
        "estimated_close": 603.0,
    }
    base.update(over)
    return base


# ── grade_prediction ───────────────────────────────────────────────────────

def test_skillful_forecast_beats_the_naive_spot_baseline():
    # spot 600, we forecast 603, price closes 604 → est err 1 < naive err 4.
    out = grade_prediction(_pred(), actual_close=604.0)
    assert out["close_abs_err"] == 1.0
    assert out["naive_abs_err"] == 4.0
    assert out["skill"] == 3.0          # naive_err − model_err, positive ⇒ value added
    assert out["in_range"] is True
    assert out["dir_predicted"] == "up" and out["dir_actual"] == "up"
    assert out["dir_correct"] is True


def test_unskillful_forecast_has_negative_skill():
    # We forecast 603 but price barely moves to 600.5: naive (stay at spot) wins.
    out = grade_prediction(_pred(), actual_close=600.5)
    assert out["close_abs_err"] == 2.5
    assert out["naive_abs_err"] == 0.5
    assert out["skill"] == -2.0
    assert out["dir_correct"] is True   # both called "up", even if magnitude was off


def test_out_of_range_close_is_flagged():
    out = grade_prediction(_pred(), actual_close=611.0)  # above ceiling 608
    assert out["in_range"] is False


def test_wrong_direction_is_caught():
    # Forecast up (603 > 600) but price closes below spot.
    out = grade_prediction(_pred(), actual_close=597.0)
    assert out["dir_predicted"] == "up"
    assert out["dir_actual"] == "down"
    assert out["dir_correct"] is False


def test_missing_estimated_close_raises():
    p = _pred()
    p.pop("estimated_close")
    try:
        grade_prediction(p, 604.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_missing_band_leaves_in_range_blank_but_still_grades_close():
    p = _pred(floor="", ceiling="")
    out = grade_prediction(p, actual_close=604.0)
    assert out["in_range"] == ""
    assert out["close_abs_err"] == 1.0   # close error still computed


def test_pct_error_is_relative_to_actual_close():
    out = grade_prediction(_pred(estimated_close=600.0), actual_close=600.0 * 1.01)
    # |606 − 600| / 606 * 100
    assert abs(out["close_pct_err"] - (6.0 / 606.0 * 100.0)) < 1e-6


# ── NaN / inf handling (the production bug: blank Sheets cells read back as NaN) ─
#
# pandas coerces a blank Predictions cell to NaN, and `nan is not None` is True,
# so the old `is None` guard let NaN flow into the metrics — producing an
# all-NaN outcome row that gspread then rejected ("Out of range float values are
# not JSON compliant"), silently grading nothing. These pin the fix: a missing/
# garbage number must read as genuinely missing → skip (ValueError), never as a
# poisoned-but-passing forecast.

def test_nan_estimated_close_is_treated_as_missing():
    # Older daily predictions predate the estimated_close column → NaN on read.
    out_pred = _pred(estimated_close=float("nan"))
    try:
        grade_prediction(out_pred, 604.0)
        assert False, "expected ValueError for NaN estimated_close"
    except ValueError:
        pass


def test_nan_spot_is_treated_as_missing():
    try:
        grade_prediction(_pred(spot_price=float("nan")), 604.0)
        assert False, "expected ValueError for NaN spot"
    except ValueError:
        pass


def test_inf_estimated_close_is_treated_as_missing():
    try:
        grade_prediction(_pred(estimated_close=float("inf")), 604.0)
        assert False, "expected ValueError for non-finite estimated_close"
    except ValueError:
        pass


def test_nan_actual_close_raises():
    try:
        grade_prediction(_pred(), actual_close=float("nan"))
        assert False, "expected ValueError for NaN realized close"
    except ValueError:
        pass


def test_nan_band_leaves_in_range_blank_but_still_grades_close():
    # A NaN floor/ceiling (blank band cell) must not poison the row: in_range
    # falls back to blank, but the close error is still scored.
    out = grade_prediction(_pred(floor=float("nan"), ceiling=float("nan")),
                           actual_close=604.0)
    assert out["in_range"] == ""
    assert out["close_abs_err"] == 1.0
    # Every numeric field that *is* produced must be finite (JSON-writable).
    import math as _m
    for k, v in out.items():
        if isinstance(v, float):
            assert _m.isfinite(v), f"{k} is non-finite: {v}"


# ── summarize_track_record ───────────────────────────────────────────────────

def _graded(skill, in_range, dir_correct, close_err, naive_err):
    return {
        "close_abs_err": close_err, "naive_abs_err": naive_err, "skill": skill,
        "close_pct_err": close_err / 600.0 * 100.0,
        "in_range": in_range, "dir_correct": dir_correct,
    }


def test_summary_is_safe_on_empty_frame():
    s = summarize_track_record(pd.DataFrame())
    assert s["n_graded"] == 0
    assert s["beats_naive"] is None
    assert s["mean_abs_err"] is None


def test_summary_aggregates_skill_and_rates():
    df = pd.DataFrame([
        _graded(skill=3.0, in_range=True, dir_correct=True, close_err=1.0, naive_err=4.0),
        _graded(skill=-2.0, in_range=True, dir_correct=True, close_err=2.5, naive_err=0.5),
        _graded(skill=1.0, in_range=False, dir_correct=False, close_err=2.0, naive_err=3.0),
    ])
    s = summarize_track_record(df)
    # Stored metrics are rounded to 4 dp, so compare at that precision.
    assert s["n_graded"] == 3
    assert abs(s["mean_abs_err"] - (1.0 + 2.5 + 2.0) / 3) < 1e-3
    assert abs(s["naive_mean_abs_err"] - (4.0 + 0.5 + 3.0) / 3) < 1e-3
    # mean model err (1.833) < mean naive err (2.5) ⇒ the model beats doing nothing.
    assert s["beats_naive"] is True
    assert abs(s["mean_skill"] - (3.0 - 2.0 + 1.0) / 3) < 1e-3
    assert abs(s["skill_rate"] - 2 / 3) < 1e-3      # 2 of 3 rows had positive skill
    assert abs(s["in_range_rate"] - 2 / 3) < 1e-3
    assert abs(s["dir_accuracy"] - 2 / 3) < 1e-3


def test_summary_parses_sheet_roundtrip_string_bools():
    # Google Sheets hands booleans back as 'TRUE'/'FALSE' strings.
    df = pd.DataFrame([
        {**_graded(2.0, "TRUE", "TRUE", 1.0, 3.0)},
        {**_graded(2.0, "FALSE", "TRUE", 1.0, 3.0)},
    ])
    s = summarize_track_record(df)
    assert abs(s["in_range_rate"] - 0.5) < 1e-9
    assert abs(s["dir_accuracy"] - 1.0) < 1e-9


# ── pending_predictions (maturity + idempotency) ─────────────────────────────

def test_pending_selects_matured_and_excludes_future():
    preds = pd.DataFrame([
        _pred(date="2026-06-01", expiry="2026-06-01"),   # matured
        _pred(date="2026-06-02", expiry="2026-06-30"),   # not yet (future expiry)
    ])
    pend = pending_predictions(preds, pd.DataFrame(), as_of="2026-06-03")
    assert len(pend) == 1
    assert pend.iloc[0]["expiry"] == "2026-06-01"


def test_pending_excludes_already_graded():
    preds = pd.DataFrame([_pred(date="2026-06-01", expiry="2026-06-01")])
    outs = pd.DataFrame([{
        "pred_date": "2026-06-01", "ticker": "SPY", "expiry": "2026-06-01",
    }])
    pend = pending_predictions(preds, outs, as_of="2026-06-03")
    assert pend.empty


def test_pending_blank_expiry_matures_same_day():
    preds = pd.DataFrame([_pred(date="2026-06-01", expiry="")])
    pend = pending_predictions(preds, pd.DataFrame(), as_of="2026-06-03")
    assert len(pend) == 1


def test_pending_on_empty_predictions_is_empty():
    assert pending_predictions(pd.DataFrame(), pd.DataFrame()).empty


# ── join for the dashboard view ──────────────────────────────────────────────

def test_join_keeps_pending_rows_and_merges_graded():
    preds = pd.DataFrame([
        _pred(date="2026-06-01", expiry="2026-06-01"),
        _pred(date="2026-06-02", expiry="2026-06-02"),
    ])
    outs = pd.DataFrame([{
        "pred_date": "2026-06-01", "ticker": "SPY", "expiry": "2026-06-01",
        "actual_close": 604.0, "close_abs_err": 1.0, "skill": 3.0,
    }])
    merged = join_predictions_outcomes(preds, outs)
    assert len(merged) == 2
    row1 = merged[merged["date"] == "2026-06-01"].iloc[0]
    row2 = merged[merged["date"] == "2026-06-02"].iloc[0]
    assert row1["actual_close"] == 604.0          # graded row carries the outcome
    assert pd.isna(row2["actual_close"])          # still-pending row stays blank


# ── post-close degenerate-forecast guards (the estimated==actual bug) ────────
# The after-close recorder used to log a "forecast" for the SAME day's already-
# expired 0DTE, anchored on the settled close — the grader then scored it
# against that very close (spot == actual, naive error exactly 0). These guards
# make such rows unrecordable, ungradable, and invisible to the scorecard.

from track_record import prediction_is_post_close, drop_degenerate_outcomes


def test_post_close_same_day_prediction_is_flagged():
    # Recorded 13:16 PT on its own expiry day — the close already printed.
    p = _pred(expiry="2026-06-15", pred_time="2026-06-15 13:16:00")
    assert prediction_is_post_close(p) is True


def test_pre_close_same_day_prediction_is_not_flagged():
    # The 6:25am PT pre-open pin records BEFORE the close — a real forecast.
    p = _pred(expiry="2026-06-15", pred_time="2026-06-15 06:25:00")
    assert prediction_is_post_close(p) is False


def test_overnight_prediction_for_next_session_is_not_flagged():
    # Recorded after Monday's close FOR Tuesday's expiry — a real forecast.
    p = _pred(expiry="2026-06-16", pred_time="2026-06-15 13:16:00")
    assert prediction_is_post_close(p) is False


def test_prediction_after_expiry_day_is_flagged():
    p = _pred(expiry="2026-06-15", pred_time="2026-06-16 09:00:00")
    assert prediction_is_post_close(p) is True


def test_legacy_utc_preopen_stamp_is_not_misread_as_post_close():
    # Rows written before the 2026-06-10 cutover are naive UTC: 16:15 UTC is
    # 09:15 PT (pre-open) — it must NOT be flagged as a 4:15pm post-close run.
    p = _pred(expiry="2026-06-09", pred_time="2026-06-09 16:15:37")
    assert prediction_is_post_close(p) is False


def test_legacy_utc_after_close_stamp_is_still_flagged():
    # 23:48 UTC on the expiry day = 16:48 PT — recorded after the close.
    p = _pred(expiry="2026-06-08", pred_time="2026-06-08 23:48:37")
    assert prediction_is_post_close(p) is True


def test_missing_timestamp_is_not_flagged():
    assert prediction_is_post_close(_pred(expiry="2026-06-15")) is False


def test_drop_degenerate_outcomes_removes_self_graded_rows():
    outs = pd.DataFrame([
        # The real ledger rows that exposed the bug: spot == actual, naive 0.
        {"pred_date": "2026-06-05", "ticker": "SPY", "expiry": "2026-06-05",
         "spot_at_pred": 737.55, "estimated_close": 737.55,
         "actual_close": 737.55, "naive_abs_err": 0.0},
        # A legitimate overnight forecast — keep.
        {"pred_date": "2026-06-08", "ticker": "SPY", "expiry": "2026-06-09",
         "spot_at_pred": 739.22, "estimated_close": 739.91,
         "actual_close": 737.55, "naive_abs_err": 1.67},
    ])
    kept = drop_degenerate_outcomes(outs)
    assert len(kept) == 1
    assert kept.iloc[0]["expiry"] == "2026-06-09"


def test_drop_degenerate_keeps_same_day_row_with_pre_close_pred_time():
    # A pre-open call where the close coincidentally landed exactly on spot is
    # a legitimate (and perfect-naive) outcome — pred_time proves it.
    outs = pd.DataFrame([{
        "pred_date": "2026-06-12", "ticker": "SPY", "expiry": "2026-06-12",
        "spot_at_pred": 740.0, "estimated_close": 741.0, "actual_close": 740.0,
        "naive_abs_err": 0.0, "pred_time": "2026-06-12 06:25:00",
    }])
    assert len(drop_degenerate_outcomes(outs)) == 1


def test_drop_degenerate_outcomes_safe_on_empty():
    assert drop_degenerate_outcomes(pd.DataFrame()).empty


def test_graded_outcome_carries_pred_time_and_pacific_graded_at():
    out = grade_prediction(
        _pred(expiry="2026-06-16", timestamp="2026-06-15 13:16:00"),
        actual_close=604.0,
    )
    assert out["pred_time"] == "2026-06-15 13:16:00"
    # graded_at is a parseable wall-clock stamp (Pacific by construction).
    assert pd.to_datetime(out["graded_at"]) is not pd.NaT
