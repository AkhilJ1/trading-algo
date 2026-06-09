"""
Track Record — grading the dealer-pin close forecast against reality.

Step 3 gives us a durable, point-in-time forecast ledger (the Predictions
sheet) and a deferred-grading store (the Outcomes sheet). This module is the
pure logic that turns "what we predicted" + "what actually happened" into an
honest, growing scorecard.

The core idea is to never grade a forecast against itself. A dealer-pin
estimated close is only *skillful* if it beats the naive null model — "price
just stays where it was at prediction time (spot)." We therefore record, for
every matured prediction:

  * close_abs_err   = |actual_close − estimated_close|   (how wrong we were)
  * naive_abs_err   = |actual_close − spot_at_pred|      (how wrong "do nothing" was)
  * skill           = naive_abs_err − close_abs_err      (>0 ⇒ we added value)
  * in_range        = floor ≤ actual_close ≤ ceiling     (did the band hold?)
  * dir_correct     = sign(est − spot) == sign(actual − spot)

Everything here is pure and offline: dicts and DataFrames in, dicts and
DataFrames out. The network-touching part (fetching the realized close) lives
in grade_predictions.py so this stays trivially testable.
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd

# Below this absolute move we treat a close as "flat" rather than up/down, so a
# forecast that lands essentially on spot is not scored as a directional call.
_FLAT_EPS = 1e-9


def _num(value):
    """float(value), or None for None / blank / NaN / inf / non-numeric.

    Predictions are read back from Sheets (or CSV) through pandas, which coerces
    a blank cell to NaN — and NaN sails straight through an `is None` check
    (`nan is not None`). If we then `float(nan)` and carry it forward, every
    derived metric becomes NaN and the gspread write fails with
    "Out of range float values are not JSON compliant", rejecting the whole row.
    Normalizing missing/garbage numbers to None here lets the grader treat a
    blank estimated_close or spot as genuinely missing (→ skip) instead of
    silently producing an un-writable, all-NaN outcome.
    """
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _direction(value: float, reference: float) -> str:
    """'up' / 'down' / 'flat' for `value` relative to `reference`."""
    diff = value - reference
    if abs(diff) < _FLAT_EPS:
        return "flat"
    return "up" if diff > 0 else "down"


def grade_prediction(pred: dict, actual_close: float, graded_at: str = None) -> dict:
    """
    Grade one matured prediction against its realized close.

    `pred` uses the Predictions schema (date/ticker/spot_price/floor/ceiling/
    expiry/estimated_close). Returns a dict keyed by the Outcomes schema, ready
    to hand to sheets_logger.log_outcome().

    Raises ValueError if the prediction lacks the fields needed to grade it
    (no estimated_close or no spot) — the caller decides whether to skip.
    """
    spot = _num(pred.get("spot_at_pred", pred.get("spot_price")))
    est = _num(pred.get("estimated_close"))
    if spot is None or est is None:
        raise ValueError("prediction missing spot or estimated_close")

    actual = _num(actual_close)
    if actual is None:
        raise ValueError("realized close is not a finite number")

    floor = _num(pred.get("floor"))
    ceiling = _num(pred.get("ceiling"))

    close_abs_err = abs(actual - est)
    naive_abs_err = abs(actual - spot)
    skill = naive_abs_err - close_abs_err
    close_pct_err = (close_abs_err / actual * 100.0) if actual else float("nan")

    in_range = ""
    if floor is not None and ceiling is not None:
        in_range = bool(floor <= actual <= ceiling)

    dir_pred = _direction(est, spot)
    dir_actual = _direction(actual, spot)

    pred_date = pred.get("pred_date", pred.get("date", ""))
    if isinstance(pred_date, (datetime, pd.Timestamp)):
        pred_date = pred_date.strftime("%Y-%m-%d")

    return {
        "pred_date": str(pred_date)[:10],
        "ticker": pred.get("ticker", ""),
        "expiry": pred.get("expiry", ""),
        "graded_at": graded_at or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "spot_at_pred": round(spot, 2),
        "estimated_close": round(est, 2),
        "floor": round(floor, 2) if floor is not None else "",
        "ceiling": round(ceiling, 2) if ceiling is not None else "",
        "actual_close": round(actual, 2),
        "close_abs_err": round(close_abs_err, 4),
        "close_pct_err": round(close_pct_err, 4) if actual else "",
        "in_range": in_range,
        "dir_predicted": dir_pred,
        "dir_actual": dir_actual,
        "dir_correct": bool(dir_pred == dir_actual),
        "naive_abs_err": round(naive_abs_err, 4),
        "skill": round(skill, 4),
    }


def _outcome_key(pred_date, ticker, expiry) -> tuple:
    """Stable identity for a graded forecast (one per pred_date/ticker/expiry)."""
    d = str(pred_date)[:10]
    return (d, str(ticker).upper(), str(expiry))


def _naive_day(value) -> pd.Timestamp:
    """Parse to a tz-naive, midnight-normalized Timestamp (NaT on failure)."""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ts
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def pending_predictions(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    as_of=None,
) -> pd.DataFrame:
    """
    Predictions that have *matured* (expiry on/before `as_of`) and have not yet
    been graded (no matching row in `outcomes`). These are what the grading
    runner needs to fetch realized closes for.

    `as_of` defaults to today. A prediction with a blank/unparseable expiry is
    treated as same-day and matures immediately.
    """
    if predictions is None or predictions.empty:
        return pd.DataFrame(columns=getattr(predictions, "columns", None))

    as_of = _naive_day(as_of) if as_of is not None else _naive_day(pd.Timestamp.utcnow())

    graded = set()
    if outcomes is not None and not outcomes.empty:
        for _, o in outcomes.iterrows():
            graded.add(_outcome_key(o.get("pred_date"), o.get("ticker"), o.get("expiry")))

    rows = []
    for _, p in predictions.iterrows():
        pred_date = p.get("date", p.get("pred_date"))
        expiry_raw = p.get("expiry", "")
        # Mature when the expiry day is in the past relative to as_of. A blank
        # expiry falls back to the prediction date (same-day forecast).
        exp = _naive_day(expiry_raw)
        if pd.isna(exp):
            exp = _naive_day(pred_date)
        if pd.isna(exp) or exp > as_of:
            continue
        if _outcome_key(pred_date, p.get("ticker"), expiry_raw) in graded:
            continue
        rows.append(p)

    if not rows:
        return predictions.iloc[0:0]
    return pd.DataFrame(rows).reset_index(drop=True)


def summarize_track_record(outcomes: pd.DataFrame) -> dict:
    """
    Aggregate a graded-outcomes frame into headline track-record stats.

    Returns zeros/None on an empty frame rather than raising, so the dashboard
    can call it before any forecast has matured.
    """
    cols = ["close_abs_err", "naive_abs_err", "skill", "close_pct_err"]
    empty = {
        "n_graded": 0,
        "mean_abs_err": None, "median_abs_err": None, "mean_pct_err": None,
        "naive_mean_abs_err": None, "mean_skill": None,
        "skill_rate": None, "beats_naive": None,
        "in_range_rate": None, "dir_accuracy": None,
    }
    if outcomes is None or outcomes.empty:
        return empty

    df = outcomes.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close_abs_err"])
    n = len(df)
    if n == 0:
        return empty

    mean_abs = float(df["close_abs_err"].mean())
    naive_mean = float(df["naive_abs_err"].mean()) if "naive_abs_err" in df else float("nan")
    mean_skill = float(df["skill"].mean()) if "skill" in df else float("nan")
    skill_rate = float((df["skill"] > 0).mean()) if "skill" in df else float("nan")

    in_range_rate = _bool_rate(df.get("in_range"))
    dir_accuracy = _bool_rate(df.get("dir_correct"))

    return {
        "n_graded": n,
        "mean_abs_err": round(mean_abs, 4),
        "median_abs_err": round(float(df["close_abs_err"].median()), 4),
        "mean_pct_err": round(float(df["close_pct_err"].mean()), 4)
            if "close_pct_err" in df and df["close_pct_err"].notna().any() else None,
        "naive_mean_abs_err": round(naive_mean, 4) if not np.isnan(naive_mean) else None,
        "mean_skill": round(mean_skill, 4) if not np.isnan(mean_skill) else None,
        "skill_rate": round(skill_rate, 4) if not np.isnan(skill_rate) else None,
        "beats_naive": bool(mean_abs < naive_mean) if not np.isnan(naive_mean) else None,
        "in_range_rate": round(in_range_rate, 4) if in_range_rate is not None else None,
        "dir_accuracy": round(dir_accuracy, 4) if dir_accuracy is not None else None,
    }


def _bool_rate(series) -> float:
    """Fraction of truthy values in a column that may hold bools or 'TRUE'/'FALSE' strings."""
    if series is None:
        return None
    s = series.dropna()
    s = s[s.astype(str).str.strip() != ""]
    if s.empty:
        return None
    truthy = s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])
    return float(truthy.mean())


def join_predictions_outcomes(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join predictions to their grades on (date, ticker, expiry) for the
    dashboard's track-record view. Ungraded (still-pending) predictions keep
    their forecast columns with blank outcome columns.
    """
    if predictions is None or predictions.empty:
        return pd.DataFrame()

    preds = predictions.copy()
    preds["_key"] = [
        _outcome_key(r.get("date", r.get("pred_date")), r.get("ticker"), r.get("expiry"))
        for _, r in preds.iterrows()
    ]

    if outcomes is None or outcomes.empty:
        merged = preds
    else:
        outs = outcomes.copy()
        outs["_key"] = [
            _outcome_key(r.get("pred_date"), r.get("ticker"), r.get("expiry"))
            for _, r in outs.iterrows()
        ]
        outcome_only = [c for c in outs.columns if c not in preds.columns or c == "_key"]
        merged = preds.merge(outs[outcome_only], on="_key", how="left")

    return merged.drop(columns=["_key"], errors="ignore")
