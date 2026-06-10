"""
Google Sheets Logger — Persistent prediction and weight tracking.
Uses gspread with service account authentication.
Works on Streamlit Cloud (st.secrets) and locally (env var / JSON file).
Falls back to local CSV if Sheets is unavailable.
"""

import math
import os
from datetime import date, datetime

import pandas as pd

from config import (
    GSHEET_SPREADSHEET_NAME,
    GSHEET_PREDICTIONS_SHEET,
    GSHEET_WEIGHTS_SHEET,
    GSHEET_OUTCOMES_SHEET,
    GSHEET_CALIBRATION_SHEET,
    GSHEET_PIN_FORECASTS_SHEET,
    GSHEET_PIN_OUTCOMES_SHEET,
    GSHEET_LEVELS_SHEET,
    GSHEET_FEATURES_SHEET,
    SIGNAL_WEIGHTS,
)

# All ledger timestamps are recorded in US Pacific time (the user's clock).
# They used to be naive UTC, which made a 6:25am PT pre-open run read as a
# mid-day mystery and left the dashboard showing dates with a 00:00:00 time.
try:
    from zoneinfo import ZoneInfo
    _PT = ZoneInfo("America/Los_Angeles")
except Exception:  # pragma: no cover - zoneinfo ships with the py3.9+ runtime
    _PT = None


def pacific_now() -> datetime:
    """Current time in US Pacific (naive UTC if zoneinfo is unavailable)."""
    return datetime.now(_PT) if _PT is not None else datetime.utcnow()

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Point-in-time forecast snapshot. The first 12 columns are the original
# schema; estimated_close / pin_target / max_pain were appended (item 4) so the
# durable record captures the dealer-pin close forecast we later grade. New
# columns are append-only and backward compatible — old rows simply read blank
# for the trailing fields, and _ensure_sheet() rewrites row 1 on schema change.
PREDICTION_HEADERS = [
    "date", "timestamp", "ticker", "spot_price", "floor", "ceiling",
    "bias", "confidence", "expiry", "vix", "gex_net", "regime",
    "estimated_close", "pin_target", "max_pain",
    # Provenance (appended so existing rows stay aligned; _ensure_sheet updates
    # the header on the next write):
    #   spot_source  — 'live_override' (live pre/post-market anchor) or
    #                  'daily_close' (settled close). This is the durable proof
    #                  that a morning pin was anchored on a LIVE price, not a
    #                  stale close.
    #   chain_source — which backend actually served the option chain this row
    #                  was built from ('schwab' or 'yfinance'), so the ledger
    #                  itself shows whether Schwab or the fallback was used.
    "spot_source", "chain_source",
]
WEIGHT_HEADERS = [
    "date", "weight_name", "old_value", "new_value", "reason",
]
# Deferred grading of matured forecasts: one row per prediction once its
# realized close is known. Joining Predictions↔Outcomes on (date, ticker)
# yields the scored, point-in-time-honest track record.
OUTCOME_HEADERS = [
    "pred_date", "ticker", "expiry", "graded_at",
    "spot_at_pred", "estimated_close", "floor", "ceiling",
    "actual_close", "close_abs_err", "close_pct_err", "in_range",
    "dir_predicted", "dir_actual", "dir_correct",
    "naive_abs_err", "skill",
    # Appended (backward compatible — old rows read blank): the Pacific-time
    # moment the FORECAST was recorded, copied from the prediction row so the
    # scorecard can show when the call was actually made (e.g. 06:25 pre-open)
    # instead of a bare date that renders as 00:00:00.
    "pred_time",
]
# Periodic, point-in-time-honest calibration snapshot of the evidence-based
# range engine (range_calibration.py): how often each confidence band actually
# contained the realized next-session close over a trailing window, plus the
# out-of-sample width verdict. One row per scheduled run → a calibration drift
# time series the dashboard / a human can watch without any local machine.
CALIBRATION_HEADERS = [
    "date", "ticker", "window", "n_days",
    "cov_1sigma", "cov_1_5sigma", "cov_2sigma",
    "calibration_error", "mean_width_pct",
    "best_width", "baseline_test_error", "proposed_test_error", "improved",
]

# Module-level cache
_client = None
_spreadsheet = None


def _get_credentials():
    """Load service account credentials from Streamlit secrets or env var."""
    from google.oauth2.service_account import Credentials

    # Try Streamlit secrets first (works on Cloud)
    try:
        import streamlit as st
        creds_dict = dict(st.secrets["gcp_service_account"])
        return Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    except Exception:
        pass

    # Fallback to JSON file
    creds_path = os.environ.get("GOOGLE_SHEETS_CREDS", "")
    if creds_path and os.path.exists(creds_path):
        return Credentials.from_service_account_file(creds_path, scopes=SCOPES)

    raise RuntimeError(
        "No Google credentials found. Set st.secrets['gcp_service_account'] "
        "or GOOGLE_SHEETS_CREDS env var pointing to a service account JSON."
    )


def get_client():
    """Authorized gspread client (cached at module level)."""
    global _client
    if _client is None:
        import gspread
        _client = gspread.authorize(_get_credentials())
    return _client


def get_spreadsheet():
    """Open (or create) the main spreadsheet."""
    global _spreadsheet
    if _spreadsheet is not None:
        return _spreadsheet
    import gspread
    client = get_client()
    try:
        _spreadsheet = client.open(GSHEET_SPREADSHEET_NAME)
    except gspread.exceptions.SpreadsheetNotFound:
        _spreadsheet = client.create(GSHEET_SPREADSHEET_NAME)
    return _spreadsheet


def _ensure_sheet(spreadsheet, title, headers):
    """Get or create a worksheet with the given headers. Updates headers if they changed."""
    import gspread
    try:
        ws = spreadsheet.worksheet(title)
        # Check if headers match; update row 1 if schema changed. Grow the
        # grid first when columns were appended to the schema — sheets created
        # with cols=len(old_headers) reject writes past their last column
        # ("exceeds grid limits") instead of auto-expanding.
        existing = ws.row_values(1)
        if existing != headers:
            if len(headers) > ws.col_count:
                ws.add_cols(len(headers) - ws.col_count)
            ws.update("A1", [headers], value_input_option="RAW")
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=1000, cols=len(headers))
        ws.append_row(headers, value_input_option="RAW")
    return ws


def is_sheets_available() -> bool:
    """Check if Google Sheets credentials are configured."""
    try:
        _get_credentials()
        return True
    except Exception:
        return False


def _num_or_blank(value, ndigits=2):
    """Round a number for sheet storage, or '' if it is None/non-numeric/non-finite."""
    try:
        if value is None:
            return ""
        f = float(value)
        return round(f, ndigits) if math.isfinite(f) else ""
    except (TypeError, ValueError):
        return ""


def _cell_safe(value):
    """Make one cell safe to write through gspread (value_input_option='RAW').

    gspread serializes the row to JSON, and JSON has no NaN/Infinity — so a
    single non-finite float raises "Out of range float values are not JSON
    compliant" and rejects the ENTIRE row (this silently broke every graded
    Outcomes write). Coerce non-finite numbers to a blank cell; leave strings,
    bools, ints and finite floats untouched.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value  # None, "", or a normal string — write as-is
    return value if math.isfinite(f) else ""


def _json_safe(row):
    """Sanitize a whole row so a stray NaN/inf can never reject the write."""
    return [_cell_safe(v) for v in row]


def _prediction_row(
    date_str, ticker, spot_price, floor, ceiling,
    bias, confidence, expiry,
    vix=None, gex_net=None, regime=None,
    estimated_close=None, pin_target=None, max_pain=None,
    spot_source="", chain_source="",
):
    """Build a Predictions row in PREDICTION_HEADERS order (shared by sheet/CSV)."""
    now = pacific_now().strftime("%Y-%m-%d %H:%M:%S")
    return _json_safe([
        date_str, now, ticker, _num_or_blank(spot_price),
        _num_or_blank(floor), _num_or_blank(ceiling),
        bias, _num_or_blank(confidence, 1), expiry,
        _num_or_blank(vix), _num_or_blank(gex_net), regime or "",
        _num_or_blank(estimated_close), _num_or_blank(pin_target),
        _num_or_blank(max_pain),
        spot_source or "", chain_source or "",
    ])


def log_prediction(
    date_str, ticker, spot_price, floor, ceiling,
    bias, confidence, expiry,
    vix=None, gex_net=None, regime=None,
    estimated_close=None, pin_target=None, max_pain=None,
    spot_source="", chain_source="",
    sheet=GSHEET_PREDICTIONS_SHEET,
) -> bool:
    """Append one prediction row. Returns True on success.

    `sheet` selects the destination worksheet so the same point-in-time forecast
    schema can back both the after-close Predictions ledger and the dedicated
    pre-open PinForecasts ledger without code duplication.
    """
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, sheet, PREDICTION_HEADERS)
        row = _prediction_row(
            date_str, ticker, spot_price, floor, ceiling,
            bias, confidence, expiry, vix, gex_net, regime,
            estimated_close, pin_target, max_pain,
            spot_source, chain_source,
        )
        ws.append_row(row, value_input_option="RAW")
        return True
    except Exception as e:
        print(f"[sheets_logger] Error logging prediction: {e}")
        return False


def log_weight_change(weight_name, old_value, new_value, reason) -> bool:
    """Append one weight-change row."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, GSHEET_WEIGHTS_SHEET, WEIGHT_HEADERS)
        row = _json_safe([
            date.today().isoformat(), weight_name,
            _num_or_blank(old_value, 4), _num_or_blank(new_value, 4), reason,
        ])
        ws.append_row(row, value_input_option="RAW")
        return True
    except Exception as e:
        print(f"[sheets_logger] Error logging weight change: {e}")
        return False


def _dedupe_latest_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the latest prediction per (day, ticker, EXPIRY).

    Expiry must be part of the identity. With (date, ticker) alone, the 1:16pm
    recorder's new overnight row (expiry = next session) silently SHADOWED the
    previous evening's still-ungraded forecast for today's expiry — same date
    key, later timestamp — so the matured forecast vanished from the grader's
    view and the dealer-pin track record never accumulated a single legit row.
    Outcomes are keyed by (pred_date, ticker, expiry); reads must match.
    """
    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        subset = ["date", "ticker"] + (["expiry"] if "expiry" in df.columns else [])
        df = df.sort_values("timestamp").drop_duplicates(
            subset=subset, keep="last"
        ).reset_index(drop=True)
    return df


def read_predictions(sheet=GSHEET_PREDICTIONS_SHEET) -> pd.DataFrame:
    """Read all predictions from `sheet`. Deduplicates to latest per ticker per day per expiry."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, sheet, PREDICTION_HEADERS)
        data = ws.get_all_records()
        if not data:
            return pd.DataFrame(columns=PREDICTION_HEADERS)
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for col in ["spot_price", "floor", "ceiling", "confidence", "vix",
                    "gex_net", "estimated_close", "pin_target", "max_pain"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return _dedupe_latest_predictions(df)
    except Exception as e:
        print(f"[sheets_logger] Error reading predictions: {e}")
        return pd.DataFrame(columns=PREDICTION_HEADERS)


def read_weight_history() -> pd.DataFrame:
    """Read weight change history from the sheet."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, GSHEET_WEIGHTS_SHEET, WEIGHT_HEADERS)
        data = ws.get_all_records()
        if not data:
            return pd.DataFrame(columns=WEIGHT_HEADERS)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"[sheets_logger] Error reading weight history: {e}")
        return pd.DataFrame(columns=WEIGHT_HEADERS)


def get_current_weights() -> dict:
    """
    Get the currently active weights.
    Reads weight history and applies all changes to the baseline.
    If no changes exist or Sheets is unavailable, returns baseline from config.
    """
    weights = dict(SIGNAL_WEIGHTS)

    try:
        history = read_weight_history()
        if history.empty:
            return weights
        for _, row in history.iterrows():
            name = row.get("weight_name", "")
            if name in weights:
                try:
                    weights[name] = float(row["new_value"])
                except (ValueError, TypeError):
                    pass
    except Exception:
        pass

    return weights


# ── CSV Fallback ──────────────────────────────────────────────────────────

def _data_path(filename):
    return os.path.join(os.path.dirname(__file__), 'data', filename)


def _append_csv(path, headers, row) -> bool:
    """Append one row to a CSV, writing the header first if the file is new."""
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    try:
        with open(path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(headers)
            writer.writerow(row)
        return True
    except Exception:
        return False


def log_prediction_csv(
    date_str, ticker, spot_price, floor, ceiling,
    bias, confidence, expiry,
    vix=None, gex_net=None, regime=None,
    estimated_close=None, pin_target=None, max_pain=None,
    spot_source="", chain_source="",
    path_name='predictions.csv',
) -> bool:
    """Fallback: log prediction to a local CSV (`path_name` under data/)."""
    row = _prediction_row(
        date_str, ticker, spot_price, floor, ceiling,
        bias, confidence, expiry, vix, gex_net, regime,
        estimated_close, pin_target, max_pain,
        spot_source, chain_source,
    )
    return _append_csv(_data_path(path_name), PREDICTION_HEADERS, row)


def read_predictions_csv(path_name='predictions.csv') -> pd.DataFrame:
    """Fallback: read predictions from a local CSV (`path_name` under data/)."""
    pred_file = _data_path(path_name)
    if os.path.exists(pred_file):
        try:
            df = pd.read_csv(pred_file)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=PREDICTION_HEADERS)


# ── Outcomes (graded forecasts) ───────────────────────────────────────────

def log_outcome(outcome: dict, sheet=GSHEET_OUTCOMES_SHEET) -> bool:
    """Append one graded-outcome row (keys = OUTCOME_HEADERS) to `sheet`."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, sheet, OUTCOME_HEADERS)
        ws.append_row(_outcome_row(outcome), value_input_option="RAW")
        return True
    except Exception as e:
        print(f"[sheets_logger] Error logging outcome: {e}")
        return False


def log_outcome_csv(outcome: dict, path_name='outcomes.csv') -> bool:
    """Fallback: append one graded-outcome row to a local CSV (`path_name`)."""
    return _append_csv(_data_path(path_name), OUTCOME_HEADERS, _outcome_row(outcome))


def _outcome_row(outcome: dict):
    """Serialize an outcome dict into OUTCOME_HEADERS order (NaN/inf → blank)."""
    return _json_safe([outcome.get(h, "") for h in OUTCOME_HEADERS])


def read_outcomes(sheet=GSHEET_OUTCOMES_SHEET, path_name='outcomes.csv') -> pd.DataFrame:
    """Read graded outcomes from `sheet` (or local CSV when Sheets is absent)."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, sheet, OUTCOME_HEADERS)
        data = ws.get_all_records()
        if not data:
            return pd.DataFrame(columns=OUTCOME_HEADERS)
        return _coerce_outcomes(pd.DataFrame(data))
    except Exception as e:
        print(f"[sheets_logger] Error reading outcomes: {e}")
        return read_outcomes_csv(path_name=path_name)


def read_outcomes_csv(path_name='outcomes.csv') -> pd.DataFrame:
    """Fallback: read graded outcomes from a local CSV (`path_name` under data/)."""
    path = _data_path(path_name)
    if os.path.exists(path):
        try:
            return _coerce_outcomes(pd.read_csv(path))
        except Exception:
            pass
    return pd.DataFrame(columns=OUTCOME_HEADERS)


def _coerce_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes on an outcomes frame."""
    if df.empty:
        return df
    if "pred_date" in df.columns:
        df["pred_date"] = pd.to_datetime(df["pred_date"], errors="coerce")
    for col in ["spot_at_pred", "estimated_close", "floor", "ceiling",
                "actual_close", "close_abs_err", "close_pct_err",
                "naive_abs_err", "skill"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Calibration snapshots (range-engine drift tracking) ───────────────────

def _calibration_row(snapshot: dict):
    """Serialize a calibration snapshot dict into CALIBRATION_HEADERS order (NaN/inf → blank)."""
    return _json_safe([snapshot.get(h, "") for h in CALIBRATION_HEADERS])


def log_calibration(snapshot: dict) -> bool:
    """Append one calibration snapshot row (keys = CALIBRATION_HEADERS) to the sheet."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, GSHEET_CALIBRATION_SHEET, CALIBRATION_HEADERS)
        ws.append_row(_calibration_row(snapshot), value_input_option="RAW")
        return True
    except Exception as e:
        print(f"[sheets_logger] Error logging calibration: {e}")
        return False


def log_calibration_csv(snapshot: dict) -> bool:
    """Fallback: append one calibration snapshot row to local CSV."""
    return _append_csv(_data_path('calibration.csv'), CALIBRATION_HEADERS, _calibration_row(snapshot))


def read_calibration() -> pd.DataFrame:
    """Read calibration snapshots from the sheet (or local CSV when Sheets is absent)."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, GSHEET_CALIBRATION_SHEET, CALIBRATION_HEADERS)
        data = ws.get_all_records()
        if not data:
            return pd.DataFrame(columns=CALIBRATION_HEADERS)
        return _coerce_calibration(pd.DataFrame(data))
    except Exception as e:
        print(f"[sheets_logger] Error reading calibration: {e}")
        return read_calibration_csv()


def read_calibration_csv() -> pd.DataFrame:
    """Fallback: read calibration snapshots from local CSV."""
    path = _data_path('calibration.csv')
    if os.path.exists(path):
        try:
            return _coerce_calibration(pd.read_csv(path))
        except Exception:
            pass
    return pd.DataFrame(columns=CALIBRATION_HEADERS)


def _coerce_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes on a calibration frame."""
    if df.empty:
        return df
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["n_days", "cov_1sigma", "cov_1_5sigma", "cov_2sigma",
                "calibration_error", "mean_width_pct", "best_width",
                "baseline_test_error", "proposed_test_error"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Pre-open dealer-pin track record ──────────────────────────────────────
# Thin wrappers over the (now sheet/path-parameterized) prediction + outcome
# machinery, pointed at the dedicated PinForecasts / PinOutcomes stores. The
# pre-open pin is a point-in-time forecast just like the after-close one — same
# schema, same grader — it just lives in its own ledger so the two never
# collide on the same SPY date and the pin gets a clean scorecard.

def log_pin_forecast(**kwargs) -> bool:
    """Append one pre-open pin forecast row to the PinForecasts sheet."""
    return log_prediction(sheet=GSHEET_PIN_FORECASTS_SHEET, **kwargs)


def log_pin_forecast_csv(**kwargs) -> bool:
    """Fallback: append one pre-open pin forecast row to data/pin_forecasts.csv."""
    return log_prediction_csv(path_name='pin_forecasts.csv', **kwargs)


def read_pin_forecasts() -> pd.DataFrame:
    """Read pre-open pin forecasts (latest per ticker per day) from the sheet."""
    return read_predictions(sheet=GSHEET_PIN_FORECASTS_SHEET)


def read_pin_forecasts_csv() -> pd.DataFrame:
    """Fallback: read pre-open pin forecasts from data/pin_forecasts.csv."""
    return read_predictions_csv(path_name='pin_forecasts.csv')


def log_pin_outcome(outcome: dict) -> bool:
    """Append one graded pin outcome row to the PinOutcomes sheet."""
    return log_outcome(outcome, sheet=GSHEET_PIN_OUTCOMES_SHEET)


def log_pin_outcome_csv(outcome: dict) -> bool:
    """Fallback: append one graded pin outcome row to data/pin_outcomes.csv."""
    return log_outcome_csv(outcome, path_name='pin_outcomes.csv')


def read_pin_outcomes() -> pd.DataFrame:
    """Read graded pin outcomes from the sheet (or local CSV when absent)."""
    return read_outcomes(sheet=GSHEET_PIN_OUTCOMES_SHEET, path_name='pin_outcomes.csv')


def read_pin_outcomes_csv() -> pd.DataFrame:
    """Fallback: read graded pin outcomes from data/pin_outcomes.csv."""
    return read_outcomes_csv(path_name='pin_outcomes.csv')


# ── Levels history (floor/ceiling per analysis run) ───────────────────────
# Append-only: one row per analysis run (the scheduled recorders and every
# dashboard Analyze). The chart reads this back to draw EACH run's floor and
# ceiling as its own line, so the levels visibly migrate through the day
# instead of the latest run silently replacing the previous lines.

LEVELS_HEADERS = [
    "date", "timestamp", "ticker", "spot",
    "floor", "ceiling", "floor2", "ceiling2",
    "est_close", "source",
]


def _levels_row(ticker, spot, floor, ceiling,
                floor2=None, ceiling2=None, est_close=None, source=""):
    now = pacific_now()
    return _json_safe([
        now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d %H:%M:%S"),
        str(ticker).upper(), _num_or_blank(spot),
        _num_or_blank(floor), _num_or_blank(ceiling),
        _num_or_blank(floor2), _num_or_blank(ceiling2),
        _num_or_blank(est_close), source or "",
    ])


def log_levels_snapshot(ticker, spot, floor, ceiling,
                        floor2=None, ceiling2=None, est_close=None,
                        source="") -> bool:
    """Append one levels row to the LevelsHistory sheet. Returns True on success."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, GSHEET_LEVELS_SHEET, LEVELS_HEADERS)
        ws.append_row(
            _levels_row(ticker, spot, floor, ceiling, floor2, ceiling2,
                        est_close, source),
            value_input_option="RAW",
        )
        return True
    except Exception as e:
        print(f"[sheets_logger] Error logging levels snapshot: {e}")
        return False


def log_levels_snapshot_csv(ticker, spot, floor, ceiling,
                            floor2=None, ceiling2=None, est_close=None,
                            source="", path_name="levels_history.csv") -> bool:
    """Fallback: append one levels row to data/levels_history.csv."""
    row = _levels_row(ticker, spot, floor, ceiling, floor2, ceiling2,
                      est_close, source)
    return _append_csv(_data_path(path_name), LEVELS_HEADERS, row)


def _coerce_levels(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    for col in ["spot", "floor", "ceiling", "floor2", "ceiling2", "est_close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_levels_history() -> pd.DataFrame:
    """Read the levels history from the sheet (or local CSV when absent)."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, GSHEET_LEVELS_SHEET, LEVELS_HEADERS)
        data = ws.get_all_records()
        if not data:
            return pd.DataFrame(columns=LEVELS_HEADERS)
        return _coerce_levels(pd.DataFrame(data))
    except Exception as e:
        print(f"[sheets_logger] Error reading levels history: {e}")
        return read_levels_history_csv()


def read_levels_history_csv(path_name="levels_history.csv") -> pd.DataFrame:
    """Fallback: read the levels history from data/levels_history.csv."""
    path = _data_path(path_name)
    if os.path.exists(path):
        try:
            return _coerce_levels(pd.read_csv(path))
        except Exception:
            pass
    return pd.DataFrame(columns=LEVELS_HEADERS)


# ── Feature log (ML-ready point-in-time snapshots) ────────────────────────
# One wide row per scheduled run; see feature_log.py for what each block of
# columns means and the research behind it. date/timestamp are Pacific.

FEATURE_HEADERS = [
    "date", "timestamp", "session", "ticker", "expiry", "dte",
    "spot", "spot_source", "chain_source", "stale",
    "floor", "ceiling", "bias", "confidence", "est_close", "pin_target",
    "gex_net", "gamma_strength", "gamma_pin_strike", "pull_fraction",
    "gex_floor", "gex_ceiling", "max_pain", "dist_max_pain_pct",
    "call_wall", "put_wall", "pcr_oi", "pcr_volume", "skew_ratio",
    "iv_used", "daily_em", "em_1sigma", "vrp_pct", "parkinson_rv",
    "vix", "vix_3m", "vix_term_ratio", "vix_regime",
    "fractal_dim", "regime",
    "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
    "mom_1m_sign", "mom_3m_sign", "mom_6m_sign", "mom_12m_sign", "rsi14",
    "prior_close", "gap_pct", "ovn_high", "ovn_low", "ovn_range_pct",
    "bar_date", "bar_close", "bar_ret_pct", "bar_range_pct", "bar_clv",
    "bar_volume", "vol_vs_20d", "range_vs_em",
]


def _feature_row_list(row: dict):
    now = pacific_now()
    stamped = dict(row)
    stamped.setdefault("date", now.strftime("%Y-%m-%d"))
    stamped.setdefault("timestamp", now.strftime("%Y-%m-%d %H:%M:%S"))
    return _json_safe([
        "" if stamped.get(h) is None else stamped.get(h, "")
        for h in FEATURE_HEADERS
    ])


def log_feature_row(row: dict) -> bool:
    """Append one feature row to the FeatureLog sheet. Returns True on success."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, GSHEET_FEATURES_SHEET, FEATURE_HEADERS)
        ws.append_row(_feature_row_list(row), value_input_option="RAW")
        return True
    except Exception as e:
        print(f"[sheets_logger] Error logging feature row: {e}")
        return False


def log_feature_row_csv(row: dict, path_name="feature_log.csv") -> bool:
    """Fallback: append one feature row to data/feature_log.csv."""
    return _append_csv(_data_path(path_name), FEATURE_HEADERS, _feature_row_list(row))


def read_feature_log() -> pd.DataFrame:
    """Read the feature log from the sheet (or local CSV when absent)."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, GSHEET_FEATURES_SHEET, FEATURE_HEADERS)
        data = ws.get_all_records()
        if not data:
            return pd.DataFrame(columns=FEATURE_HEADERS)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"[sheets_logger] Error reading feature log: {e}")
        path = _data_path("feature_log.csv")
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                pass
        return pd.DataFrame(columns=FEATURE_HEADERS)
