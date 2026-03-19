"""
Google Sheets Logger — Persistent prediction and weight tracking.
Uses gspread with service account authentication.
Works on Streamlit Cloud (st.secrets) and locally (env var / JSON file).
Falls back to local CSV if Sheets is unavailable.
"""

import os
from datetime import date, datetime

import pandas as pd

from config import (
    GSHEET_SPREADSHEET_NAME,
    GSHEET_PREDICTIONS_SHEET,
    GSHEET_WEIGHTS_SHEET,
    SIGNAL_WEIGHTS,
)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

PREDICTION_HEADERS = [
    "date", "timestamp", "ticker", "spot_price", "floor", "ceiling",
    "bias", "confidence", "expiry", "vix", "gex_net", "regime",
]
WEIGHT_HEADERS = [
    "date", "weight_name", "old_value", "new_value", "reason",
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
        # Check if headers match; update row 1 if schema changed
        existing = ws.row_values(1)
        if existing != headers:
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


def log_prediction(
    date_str, ticker, spot_price, floor, ceiling,
    bias, confidence, expiry,
    vix=None, gex_net=None, regime=None,
) -> bool:
    """Append one prediction row. Returns True on success."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, GSHEET_PREDICTIONS_SHEET, PREDICTION_HEADERS)
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        row = [
            date_str, now, ticker, round(spot_price, 2),
            round(floor, 2), round(ceiling, 2),
            bias, round(confidence, 1), expiry,
            round(vix, 2) if vix is not None else "",
            round(gex_net, 2) if gex_net is not None else "",
            regime or "",
        ]
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
        row = [
            date.today().isoformat(), weight_name,
            round(old_value, 4), round(new_value, 4), reason,
        ]
        ws.append_row(row, value_input_option="RAW")
        return True
    except Exception as e:
        print(f"[sheets_logger] Error logging weight change: {e}")
        return False


def read_predictions() -> pd.DataFrame:
    """Read all predictions from the sheet. Deduplicates to latest per ticker per day."""
    try:
        ss = get_spreadsheet()
        ws = _ensure_sheet(ss, GSHEET_PREDICTIONS_SHEET, PREDICTION_HEADERS)
        data = ws.get_all_records()
        if not data:
            return pd.DataFrame(columns=PREDICTION_HEADERS)
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for col in ["spot_price", "floor", "ceiling", "confidence", "vix", "gex_net"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Keep only the latest prediction per ticker per day
        if "timestamp" in df.columns and not df["timestamp"].isna().all():
            df = df.sort_values("timestamp").drop_duplicates(
                subset=["date", "ticker"], keep="last"
            ).reset_index(drop=True)
        return df
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

def log_prediction_csv(
    date_str, ticker, spot_price, floor, ceiling,
    bias, confidence, expiry,
    vix=None, gex_net=None, regime=None,
) -> bool:
    """Fallback: log prediction to local CSV."""
    import csv
    pred_file = os.path.join(os.path.dirname(__file__), 'data', 'predictions.csv')
    os.makedirs(os.path.dirname(pred_file), exist_ok=True)
    write_header = not os.path.exists(pred_file)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(pred_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(PREDICTION_HEADERS)
            writer.writerow([
                date_str, now, ticker, round(spot_price, 2),
                round(floor, 2), round(ceiling, 2),
                bias, round(confidence, 1), expiry,
                round(vix, 2) if vix is not None else "",
                round(gex_net, 2) if gex_net is not None else "",
                regime or "",
            ])
        return True
    except Exception:
        return False


def read_predictions_csv() -> pd.DataFrame:
    """Fallback: read predictions from local CSV."""
    pred_file = os.path.join(os.path.dirname(__file__), 'data', 'predictions.csv')
    if os.path.exists(pred_file):
        try:
            df = pd.read_csv(pred_file)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=PREDICTION_HEADERS)
