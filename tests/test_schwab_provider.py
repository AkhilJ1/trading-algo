"""Tests for the Schwab provider's pure translation logic.

The Schwab module is "dormant" in CI — it can't make live calls without creds —
but its *normalization* is exactly what keeps downstream GEX/IV math correct
(Requirement 3). We exercise every transform offline with a fake Schwab client,
so the percent->decimal IV, the -999 sentinel, the symbol mapping, and the
exp-map flattening are all locked down.
"""

import math
from datetime import date, datetime

import pandas as pd

from providers.schwab_provider import (
    SchwabProvider, _translate_symbol, _period_to_start, _clean_iv,
)

CHAIN_COLUMNS = [
    "strike", "openInterest", "volume",
    "impliedVolatility", "bid", "ask", "lastPrice",
]


def _days_ago(dt):
    return (datetime.now() - dt).total_seconds() / 86400.0


# ── symbol translation ─────────────────────────────────────────────────────
def test_translate_symbol_maps_caret_to_dollar():
    assert _translate_symbol("^VIX") == "$VIX"
    assert _translate_symbol("^vix3m") == "$VIX3M"


def test_translate_symbol_passes_equities_through():
    assert _translate_symbol(" spy ") == "SPY"
    assert _translate_symbol("AAPL") == "AAPL"


# ── period parsing ──────────────────────────────────────────────────────────
def test_period_to_start_units():
    assert abs(_days_ago(_period_to_start("5d")) - 5) < 1
    assert abs(_days_ago(_period_to_start("3wk")) - 21) < 1
    assert abs(_days_ago(_period_to_start("2mo")) - 60) < 1
    assert abs(_days_ago(_period_to_start("1y")) - 365) < 1


def test_period_to_start_unknown_defaults_to_one_year():
    assert abs(_days_ago(_period_to_start("garbage")) - 365) < 1
    assert abs(_days_ago(_period_to_start("")) - 365) < 1


def test_period_to_start_ytd_and_max():
    now = datetime.now()
    ytd = _period_to_start("ytd")
    assert (ytd.year, ytd.month, ytd.day) == (now.year, 1, 1)
    assert _days_ago(_period_to_start("max")) > 365 * 19


# ── IV cleaning ─────────────────────────────────────────────────────────────
def test_clean_iv_percent_to_decimal():
    assert abs(_clean_iv(18.5) - 0.185) < 1e-9


def test_clean_iv_sentinels_and_garbage_become_nan():
    for bad in (-999.0, 0, 999, 1500, "abc", None):
        assert math.isnan(_clean_iv(bad))


# ── date parsing ────────────────────────────────────────────────────────────
def test_parse_date_strips_dte_suffix():
    assert SchwabProvider._parse_date("2026-06-19") == date(2026, 6, 19)
    assert SchwabProvider._parse_date("2026-06-19:3") == date(2026, 6, 19)


# ── exp-map flattening ──────────────────────────────────────────────────────
def test_exp_map_to_df_flattens_filters_and_sorts():
    exp_map = {
        "2026-06-19:3": {
            "420.0": [{
                "strikePrice": 420.0, "openInterest": 100, "totalVolume": 50,
                "volatility": 18.5, "bid": 1.0, "ask": 1.2, "last": 1.1,
            }],
            "410.0": [{
                "strikePrice": 410.0, "openInterest": 7, "totalVolume": 3,
                "volatility": -999.0, "bid": 0.5, "ask": 0.6, "last": 0.55,
            }],
        },
        # A different expiry must be filtered out by the want_date guard.
        "2026-07-17:31": {
            "500.0": [{"strikePrice": 500.0, "volatility": 20.0}],
        },
    }
    df = SchwabProvider._exp_map_to_df(exp_map, "2026-06-19")
    assert list(df.columns) == CHAIN_COLUMNS
    assert list(df["strike"]) == [410.0, 420.0]  # sorted ascending, other expiry dropped
    iv420 = df.loc[df["strike"] == 420.0, "impliedVolatility"].iloc[0]
    iv410 = df.loc[df["strike"] == 410.0, "impliedVolatility"].iloc[0]
    assert abs(iv420 - 0.185) < 1e-9           # 18.5% -> 0.185
    assert math.isnan(iv410)                   # -999 sentinel -> NaN


def test_exp_map_to_df_empty_returns_seven_cols():
    df = SchwabProvider._exp_map_to_df({}, "2026-06-19")
    assert df.empty
    assert list(df.columns) == CHAIN_COLUMNS


# ── client-driven methods (fake Schwab client, no network) ──────────────────
class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, price=None, expirations=None, chain=None, raise_on=None):
        self._price = price
        self._expirations = expirations
        self._chain = chain
        self._raise_on = raise_on or set()

    def get_price_history_every_day(self, symbol, start_datetime=None, end_datetime=None):
        if "price" in self._raise_on:
            raise RuntimeError("boom price")
        return _Resp(self._price)

    def get_option_expiration_chain(self, symbol):
        if "exp" in self._raise_on:
            raise RuntimeError("boom exp")
        return _Resp(self._expirations)

    def get_option_chain(self, symbol, from_date=None, to_date=None):
        if "chain" in self._raise_on:
            raise RuntimeError("boom chain")
        return _Resp(self._chain)


def _provider_with(client):
    p = SchwabProvider()
    p._client = client  # bypass _get_client(): no creds/token file needed
    return p


def test_get_price_history_builds_ohlcv_on_datetime_index():
    candles = {"candles": [
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1000, "datetime": 1735689600000},
        {"open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 2000, "datetime": 1735776000000},
    ]}
    df = _provider_with(_FakeClient(price=candles)).get_price_history("SPY", "5d", "1d")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "Date"
    assert float(df["Close"].iloc[-1]) == 2.0


def test_get_price_history_empty_paths():
    assert _provider_with(_FakeClient(price={"candles": []})).get_price_history("SPY", "5d", "1d").empty
    assert _provider_with(_FakeClient(raise_on={"price"})).get_price_history("SPY", "5d", "1d").empty
    # candles present but missing the datetime column -> empty
    no_dt = {"candles": [{"open": 1, "close": 2}]}
    assert _provider_with(_FakeClient(price=no_dt)).get_price_history("SPY", "5d", "1d").empty


def test_get_expirations_extracts_iso_dates():
    payload = {"expirationList": [
        {"expirationDate": "2026-06-19"},
        {"expirationDate": "2026-07-17T00:00:00Z"},  # truncated to 10 chars
        {"foo": "bar"},                               # no date -> skipped
    ]}
    out = _provider_with(_FakeClient(expirations=payload)).get_expirations("SPY")
    assert out == ["2026-06-19", "2026-07-17"]


def test_get_expirations_empty_on_exception():
    assert _provider_with(_FakeClient(raise_on={"exp"})).get_expirations("SPY") == []


def test_get_option_chain_splits_calls_and_puts():
    chain = {
        "callExpDateMap": {
            "2026-06-19:3": {"420.0": [{
                "strikePrice": 420.0, "volatility": 20.0, "openInterest": 100,
                "totalVolume": 5, "bid": 1, "ask": 2, "last": 1.5,
            }]},
        },
        "putExpDateMap": {
            "2026-06-19:3": {"410.0": [{
                "strikePrice": 410.0, "volatility": 25.0, "openInterest": 80,
                "totalVolume": 4, "bid": 1, "ask": 2, "last": 1.5,
            }]},
        },
    }
    calls, puts = _provider_with(_FakeClient(chain=chain)).get_option_chain("SPY", "2026-06-19")
    assert list(calls["strike"]) == [420.0]
    assert list(puts["strike"]) == [410.0]
    assert abs(calls["impliedVolatility"].iloc[0] - 0.20) < 1e-9


def test_price_history_timestamps_are_naive_eastern():
    """Epoch-ms candles are UTC; the provider contract is NAIVE EASTERN
    (matching yfinance after tz_localize(None)). Naive-UTC stamps sat +4/5h
    off the ET clock, which pushed the afternoon session into the charts'
    overnight rangebreak — the 'chart shows the wrong day's tape' bug on
    Schwab-served intraday views."""
    # 2026-06-09 13:30 UTC == 09:30 ET (the open, EDT) == epoch 1781011800000
    # 2026-06-09 19:55 UTC == 15:55 ET (last RTH 5m bar) == 1781034900000
    candles = {"candles": [
        {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 1,
         "datetime": 1781011800000},
        {"open": 1.5, "high": 2.5, "low": 1, "close": 2.0, "volume": 2,
         "datetime": 1781034900000},
    ]}
    class _C(_FakeClient):
        def get_price_history_every_five_minutes(self, symbol, start_datetime=None, end_datetime=None):
            return _Resp(self._price)
    df = _provider_with(_C(price=candles)).get_price_history("SPY", "2d", "5m")
    assert str(df.index[0]) == "2026-06-09 09:30:00"   # ET open, tz-naive
    assert str(df.index[-1]) == "2026-06-09 15:55:00"  # ET afternoon PRESENT
    assert df.index.tz is None


def test_hourly_interval_maps_to_thirty_minute_endpoint_not_daily():
    """'1h' has no Schwab endpoint; it must use the 30-minute one — falling
    through to DAILY silently broke the Monthly structure view."""
    calls = []
    class _C(_FakeClient):
        def get_price_history_every_thirty_minutes(self, symbol, start_datetime=None, end_datetime=None):
            calls.append("30m")
            return _Resp(self._price)
        def get_price_history_every_day(self, symbol, start_datetime=None, end_datetime=None):
            calls.append("daily")
            return _Resp(self._price)
    candles = {"candles": [{"open": 1, "high": 2, "low": 0.5, "close": 1.5,
                            "volume": 1, "datetime": 1781011800000}]}
    _provider_with(_C(price=candles)).get_price_history("SPY", "1mo", "1h")
    assert calls == ["30m"]


def test_minute_history_requests_extended_hours():
    """Minute endpoints must ask Schwab for pre/post-market candles —
    RTH-only responses left the charts' last candle at 12:55pm PT with no
    closing/after-hours prints. The daily endpoint takes no such kwarg."""
    seen = {}
    candles = {"candles": [{"open": 1, "high": 2, "low": 0.5, "close": 1.5,
                            "volume": 1, "datetime": 1781011800000}]}
    class _C(_FakeClient):
        def get_price_history_every_five_minutes(self, symbol, start_datetime=None,
                                                 end_datetime=None, **kw):
            seen.update(kw)
            return _Resp(self._price)
    _provider_with(_C(price=candles)).get_price_history("SPY", "2d", "5m")
    assert seen.get("need_extended_hours_data") is True


def test_minute_history_tolerates_old_schwab_py_without_kwarg():
    """Older schwab-py without need_extended_hours_data must still work via
    the TypeError fallback (RTH-only, but never a blank chart)."""
    candles = {"candles": [{"open": 1, "high": 2, "low": 0.5, "close": 1.5,
                            "volume": 1, "datetime": 1781011800000}]}
    class _C(_FakeClient):
        def get_price_history_every_five_minutes(self, symbol, start_datetime=None,
                                                 end_datetime=None):
            return _Resp(self._price)   # no extended-hours kwarg accepted
    df = _provider_with(_C(price=candles)).get_price_history("SPY", "2d", "5m")
    assert not df.empty
