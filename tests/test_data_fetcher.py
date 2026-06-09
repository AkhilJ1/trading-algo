"""Tests for data_fetcher.fetch_stock_data — the OHLCV path that feeds the
strategies and the daily SPY recorder (Requirement 1). Verifies column
normalization, the per-day CSV cache, and the empty-frame guard."""

import glob
import os

import pandas as pd

import data_fetcher
from tests.conftest import FakeProvider, make_price_df


def _use_provider(monkeypatch, provider):
    monkeypatch.setattr(data_fetcher, "get_provider", lambda *a, **k: provider)


def test_normalizes_columns_and_writes_cache(tmp_cache, monkeypatch):
    prov = FakeProvider(name="schwab", price=make_price_df([10.0, 11.0, 12.0]))
    _use_provider(monkeypatch, prov)
    df = data_fetcher.fetch_stock_data("SPY", "1y", "1d", use_cache=False)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert float(df["Close"].iloc[-1]) == 12.0
    files = glob.glob(os.path.join(tmp_cache, "SPY_1y_1d_*.csv"))
    assert len(files) == 1


def test_second_call_served_from_cache(tmp_cache, monkeypatch):
    hits = {"n": 0}

    class Counting(FakeProvider):
        def get_price_history(self, *a, **k):
            hits["n"] += 1
            return super().get_price_history(*a, **k)

    _use_provider(monkeypatch, Counting(name="schwab", price=make_price_df([10.0, 11.0])))
    a = data_fetcher.fetch_stock_data("SPY", "1y", "1d", use_cache=True)
    b = data_fetcher.fetch_stock_data("SPY", "1y", "1d", use_cache=True)
    assert hits["n"] == 1                      # provider hit once; 2nd read from CSV
    assert float(a["Close"].iloc[-1]) == 11.0
    assert float(b["Close"].iloc[-1]) == 11.0


def test_empty_provider_returns_empty_frame(tmp_cache, monkeypatch):
    _use_provider(monkeypatch, FakeProvider(name="schwab", price=pd.DataFrame()))
    df = data_fetcher.fetch_stock_data("SPY", "1y", "1d", use_cache=False)
    assert df.empty
    # nothing should be cached for an empty result
    assert glob.glob(os.path.join(tmp_cache, "SPY_1y_1d_*.csv")) == []


def test_drops_rows_with_nan_close(tmp_cache, monkeypatch):
    import numpy as np
    price = make_price_df([10.0, 11.0, 12.0])
    price.iloc[-1, price.columns.get_loc("Close")] = np.nan
    _use_provider(monkeypatch, FakeProvider(name="schwab", price=price))
    df = data_fetcher.fetch_stock_data("SPY", "1y", "1d", use_cache=False)
    assert len(df) == 2                        # the NaN-Close row is dropped
    assert float(df["Close"].iloc[-1]) == 11.0


def _one_bar(low, close, when):
    """Single-row OHLCV frame dated `when` (a date) with a controllable low."""
    idx = pd.to_datetime([when.isoformat()])
    return pd.DataFrame(
        {"Open": [close], "High": [close + 1.0], "Low": [low],
         "Close": [close], "Volume": [1_000_000]},
        index=idx,
    )


def test_today_bar_in_cache_is_refetched_not_served_stale(tmp_cache, monkeypatch):
    """Regression: a cache file first written EARLIER today holds today's
    *partial* bar (e.g. an intraday low of 742 the session later blew through,
    real low 738 / close 739). The next call must re-fetch so the current day's
    OHLC is fresh — serving the stale candle made the Fractal chart's low never
    reach the true session low (the 'graph shows 742 but SPY closed 739' bug)."""
    import datetime as dt
    today = dt.date.today()
    hits = {"n": 0}

    class Counting(FakeProvider):
        def get_price_history(self, *a, **k):
            hits["n"] += 1
            # First pull = stale partial bar; re-fetch = settled session bar.
            return (_one_bar(742.0, 745.0, today) if hits["n"] == 1
                    else _one_bar(738.0, 739.0, today))

    _use_provider(monkeypatch, Counting(name="schwab"))

    first = data_fetcher.fetch_stock_data("SPY", "1y", "1d", use_cache=True)
    assert hits["n"] == 1                           # cache miss -> provider
    assert float(first["Low"].iloc[-1]) == 742.0    # stale low written to cache

    second = data_fetcher.fetch_stock_data("SPY", "1y", "1d", use_cache=True)
    assert hits["n"] == 2                            # today-bar -> re-fetched
    assert float(second["Low"].iloc[-1]) == 738.0   # fresh low, not stale 742


def test_prior_day_cache_is_served_without_refetch(tmp_cache, monkeypatch):
    """Boundary: once the newest cached bar is a COMPLETED prior day, it never
    changes, so the cache is served instantly with no extra provider hit."""
    import datetime as dt
    yesterday = dt.date.today() - dt.timedelta(days=1)
    hits = {"n": 0}

    class Counting(FakeProvider):
        def get_price_history(self, *a, **k):
            hits["n"] += 1
            return _one_bar(100.0, 101.0, yesterday)

    _use_provider(monkeypatch, Counting(name="schwab"))
    data_fetcher.fetch_stock_data("SPY", "1y", "1d", use_cache=True)
    data_fetcher.fetch_stock_data("SPY", "1y", "1d", use_cache=True)
    assert hits["n"] == 1                            # 2nd call served from CSV
