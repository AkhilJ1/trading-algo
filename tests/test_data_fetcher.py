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
