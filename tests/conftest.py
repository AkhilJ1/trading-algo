"""
Shared pytest fixtures and fakes for the data-pipeline test suite.

Everything here is offline and deterministic — no network, no Schwab creds, no
yfinance calls. We exercise the *machinery* (quality gates, fallback routing,
normalization, caching, source/staleness tagging) with controllable fakes so the
tests are fast and reproducible in CI.
"""

import pandas as pd
import pytest


# 7-column option-chain contract every provider must satisfy.
CHAIN_COLUMNS = [
    "strike", "openInterest", "volume",
    "impliedVolatility", "bid", "ask", "lastPrice",
]


def make_chain(n=12, iv=0.20, oi=100, start_strike=400.0):
    """Build a synthetic option-chain DataFrame with `n` rows.

    Defaults produce *usable* rows (iv > 0.05 and oi > 0). Pass iv=0 or oi=0 to
    make every row fail the quality gate.
    """
    rows = []
    for i in range(n):
        strike = start_strike + i
        rows.append({
            "strike": strike,
            "openInterest": oi,
            "volume": 10,
            "impliedVolatility": iv,
            "bid": 1.0,
            "ask": 1.2,
            "lastPrice": 1.1,
        })
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


def make_price_df(closes=(100.0, 101.0, 102.5)):
    """Build a small OHLCV frame on a DatetimeIndex."""
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "Open": list(closes),
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": list(closes),
            "Volume": [1_000_000] * len(closes),
        },
        index=idx,
    )


class FakeProvider:
    """A configurable DataProvider stand-in for fetcher/health tests.

    You hand it the expirations, the (calls, puts) chain, and the price history
    it should return; it records calls so tests can assert routing.
    """

    def __init__(self, name="fake", expirations=None, chain=None, price=None,
                 last_source=None, raise_on=None):
        self.name = name
        self._expirations = expirations if expirations is not None else ["2026-06-19"]
        self._chain = chain
        self._price = price if price is not None else make_price_df()
        self.last_source = last_source if last_source is not None else {}
        self._raise_on = raise_on or set()
        self.calls = []

    def get_expirations(self, ticker):
        self.calls.append(("get_expirations", ticker))
        if "expirations" in self._raise_on:
            raise RuntimeError("boom expirations")
        return list(self._expirations)

    def get_option_chain(self, ticker, expiry):
        self.calls.append(("get_option_chain", ticker, expiry))
        if "chain" in self._raise_on:
            raise RuntimeError("boom chain")
        if self._chain is None:
            return make_chain(), make_chain()
        return self._chain

    def get_price_history(self, ticker, period, interval):
        self.calls.append(("get_price_history", ticker, period, interval))
        if "price" in self._raise_on:
            raise RuntimeError("boom price")
        return self._price


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """Point both fetchers at an isolated temp cache dir."""
    import options_fetcher
    import data_fetcher
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(options_fetcher, "CACHE_DIR", str(cache))
    monkeypatch.setattr(data_fetcher, "CACHE_DIR", str(cache))
    return str(cache)
