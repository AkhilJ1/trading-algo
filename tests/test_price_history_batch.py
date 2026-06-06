"""Tests for get_price_history_batch — the multi-symbol fetch the screener uses.

The screener used to call yfinance's batch download directly, bypassing the
provider abstraction. It now goes through get_price_history_batch so that with
DATA_PROVIDER=schwab the universe is pulled from Schwab (degrading to yfinance
per symbol), while the yfinance backend keeps its fast one-shot download.

Three contracts are exercised offline:
  * the base default loops get_price_history per symbol and omits misses,
  * the yfinance override fans a single yf.download out into per-symbol frames,
  * the FallbackProvider (inherited default) routes Schwab -> yfinance per symbol.
"""

import pandas as pd

from providers.base import DataProvider
from providers.fallback_provider import FallbackProvider
from providers.yfinance_provider import YFinanceProvider
from tests.conftest import FakeProvider, make_price_df


# ── base default (per-symbol loop) ───────────────────────────────────────────

def test_base_default_loops_and_returns_per_symbol():
    prov = FakeProvider(name="schwab", price=make_price_df([10.0, 11.0]))
    out = prov.get_price_history_batch(["AAA", "BBB"], "1y", "1wk")
    assert set(out) == {"AAA", "BBB"}
    assert float(out["AAA"]["Close"].iloc[-1]) == 11.0
    # One per-symbol fetch per ticker (no hidden batch endpoint).
    price_calls = [c for c in prov.calls if c[0] == "get_price_history"]
    assert [c[1] for c in price_calls] == ["AAA", "BBB"]


def test_base_default_omits_symbols_with_no_data():
    # A backend that has some symbols and not others: missing ones drop out
    # instead of sinking the whole batch.
    class PartialProvider(DataProvider):
        name = "partial"
        def get_price_history(self, ticker, period, interval):
            if ticker == "GOOD":
                return make_price_df([5.0, 6.0])
            if ticker == "EMPTY":
                return pd.DataFrame()
            raise RuntimeError("boom")
        def get_expirations(self, ticker):
            return []
        def get_option_chain(self, ticker, expiry):
            return pd.DataFrame(), pd.DataFrame()

    out = PartialProvider().get_price_history_batch(
        ["GOOD", "EMPTY", "RAISES"], "1y", "1wk"
    )
    assert set(out) == {"GOOD"}


def test_base_default_empty_universe():
    assert FakeProvider().get_price_history_batch([], "1y", "1wk") == {}


# ── yfinance override (one download, fanned out per symbol) ───────────────────

def test_yfinance_override_splits_multiindex(monkeypatch):
    import providers.yfinance_provider as yp

    idx = pd.date_range("2026-01-01", periods=3, freq="W")
    cols = pd.MultiIndex.from_product(
        [["AAA", "BBB"], ["Open", "High", "Low", "Close", "Volume"]]
    )
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    raw[("BBB", "Close")] = [7.0, 8.0, 9.0]

    seen = {}
    def fake_download(syms, **kw):
        seen["syms"] = syms
        return raw
    monkeypatch.setattr(yp.yf, "download", fake_download)

    out = YFinanceProvider().get_price_history_batch(["AAA", "BBB"], "1y", "1wk")
    assert set(out) == {"AAA", "BBB"}
    assert float(out["BBB"]["Close"].iloc[-1]) == 9.0
    assert seen["syms"] == ["AAA", "BBB"]   # one call, not two


def test_yfinance_override_empty_universe(monkeypatch):
    import providers.yfinance_provider as yp
    called = {"n": 0}
    monkeypatch.setattr(yp.yf, "download", lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    assert YFinanceProvider().get_price_history_batch([], "1y", "1wk") == {}
    assert called["n"] == 0   # short-circuits before hitting the network


# ── FallbackProvider (inherited default → Schwab primary, yfinance net) ───────

def test_fallback_batch_uses_primary_per_symbol():
    primary = FakeProvider(name="schwab", price=make_price_df([100.0, 110.0]))
    secondary = FakeProvider(name="yfinance", price=make_price_df([1.0, 2.0]))
    fb = FallbackProvider(primary, secondary)
    out = fb.get_price_history_batch(["AAA", "BBB"], "1y", "1wk")
    assert set(out) == {"AAA", "BBB"}
    assert float(out["AAA"]["Close"].iloc[-1]) == 110.0   # from schwab
    assert fb.last_source["price"] == "schwab"
    assert secondary.calls == []                          # net never touched


def test_fallback_batch_degrades_per_symbol_when_primary_down():
    primary = FakeProvider(name="schwab", raise_on={"price"})
    secondary = FakeProvider(name="yfinance", price=make_price_df([3.0, 4.0]))
    fb = FallbackProvider(primary, secondary)
    out = fb.get_price_history_batch(["AAA", "BBB"], "1y", "1wk")
    assert set(out) == {"AAA", "BBB"}
    assert float(out["BBB"]["Close"].iloc[-1]) == 4.0     # from yfinance
    assert fb.last_source["price"] == "yfinance"
