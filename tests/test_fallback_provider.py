"""Tests for FallbackProvider — Schwab primary, yfinance safety net.

This routing is what keeps the pipeline autonomous across the weekly Schwab
re-auth gap (Requirement 2), and what records the true source for monitoring
(Requirement 3).
"""

import pandas as pd

from providers.fallback_provider import FallbackProvider
from tests.conftest import FakeProvider, make_chain, make_price_df


def _fb(primary, secondary):
    return FallbackProvider(primary, secondary)


# ── name / construction ────────────────────────────────────────────────────
def test_name_combines_both_backends():
    fb = _fb(FakeProvider(name="schwab"), FakeProvider(name="yfinance"))
    assert fb.name == "schwab->yfinance"


# ── price routing ──────────────────────────────────────────────────────────
def test_price_uses_primary_when_usable():
    primary = FakeProvider(name="schwab", price=make_price_df([100.0, 110.0]))
    secondary = FakeProvider(name="yfinance")
    fb = _fb(primary, secondary)
    out = fb.get_price_history("SPY", "5d", "1d")
    assert float(out["Close"].iloc[-1]) == 110.0
    assert fb.last_source["price"] == "schwab"
    # secondary must NOT have been consulted
    assert secondary.calls == []


def test_price_falls_back_when_primary_raises():
    primary = FakeProvider(name="schwab", raise_on={"price"})
    secondary = FakeProvider(name="yfinance", price=make_price_df([50.0, 51.0]))
    fb = _fb(primary, secondary)
    out = fb.get_price_history("SPY", "5d", "1d")
    assert float(out["Close"].iloc[-1]) == 51.0
    assert fb.last_source["price"] == "yfinance"


def test_price_falls_back_when_primary_unusable():
    # Primary returns an empty frame -> fails price_is_usable -> secondary.
    primary = FakeProvider(name="schwab", price=pd.DataFrame())
    secondary = FakeProvider(name="yfinance", price=make_price_df([7.0, 8.0]))
    fb = _fb(primary, secondary)
    out = fb.get_price_history("SPY", "5d", "1d")
    assert float(out["Close"].iloc[-1]) == 8.0
    assert fb.last_source["price"] == "yfinance"


# ── expirations routing ────────────────────────────────────────────────────
def test_expirations_use_primary_when_nonempty():
    primary = FakeProvider(name="schwab", expirations=["2026-06-19"])
    secondary = FakeProvider(name="yfinance", expirations=["2026-07-17"])
    fb = _fb(primary, secondary)
    assert fb.get_expirations("SPY") == ["2026-06-19"]
    assert fb.last_source["expirations"] == "schwab"


def test_expirations_fall_back_when_primary_empty():
    # Empty list is falsy -> fails the validator -> secondary.
    primary = FakeProvider(name="schwab", expirations=[])
    secondary = FakeProvider(name="yfinance", expirations=["2026-07-17"])
    fb = _fb(primary, secondary)
    assert fb.get_expirations("SPY") == ["2026-07-17"]
    assert fb.last_source["expirations"] == "yfinance"


# ── chain routing ──────────────────────────────────────────────────────────
def test_chain_uses_primary_when_usable():
    good = (make_chain(12), make_chain(12))
    primary = FakeProvider(name="schwab", chain=good)
    secondary = FakeProvider(name="yfinance")
    fb = _fb(primary, secondary)
    calls, puts = fb.get_option_chain("SPY", "2026-06-19")
    assert len(calls) == 12 and len(puts) == 12
    assert fb.last_source["chain"] == "schwab"
    assert secondary.calls == []


def test_chain_falls_back_when_primary_unusable():
    bad = (make_chain(2), make_chain(2))        # only 4 usable -> < 20
    good = (make_chain(15), make_chain(15))
    primary = FakeProvider(name="schwab", chain=bad)
    secondary = FakeProvider(name="yfinance", chain=good)
    fb = _fb(primary, secondary)
    calls, puts = fb.get_option_chain("SPY", "2026-06-19")
    assert len(calls) == 15
    assert fb.last_source["chain"] == "yfinance"


def test_chain_falls_back_when_primary_raises():
    good = (make_chain(15), make_chain(15))
    primary = FakeProvider(name="schwab", raise_on={"chain"})
    secondary = FakeProvider(name="yfinance", chain=good)
    fb = _fb(primary, secondary)
    calls, _ = fb.get_option_chain("SPY", "2026-06-19")
    assert len(calls) == 15
    assert fb.last_source["chain"] == "yfinance"


# ── both sources down ──────────────────────────────────────────────────────
def test_raises_and_clears_source_when_both_fail():
    primary = FakeProvider(name="schwab", raise_on={"price"})
    secondary = FakeProvider(name="yfinance", raise_on={"price"})
    fb = _fb(primary, secondary)
    try:
        fb.get_price_history("SPY", "5d", "1d")
        assert False, "expected the secondary's exception to propagate"
    except RuntimeError:
        pass
    assert fb.last_source["price"] is None
