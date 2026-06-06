"""Tests for the dynamic candidate screener.

The screener used to call yfinance directly (`import yfinance as yf` +
`yf.download`), bypassing the provider abstraction. It now routes through
get_provider().get_price_history_batch, so with DATA_PROVIDER=schwab the S&P
universe is pulled from Schwab (yfinance per-symbol fallback). These tests pin
that routing and the per-symbol normalization offline — no network.
"""

import pandas as pd

import screener
from tests.conftest import FakeProvider, make_price_df


# ── no direct yfinance bypass remains ────────────────────────────────────────

def test_screener_has_no_direct_yfinance_handle():
    # Regression guard: the module must not re-introduce a top-level yf import.
    assert not hasattr(screener, "yf")


# ── routing: phase-1 goes through the provider batch ─────────────────────────

def test_discover_candidates_routes_through_provider(monkeypatch):
    fake = FakeProvider(name="schwab")            # default price = 3 rows
    monkeypatch.setattr(screener, "get_provider", lambda *a, **k: fake)
    monkeypatch.setattr(screener, "get_sp500_tickers", lambda: ["AAA", "BBB"])

    out = screener.discover_candidates()

    # 3-row frames fail the len>=20 pre-filter → no survivors, no phase-2 — but
    # the universe WAS fetched through the provider's batch (per-symbol loop).
    assert out == []
    fetched = [c[1] for c in fake.calls if c[0] == "get_price_history"]
    assert fetched == ["AAA", "BBB"]


def test_discover_candidates_scores_a_survivor(monkeypatch):
    # A long-enough, oversold weekly frame should survive phase-1 and get scored;
    # phase-2's backtest is stubbed so the test stays offline.
    closes = [100.0 - i for i in range(30)]        # steady decline → low RSI
    long_df = make_price_df(closes)
    fake = FakeProvider(name="schwab", price=long_df)
    monkeypatch.setattr(screener, "get_provider", lambda *a, **k: fake)
    monkeypatch.setattr(screener, "get_sp500_tickers", lambda: ["AAA"])
    monkeypatch.setattr(
        screener, "backtest_ma_crossover",
        lambda *a, **k: {"win_rate_pct": 55.0, "num_trades": 4, "total_return_pct": 12.0},
    )
    monkeypatch.setattr(
        screener, "get_buy_signal",
        lambda df: {
            "strength": 80.0, "close": 71.0, "rsi": 25.0, "buy_signal": True,
            "rsi_oversold": True, "wick_touches": 2,
            "bb_lower": 70.0, "bb_mid": 75.0, "bb_upper": 80.0,
        },
    )

    out = screener.discover_candidates(rsi_prefilter=100.0)   # let it through
    assert len(out) == 1
    assert out[0]["ticker"] == "AAA"
    # composite = strength*0.6 + win_rate*0.4 = 80*0.6 + 55*0.4 = 70.0
    assert out[0]["score"] == 70.0


# ── _normalise_df (single per-symbol frame) ──────────────────────────────────

def test_normalise_df_titlecases_and_strips_tz():
    idx = pd.date_range("2026-01-01", periods=3, freq="W", tz="US/Eastern")
    df = pd.DataFrame(
        {"open": [1, 2, 3], "high": [2, 3, 4], "low": [0, 1, 2],
         "close": [1, 2, 3], "volume": [10, 11, 12]},
        index=idx,
    )
    out = screener._normalise_df(df)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.tz is None


def test_normalise_df_handles_garbage():
    assert screener._normalise_df(None).empty


# ── universe resolution falls back to the local watchlist ────────────────────

def test_get_sp500_tickers_falls_back_to_watchlist(monkeypatch):
    from config import WATCHLIST
    screener._universe_cache = []     # bypass any cached universe
    def boom(*a, **k):
        raise RuntimeError("wikipedia down")
    monkeypatch.setattr(screener.pd, "read_html", boom)
    assert screener.get_sp500_tickers() == list(WATCHLIST)
