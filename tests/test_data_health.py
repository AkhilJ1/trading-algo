"""Tests for data_health — the daily cloud accuracy check (Requirement 3).

Two responsibilities: (1) the pipeline check flags empty/unusable/stale/no-spot
chains, and (2) cross-source reconciliation flags Schwab-vs-yfinance spot drift
beyond tolerance while treating "Schwab not configured" as a clean skip.
"""

import pandas as pd

import data_health
from tests.conftest import make_chain, make_price_df


# ── pipeline check ──────────────────────────────────────────────────────────
def test_pipeline_ok_when_usable_fresh_and_priced(monkeypatch):
    def fake_fetch(ticker, use_cache=False):
        meta = {"spot_price": 100.0, "stale": False, "source": "schwab"}
        return make_chain(12), make_chain(12), meta
    monkeypatch.setattr(data_health, "fetch_options_chain", fake_fetch)
    _meta, problems = data_health.check_pipeline("SPY")
    assert problems == []


def test_pipeline_flags_empty_chain(monkeypatch):
    def fake_fetch(ticker, use_cache=False):
        return pd.DataFrame(), pd.DataFrame(), {}
    monkeypatch.setattr(data_health, "fetch_options_chain", fake_fetch)
    _meta, problems = data_health.check_pipeline("SPY")
    assert any("empty chain" in p for p in problems)


def test_pipeline_flags_unusable_nonpositive_spot_and_stale(monkeypatch):
    def fake_fetch(ticker, use_cache=False):
        meta = {"spot_price": 0.0, "stale": True,
                "as_of": "2026-06-01", "cache_file": "opts_SPY_x.json"}
        return make_chain(1), make_chain(1), meta      # 2 usable strikes < 20
    monkeypatch.setattr(data_health, "fetch_options_chain", fake_fetch)
    _meta, problems = data_health.check_pipeline("SPY")
    joined = " ".join(problems)
    assert "usability gate" in joined
    assert "non-positive spot" in joined
    assert "STALE" in joined


# ── cross-source reconciliation ─────────────────────────────────────────────
def _patch_sources(monkeypatch, schwab_cls, yfin_cls):
    import providers.schwab_provider as sp
    import providers.yfinance_provider as yp
    monkeypatch.setattr(sp, "SchwabProvider", schwab_cls)
    monkeypatch.setattr(yp, "YFinanceProvider", yfin_cls)


def test_reconcile_clean_when_spots_agree(monkeypatch):
    class S:
        def get_price_history(self, t, period, interval):
            return make_price_df([100.0, 100.0])

    class Y:
        def get_price_history(self, t, period, interval):
            return make_price_df([100.5, 100.5])     # 0.5% drift < 1%

    _patch_sources(monkeypatch, S, Y)
    assert data_health.reconcile_sources("SPY") == []


def test_reconcile_flags_drift_beyond_tolerance(monkeypatch):
    class S:
        def get_price_history(self, t, period, interval):
            return make_price_df([100.0, 100.0])

    class Y:
        def get_price_history(self, t, period, interval):
            return make_price_df([105.0, 105.0])     # ~4.8% drift > 1%

    _patch_sources(monkeypatch, S, Y)
    out = data_health.reconcile_sources("SPY")
    assert len(out) == 1
    assert "drift" in out[0]


def test_reconcile_skips_when_schwab_unavailable(monkeypatch):
    class S:
        def get_price_history(self, *a, **k):
            raise RuntimeError("no creds/token")  # unavailable, not a data problem

    class Y:
        def get_price_history(self, t, period, interval):
            return make_price_df([100.0, 100.0])

    _patch_sources(monkeypatch, S, Y)
    assert data_health.reconcile_sources("SPY") == []
