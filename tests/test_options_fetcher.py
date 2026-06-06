"""Tests for options_fetcher — the provider-agnostic layer that does futures
proxy resolution, quality gating, staleness tagging, and (critically) the
"never overwrite good cache with bad data" rule that fixed the post-midnight
Total GEX = 0 bug (Requirement 3)."""

import glob
import os

import pandas as pd

import options_fetcher
from options_fetcher import (
    _resolve_ticker, _cache_path, _as_of_from_meta_or_filename, _chain_source,
    _drop_expired, fetch_options_chain,
)
from tests.conftest import FakeProvider, make_chain


def _use_provider(monkeypatch, provider):
    monkeypatch.setattr(options_fetcher, "get_provider", lambda *a, **k: provider)


# ── small pure helpers ──────────────────────────────────────────────────────
def test_resolve_ticker_maps_futures_to_proxy():
    assert _resolve_ticker("ES=F") == ("SPY", True)
    assert _resolve_ticker("SPY") == ("SPY", False)
    assert _resolve_ticker("spy") == ("SPY", False)


def test_cache_path_contains_safe_name_and_today():
    from datetime import date
    p = _cache_path("ES=F", "2026-06-19")
    base = os.path.basename(p)
    assert base.startswith("opts_ES_F_2026-06-19_")
    assert base.endswith(f"{date.today().isoformat()}.json")


def test_as_of_prefers_meta_then_filename():
    # meta wins
    assert _as_of_from_meta_or_filename("opts_SPY_x_2026-06-19.json",
                                        {"as_of": "2026-05-01"}) == "2026-05-01"
    # else parse the trailing date from the filename
    assert _as_of_from_meta_or_filename("opts_SPY_x_2026-06-19.json", {}) == "2026-06-19"
    # neither -> None
    assert _as_of_from_meta_or_filename("no-date-here.json", {}) is None


def test_chain_source_prefers_last_source_then_name():
    p = FakeProvider(name="schwab", last_source={"chain": "yfinance"})
    assert _chain_source(p) == "yfinance"          # fallback recorded the real source
    p2 = FakeProvider(name="schwab")               # last_source empty -> use name
    assert _chain_source(p2) == "schwab"


# ── fetch routing / caching ─────────────────────────────────────────────────
def test_usable_live_writes_cache_and_tags_source(tmp_cache, monkeypatch):
    prov = FakeProvider(name="schwab", chain=(make_chain(12), make_chain(12)))
    _use_provider(monkeypatch, prov)
    calls, puts, meta = fetch_options_chain("SPY", "2026-06-19", use_cache=False)
    assert len(calls) == 12 and len(puts) == 12
    assert meta["source"] == "schwab"
    assert meta["stale"] is False
    assert meta["spot_price"] == 102.5            # last Close of default make_price_df
    # a cache file for today was persisted
    files = glob.glob(os.path.join(tmp_cache, "opts_SPY_2026-06-19_*.json"))
    assert len(files) == 1


def test_unusable_live_without_cache_returns_live_but_persists_nothing(tmp_cache, monkeypatch):
    prov = FakeProvider(name="schwab", chain=(make_chain(1), make_chain(1)))  # 2 < 20
    _use_provider(monkeypatch, prov)
    calls, puts, meta = fetch_options_chain("SPY", "2026-06-19", use_cache=False)
    # best-effort live data is returned ...
    assert len(calls) == 1
    # ... but nothing was cached, so a bad snapshot can't poison later reads
    assert glob.glob(os.path.join(tmp_cache, "opts_SPY_*.json")) == []


def test_unusable_live_falls_back_to_last_good_cache(tmp_cache, monkeypatch):
    # 1) prime a GOOD cache.
    good = FakeProvider(name="schwab", chain=(make_chain(12), make_chain(12)))
    _use_provider(monkeypatch, good)
    _, _, meta_good = fetch_options_chain("SPY", "2026-06-19", use_cache=False)
    assert meta_good["stale"] is False

    # 2) now the live source goes bad; bypass today's fresh cache with use_cache=False
    #    so we exercise the live->unusable->good-cache fallback path.
    bad = FakeProvider(name="schwab", chain=(make_chain(1), make_chain(1)))
    _use_provider(monkeypatch, bad)
    calls, puts, meta = fetch_options_chain("SPY", "2026-06-19", use_cache=False)
    assert meta["stale"] is True
    assert meta["as_of"] is not None
    assert meta["cache_file"]
    assert len(calls) == 12          # served from the good cache, not the 1-row bad live


def test_fresh_today_cache_is_served_when_use_cache_true(tmp_cache, monkeypatch):
    prov = FakeProvider(name="schwab", chain=(make_chain(12), make_chain(12)))
    _use_provider(monkeypatch, prov)
    fetch_options_chain("SPY", "2026-06-19", use_cache=False)   # write cache
    # second call with use_cache=True should read the fresh file (stale False)
    calls, puts, meta = fetch_options_chain("SPY", "2026-06-19", use_cache=True)
    assert meta["stale"] is False
    assert len(calls) == 12


def test_provider_exception_falls_back_to_cache(tmp_cache, monkeypatch):
    good = FakeProvider(name="schwab", chain=(make_chain(12), make_chain(12)))
    _use_provider(monkeypatch, good)
    fetch_options_chain("SPY", "2026-06-19", use_cache=False)   # prime good cache

    boom = FakeProvider(name="schwab", raise_on={"chain"})
    _use_provider(monkeypatch, boom)
    calls, puts, meta = fetch_options_chain("SPY", "2026-06-19", use_cache=False)
    assert meta["stale"] is True
    assert len(calls) == 12


def test_no_expirations_no_cache_returns_empty(tmp_cache, monkeypatch):
    prov = FakeProvider(name="schwab", expirations=[])
    _use_provider(monkeypatch, prov)
    calls, puts, meta = fetch_options_chain("SPY", use_cache=False)
    assert calls.empty and puts.empty and meta == {}


# ── expired-expiry filter (the Schwab silent-fallback bug) ───────────────────
def test_drop_expired_filters_past_keeps_today_and_future():
    """Schwab lists the just-expired date first; only >= today should survive,
    and today's still-live 0DTE must be kept for the pre-open recorder."""
    from datetime import date, timedelta
    today = date.today()
    past = (today - timedelta(days=3)).isoformat()
    tdy = today.isoformat()
    fut = (today + timedelta(days=5)).isoformat()
    assert _drop_expired([past, tdy, fut]) == [tdy, fut]   # today's 0DTE survives
    assert _drop_expired(["weird", fut]) == ["weird", fut]  # unparseable kept
    assert _drop_expired([past]) == [past]                  # all-past: defensive no-op
    assert _drop_expired([]) == []


def test_fetch_skips_expired_first_expiry(tmp_cache, monkeypatch):
    """Reproduces the silent-fallback bug: Schwab returns the expired date FIRST.
    fetch_options_chain must select the nearest LIVE expiry, not available[0]."""
    from datetime import date, timedelta
    past = (date.today() - timedelta(days=2)).isoformat()    # already expired
    future = (date.today() + timedelta(days=5)).isoformat()  # live
    prov = FakeProvider(name="schwab", expirations=[past, future],
                        chain=(make_chain(12), make_chain(12)))
    _use_provider(monkeypatch, prov)
    calls, puts, meta = fetch_options_chain("SPY", use_cache=False)  # no explicit expiry
    assert meta["expiry"] == future       # picked the live expiry, not the expired one
    assert meta["source"] == "schwab"     # so Schwab serves the chain (no fallback)
    assert len(calls) == 12
