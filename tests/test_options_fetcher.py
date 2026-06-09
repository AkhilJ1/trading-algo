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
    and today's still-live 0DTE must be kept for the pre-open recorder —
    but ONLY until the close: a post-close run must roll to the next session
    (a same-day "forecast" after the close is the estimated==actual bug)."""
    from datetime import datetime, date, timedelta
    # Pin the clock to a weekday so the test passes at any wall time:
    # Tue 2026-06-09, once mid-session and once just after the 16:00 ET close.
    intraday = datetime(2026, 6, 9, 11, 0)     # 11:00 ET — session live
    post_close = datetime(2026, 6, 9, 16, 16)  # 16:16 ET — session closed
    today = date(2026, 6, 9)
    past = (today - timedelta(days=3)).isoformat()
    tdy = today.isoformat()
    fut = (today + timedelta(days=5)).isoformat()
    # Intraday: today's 0DTE survives.
    assert _drop_expired([past, tdy, fut], now_et=intraday) == [tdy, fut]
    assert _drop_expired(["weird", fut], now_et=intraday) == ["weird", fut]  # unparseable kept
    assert _drop_expired([past], now_et=intraday) == [past]   # all-past: defensive no-op
    assert _drop_expired([], now_et=intraday) == []
    # Post-close: today's 0DTE is over — only the future expiry survives.
    assert _drop_expired([past, tdy, fut], now_et=post_close) == [fut]
    # Saturday: the week's last expiry (Friday) is gone too.
    saturday = datetime(2026, 6, 13, 10, 0)
    fri = date(2026, 6, 12).isoformat()
    nxt = date(2026, 6, 15).isoformat()
    assert _drop_expired([fri, nxt], now_et=saturday) == [nxt]


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


# ── expiry-faithful stale fallback (the "dealer pin shows the wrong date" bug) ─
def _prime_two_expiries(monkeypatch, tmp_cache, near, far):
    """Cache GOOD snapshots for two future expiries, then force the NEAR one to be
    the newest by mtime — i.e. the exact weekend setup where the most-recently
    cached chain is the nearest expiry, not the one the user is asking about."""
    import glob as _glob
    good = FakeProvider(name="schwab", expirations=[near, far],
                        chain=(make_chain(12), make_chain(12)))
    _use_provider(monkeypatch, good)
    fetch_options_chain("SPY", far, use_cache=False)    # far cache first
    fetch_options_chain("SPY", near, use_cache=False)   # near cache second
    # Make NEAR strictly newest regardless of filesystem mtime granularity.
    import os as _os
    far_f = _glob.glob(_os.path.join(tmp_cache, f"opts_SPY_{far}_*.json"))[0]
    near_f = _glob.glob(_os.path.join(tmp_cache, f"opts_SPY_{near}_*.json"))[0]
    _os.utime(far_f, (1_000, 1_000))
    _os.utime(near_f, (2_000, 2_000))


def test_unusable_live_serves_requested_expiry_not_newest_cache(tmp_cache, monkeypatch):
    """The core bug: requesting the monthly (far) expiry after hours must serve
    the cached FAR chain — never the newer cached NEAR expiry behind the same
    label, which made the dealer pin / GEX / walls read for the wrong date."""
    from datetime import date, timedelta
    near = (date.today() + timedelta(days=3)).isoformat()
    far = (date.today() + timedelta(days=11)).isoformat()
    _prime_two_expiries(monkeypatch, tmp_cache, near, far)

    # Live now returns an unusable (weekend) chain for every expiry.
    bad = FakeProvider(name="schwab", expirations=[near, far],
                       chain=(make_chain(1), make_chain(1)))
    _use_provider(monkeypatch, bad)
    calls, puts, meta = fetch_options_chain("SPY", far, use_cache=False)
    assert meta["expiry"] == far                       # served the expiry asked for
    assert meta["stale"] is True                       # from last-known-good cache
    assert not meta.get("expiry_substituted")          # no silent swap
    assert meta.get("requested_expiry") is None


def test_unusable_live_substitutes_and_flags_when_requested_expiry_uncached(tmp_cache, monkeypatch):
    """If no snapshot exists for the requested expiry, the newest-overall chain is
    served BUT flagged, so the UI can say which expiry it actually showed instead
    of silently presenting the nearest expiry as the requested one."""
    from datetime import date, timedelta
    near = (date.today() + timedelta(days=3)).isoformat()
    far = (date.today() + timedelta(days=11)).isoformat()
    uncached = (date.today() + timedelta(days=20)).isoformat()
    _prime_two_expiries(monkeypatch, tmp_cache, near, far)

    bad = FakeProvider(name="schwab", expirations=[near, far, uncached],
                       chain=(make_chain(1), make_chain(1)))
    _use_provider(monkeypatch, bad)
    calls, puts, meta = fetch_options_chain("SPY", uncached, use_cache=False)
    assert meta["expiry"] == near                      # newest available snapshot
    assert meta["expiry_substituted"] is True          # and it is flagged
    assert meta["requested_expiry"] == uncached
    assert meta["stale"] is True
