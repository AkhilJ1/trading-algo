"""
Tests for the pre-open dealer-pin pipeline.

Two new pieces, both exercised offline (no network, no creds):

  * the live-spot anchor — fetch_live_spot() + the provider get_quote() routing
    that lets a pre-open run use the pre-market price instead of yesterday's
    settled daily close, and
  * build_pin_forecast_kwargs() — the pure flattening of a composite-analysis
    result into the durable PinForecasts row the evening grader later scores.

Style matches test_track_record.py / test_record_calibration.py: hand-built
dicts in, dicts out, with controllable FakeProviders for the routing.
"""

import options_fetcher
from options_fetcher import fetch_live_spot
from providers.fallback_provider import FallbackProvider
from record_preopen_pin import build_pin_forecast_kwargs
from tests.conftest import FakeProvider


# ── provider get_quote routing (Schwab primary → yfinance → None) ────────────

def test_fallback_quote_uses_primary_when_positive():
    fb = FallbackProvider(FakeProvider(name="schwab", quote=601.4),
                          FakeProvider(name="yfinance", quote=600.0))
    assert fb.get_quote("SPY") == 601.4
    assert fb.last_source["quote"] == "schwab"


def test_fallback_quote_degrades_to_secondary_when_primary_missing():
    # Primary returns None (no live quote) → secondary supplies it.
    fb = FallbackProvider(FakeProvider(name="schwab", quote=None),
                          FakeProvider(name="yfinance", quote=600.0))
    assert fb.get_quote("SPY") == 600.0
    assert fb.last_source["quote"] == "yfinance"


def test_fallback_quote_degrades_when_primary_raises():
    fb = FallbackProvider(FakeProvider(name="schwab", raise_on={"quote"}),
                          FakeProvider(name="yfinance", quote=599.5))
    assert fb.get_quote("SPY") == 599.5
    assert fb.last_source["quote"] == "yfinance"


def test_fallback_quote_none_when_both_unavailable():
    # A missing live quote is a soft miss — None, never an exception.
    fb = FallbackProvider(FakeProvider(name="schwab", quote=None),
                          FakeProvider(name="yfinance", quote=None))
    assert fb.get_quote("SPY") is None
    assert fb.last_source["quote"] is None


def test_fallback_quote_rejects_nonpositive_primary():
    # A 0/negative print is not a usable spot → fall through to secondary.
    fb = FallbackProvider(FakeProvider(name="schwab", quote=0.0),
                          FakeProvider(name="yfinance", quote=600.0))
    assert fb.get_quote("SPY") == 600.0
    assert fb.last_source["quote"] == "yfinance"


# ── fetch_live_spot (proxy resolution + sanity gating) ───────────────────────

def test_fetch_live_spot_returns_quote():
    prov = FakeProvider(name="schwab", quote=600.25)
    assert fetch_live_spot("SPY", provider=prov) == 600.25


def test_fetch_live_spot_resolves_futures_proxy():
    # ES=F resolves to its SPY options proxy before the quote is requested.
    prov = FakeProvider(name="schwab", quote=600.0)
    assert fetch_live_spot("ES=F", provider=prov) == 600.0
    assert ("get_quote", "SPY") in prov.calls


def test_fetch_live_spot_none_when_no_quote():
    prov = FakeProvider(name="schwab", quote=None)
    assert fetch_live_spot("SPY", provider=prov) is None


def test_fetch_live_spot_rejects_nonpositive():
    prov = FakeProvider(name="schwab", quote=-1.0)
    assert fetch_live_spot("SPY", provider=prov) is None


def test_fetch_live_spot_swallows_provider_errors():
    prov = FakeProvider(name="schwab", raise_on={"quote"})
    assert fetch_live_spot("SPY", provider=prov) is None


def test_fetch_live_spot_uses_factory_when_no_provider(monkeypatch):
    prov = FakeProvider(name="schwab", quote=607.0)
    monkeypatch.setattr(options_fetcher, "get_provider", lambda *a, **k: prov)
    assert fetch_live_spot("SPY") == 607.0


# ── build_pin_forecast_kwargs (pure result → durable row) ────────────────────

def _result(**over):
    base = {
        "ticker": "SPY",
        "timestamp": "2026-06-04",
        "spot_price": 601.30,
        "floor": 598.0,
        "ceiling": 604.0,
        "bias": "bullish",
        "confidence": 62.0,
        "expiry": "2026-06-04",
        "max_pain": 600.0,
        "market_regime": "transitional",
        "spot_source": "live_override",
        "source": "schwab",
        "estimated_close": {
            "estimated_close": 600.85,
            "pin_target": 600.40,
            "max_pain": 600.0,
            "gamma_regime": "positive",
        },
    }
    base.update(over)
    return base


def test_kwargs_carry_pin_scalars_and_anchor_spot():
    kw = build_pin_forecast_kwargs(_result(), ticker="SPY", vix_val=14.2, gex_net=1.0e9)
    assert kw["date_str"] == "2026-06-04"
    assert kw["ticker"] == "SPY"
    assert kw["spot_price"] == 601.30          # the live pre-market anchor, not yesterday's close
    assert kw["estimated_close"] == 600.85     # the pin we will grade
    assert kw["pin_target"] == 600.40
    assert kw["max_pain"] == 600.0
    assert kw["expiry"] == "2026-06-04"         # today's 0DTE
    assert kw["vix"] == 14.2
    assert kw["gex_net"] == 1.0e9
    # Provenance carries through for data-accuracy auditing.
    assert kw["spot_source"] == "live_override"  # anchored on the live pre-market quote
    assert kw["chain_source"] == "schwab"        # chain served by Schwab


def test_kwargs_prefer_gamma_regime_over_vix_regime():
    # The pin engine's gamma regime is the more relevant label for a pin row.
    kw = build_pin_forecast_kwargs(_result(), ticker="SPY", regime="elevated")
    assert kw["regime"] == "positive"


def test_kwargs_fall_back_to_vix_regime_without_gamma():
    r = _result()
    r["estimated_close"].pop("gamma_regime")
    kw = build_pin_forecast_kwargs(r, ticker="SPY", regime="elevated")
    assert kw["regime"] == "elevated"


def test_kwargs_handle_non_dict_pin_gracefully():
    # If the dealer-pin sub-result is missing/None, the pin scalars blank out
    # but max_pain still falls back to the top-level value (append-only safe).
    kw = build_pin_forecast_kwargs(_result(estimated_close=None), ticker="SPY")
    assert kw["estimated_close"] is None
    assert kw["pin_target"] is None
    assert kw["max_pain"] == 600.0


def test_kwargs_default_provenance_when_absent():
    # Older result dicts without provenance default safely: spot is treated as a
    # settled daily close and the chain backend is left blank rather than guessed.
    r = _result()
    r.pop("spot_source")
    r.pop("source")
    kw = build_pin_forecast_kwargs(r, ticker="SPY")
    assert kw["spot_source"] == "daily_close"
    assert kw["chain_source"] == ""
