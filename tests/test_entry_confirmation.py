"""
Tests for the entry-confirmation checklist.

The properties that matter here are the ones that would let the page lie:
a trigger that peeks at future bars, a "wait for confirmation" measurement
that silently becomes an intersection, an arrival rate that hides the setups
where the signal never came, and a thin sample allowed to print a number.
"""

import numpy as np
import pandas as pd
import pytest

from entry_confirmation import (
    MAX_WAIT,
    PIVOT,
    RECENT_BARS,
    _pivot_flags,
    _structure_break,
    attach_confirmations,
    build_triggers,
    confirmation_stats,
    confirmation_table,
    status_text,
)
from trade_ideas import MIN_ANALOGS, evaluate_ticker


def _ohlcv(n=1500, seed=3, drift=0.0002, vol=0.018):
    """Synthetic but realistically shaped daily bars: OHLC that bracket close."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    reach = np.abs(rng.normal(0, vol * 0.6, n))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, vol * 0.3, n)),
            "High": close * (1 + reach),
            "Low": close * (1 - reach),
            "Close": close,
            "Volume": rng.lognormal(16, 0.4, n),
        },
        index=idx,
    )


def _bench(index, seed=11):
    return pd.Series(
        100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0.0003, 0.01, len(index)))),
        index=index,
    )


# ── Lookahead safety ────────────────────────────────────────────────────
# The single most dangerous failure mode: a confirmation that a chart could
# not have shown yet still scores as though you had acted on it.


#   value
#      20 ╲                                                      ╱
#          ╲                        ╱╲                         ╱
#           ╲                    ╱      ╲                   ╱
#            ╲               ╱             ╲            ╱
#             ╲          ╱                    ╲      ╱
#              ╲     ╱                           ╲╱
#          10   ╲╱                            12
#               t=10                          t=24
# Two swing lows, the second higher than the first. The higher low is a real
# fact at t=24 but is only KNOWABLE at t=27, once three more bars have printed
# without undercutting it.
_ZIGZAG = (
    list(range(20, 9, -1))          # t 0..10, trough of 10 at t=10
    + list(range(11, 19))           # t 11..18, up to 18
    + list(range(17, 11, -1))       # t 19..24, trough of 12 at t=24
    + list(range(13, 21))           # t 25..32, up again
)


def test_swing_pivot_is_not_visible_before_it_is_confirmable():
    """
    A centred window sees `PIVOT` bars into the future, so the raw pivot flag
    is knowledge a chart could not have had yet. The confirmation must be
    delayed by exactly that much — otherwise the scan buys the low three
    sessions before anyone could have identified it as a low, and every number
    downstream inherits the free lookahead.
    """
    lows = pd.Series(_ZIGZAG, dtype=float,
                     index=pd.bdate_range("2020-01-01", periods=len(_ZIGZAG)))

    raw = _pivot_flags(lows, PIVOT, low_side=True)
    assert np.flatnonzero(raw.to_numpy()).tolist() == [10, 24], "sanity: two swing lows"

    mask, last, prev = _structure_break(lows, PIVOT, low_side=True)
    assert (last, prev) == (12.0, 10.0)

    fired = np.flatnonzero(mask.to_numpy()).tolist()
    assert fired == [24 + PIVOT], (
        f"higher low must confirm at t={24 + PIVOT}, not at {fired}"
    )
    assert not bool(mask.iloc[24]), "fired on the pivot bar itself — pure lookahead"


@pytest.mark.parametrize("direction", ["oversold", "overbought"])
def test_no_trigger_depends_on_bars_after_it_fires(direction):
    """
    Truncating history must not change a single flag on the bars that remain.

    Compared over the WHOLE truncated frame, right up to its final bar, on
    purpose. Trimming a few bars off the end would hide the exact bug this
    guards: a centred window reading `PIVOT` bars ahead differs from the
    truncated frame only in that tail.
    """
    df = _ohlcv()
    cut = len(df) - 40
    full = {t.key: t.mask for t in build_triggers(df, direction)}
    early = {t.key: t.mask for t in build_triggers(df.iloc[:cut], direction)}

    assert set(full) == set(early)
    for key, mask in early.items():
        pd.testing.assert_series_equal(
            mask, full[key].iloc[:cut], check_names=False,
            obj=f"{key} changed when future bars were removed",
        )


# ── Trigger construction ────────────────────────────────────────────────


def _slot(key: str) -> str:
    """Trigger key with its direction-specific naming stripped."""
    for prefix in ("bullish_", "bearish_"):
        key = key.replace(prefix, "")
    return key.replace("_up_with_price", "_with_price").replace("_down_with_price",
                                                                "_with_price")


def test_triggers_mirror_by_direction():
    df = _ohlcv()
    bull = build_triggers(df, "oversold")
    bear = build_triggers(df, "overbought")
    assert bull and bear
    # Same slots in the same order — an oversold name must not be handed the
    # "run is rolling over" checklist, but it must not be handed a shorter one
    # either.
    assert [_slot(t.key) for t in bull] == [_slot(t.key) for t in bear]
    assert bull[0].key == "rsi_up_with_price" and bear[0].key == "rsi_down_with_price"
    assert {bull[2].key, bear[2].key} == {"bullish_divergence", "bearish_divergence"}
    for b, s in zip(bull, bear):
        assert not b.mask.equals(s.mask), f"{b.key} identical in both directions"


def test_rsi_turning_up_requires_price_to_confirm():
    """The user-facing promise is 'RSI increasing AS price increases'. A rising
    RSI on falling closes must not satisfy it."""
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)
    # Monotonically falling price: RSI can tick up on smaller down-days, but
    # price never closes higher, so the trigger must never fire.
    close = pd.Series(np.linspace(200, 100, n), index=idx)
    df = pd.DataFrame({"Close": close, "High": close, "Low": close,
                       "Volume": np.full(n, 1e6)})
    trig = next(t for t in build_triggers(df, "oversold") if t.key == "rsi_up_with_price")
    assert not trig.mask.any(), "fired without price confirming"


def test_triggers_survive_frames_without_volume():
    """Close-only frames must lose the volume row, not raise."""
    df = _ohlcv()[["Close"]]
    keys = [t.key for t in build_triggers(df, "oversold")]
    assert keys, "close-only frame produced no triggers at all"
    assert "volume" not in keys


def test_all_zero_volume_is_treated_as_missing_not_as_quiet_days():
    """
    Schwab returns 0 volume for index symbols ($VIX, $SPX), as does yfinance.
    Kept, the row would say "showed up 0% of the time" — which reads as a fact
    about the ticker rather than an absent feed.
    """
    df = _ohlcv()
    df["Volume"] = 0
    assert "volume" not in [t.key for t in build_triggers(df, "oversold")]


def test_volume_arriving_as_strings_still_works():
    """Provider frames are JSON-derived; a numeric-looking string column must
    not silently disable the volume trigger."""
    df = _ohlcv()
    df["Volume"] = df["Volume"].astype(int).astype(str)
    trig = [t for t in build_triggers(df, "oversold") if t.key == "volume"]
    assert trig and trig[0].mask.any()


def test_schwab_shaped_frame_yields_the_full_checklist():
    """
    The app runs Schwab-primary. Its provider returns Open/High/Low/Close/Volume
    on a naive-Eastern DatetimeIndex named 'Date', with integer volume — a
    different dtype and index name than the yfinance path.
    """
    df = _ohlcv()
    schwab = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    schwab["Volume"] = schwab["Volume"].astype("int64")
    schwab.index = pd.DatetimeIndex(schwab.index, name="Date")

    triggers = build_triggers(schwab, "oversold")
    assert len(triggers) == 8, [t.key for t in triggers]
    assert "volume" in [t.key for t in triggers]
    for t in triggers:
        assert t.mask.dtype == bool


def test_triggers_need_enough_history():
    thin = pd.DataFrame({"Close": np.linspace(100, 110, 20)},
                        index=pd.bdate_range("2024-01-01", periods=20))
    assert build_triggers(thin, "oversold") == []


def test_masks_are_real_booleans():
    """Shift and reindex leave object dtype with NaNs; a NaN in a mask silently
    poisons every count computed from it."""
    for t in build_triggers(_ohlcv(), "oversold"):
        assert t.mask.dtype == bool, f"{t.key} mask is {t.mask.dtype}"
        assert not t.mask.isna().any()


def test_showing_window_matches_recent_bars():
    df = _ohlcv()
    for t in build_triggers(df, "oversold"):
        bars = t.bars_since()
        assert t.showing() == (bars is not None and bars <= RECENT_BARS)


# ── Waiting for confirmation ────────────────────────────────────────────


def test_entry_is_the_confirmation_bar_not_the_setup_bar():
    """The whole design decision: you buy when the signal appears, not when the
    setup does. Intersecting the two instead would measure something else."""
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100.0, index=idx)
    # Flat everywhere except a jump that starts 4 bars after each setup.
    mask = pd.Series(False, index=idx)
    setups = [50, 150]
    for p in setups:
        mask.iloc[p + 4] = True
        close.iloc[p + 4:] = close.iloc[p + 4:] + 10.0

    bench = pd.Series(100.0, index=idx)
    s = confirmation_stats(close, bench, mask, setups, horizon=5)
    assert s is not None
    assert s["median_wait"] == 4.0, "wait must be measured to the confirmation bar"
    assert s["arrival_rate"] == 1.0


def test_setup_bar_itself_counts_as_zero_wait():
    """If the confirmation is already showing when the setup appears, you act
    immediately — that is a wait of 0, not a miss."""
    idx = pd.bdate_range("2020-01-01", periods=200)
    close = pd.Series(np.linspace(100, 120, 200), index=idx)
    mask = pd.Series(True, index=idx)
    s = confirmation_stats(close, pd.Series(100.0, index=idx), mask, [40, 90], horizon=5)
    assert s["median_wait"] == 0.0
    assert s["arrival_rate"] == 1.0


def test_setups_without_a_confirmation_lower_the_arrival_rate():
    """Trades not taken must be visible. Dropping them silently would make a
    rare signal look as available as a common one."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(np.linspace(100, 130, 300), index=idx)
    mask = pd.Series(False, index=idx)
    mask.iloc[54] = True                      # only the first setup ever confirms
    s = confirmation_stats(close, pd.Series(100.0, index=idx), mask,
                           [50, 150, 200, 250], horizon=5)
    assert s["arrival_rate"] == 0.25
    assert s["setups_considered"] == 4


def test_confirmation_beyond_the_wait_window_does_not_count():
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(np.linspace(100, 130, 300), index=idx)
    mask = pd.Series(False, index=idx)
    mask.iloc[50 + MAX_WAIT + 1] = True       # one session too late
    s = confirmation_stats(close, pd.Series(100.0, index=idx), mask, [50], horizon=5)
    assert s["arrival_rate"] == 0.0
    assert s["n"] == 0


def test_confirmation_stats_is_safe_without_setups():
    idx = pd.bdate_range("2020-01-01", periods=100)
    close = pd.Series(100.0, index=idx)
    assert confirmation_stats(close, close, pd.Series(True, index=idx), [], 5) is None


def test_overlapping_entries_are_collapsed():
    """Neighbouring setups resolving to nearby bars share almost all of their
    forward window; counting both would inflate n on one price move."""
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(np.linspace(100, 130, 300), index=idx)
    mask = pd.Series(True, index=idx)
    s = confirmation_stats(close, pd.Series(100.0, index=idx), mask,
                           [100, 101, 102, 103], horizon=10)
    assert s["arrival_rate"] == 1.0
    assert s["n"] == 1, "four overlapping entries are one independent event"


# ── Reporting ───────────────────────────────────────────────────────────


def test_thin_samples_print_no_number():
    """A 2-occurrence '+8%' would sort to the top of the table and mean nothing."""
    idea = {
        "ticker": "TEST",
        "stats": {5: {"expectancy": 0.01}},
        "confirmations": [{
            "label": "x", "where": "y", "detail": "z", "showing": False,
            "bars_since": 40,
            "stats": {5: {"arrival_rate": 0.3, "median_wait": 2.0,
                          "n": MIN_ANALOGS - 1, "sufficient": False,
                          "expectancy": 0.08}},
        }],
    }
    row = confirmation_table(idea, 5)[0]
    assert row["5d vs SPY"] == "too few"
    assert row["vs not waiting"] == "too few"
    assert row["n"] == MIN_ANALOGS - 1


def test_delta_against_not_waiting_is_scaled_to_percentage_points():
    """expectancy is a FRACTION. The same slip once rendered every lift on this
    page as '+0pp'."""
    idea = {
        "ticker": "TEST",
        "stats": {10: {"expectancy": 0.01}},        # +1.00% buying immediately
        "confirmations": [{
            "label": "x", "where": "y", "detail": "z", "showing": True,
            "bars_since": 0,
            "stats": {10: {"arrival_rate": 0.5, "median_wait": 3.0,
                           "n": 25, "sufficient": True, "expectancy": 0.025}},
        }],
    }
    row = confirmation_table(idea, 10)[0]
    assert row["10d vs SPY"] == "+2.50%"
    assert row["vs not waiting"] == "+1.5pp"


def test_compact_table_drops_columns_for_phones():
    idea = {
        "ticker": "TEST", "stats": {5: {"expectancy": 0.0}},
        "confirmations": [{
            "label": "x", "where": "y", "detail": "z", "showing": True,
            "bars_since": 1, "stats": {},
        }],
    }
    assert "Latest reading" in confirmation_table(idea, 5)[0]
    assert "Latest reading" not in confirmation_table(idea, 5, compact=True)[0]


@pytest.mark.parametrize(
    "row,expected",
    [
        ({"bars_since": 0, "showing": True}, "✅ today"),
        ({"bars_since": 3, "showing": True}, "✅ 3d ago"),
        ({"bars_since": 40, "showing": False}, "— 40d ago"),
        ({"bars_since": None, "showing": False}, "— never"),
    ],
)
def test_status_text(row, expected):
    assert status_text(row) == expected


# ── End to end ──────────────────────────────────────────────────────────


def test_attach_confirmations_populates_an_idea():
    df = _ohlcv(n=2000)
    bench = _bench(df.index)
    idea = evaluate_ticker(df, bench, ticker="TEST", with_confirmations=True)

    assert idea is not None
    rows = idea["confirmations"]
    assert rows, "no confirmations attached"
    assert idea["confirmations_showing"] == sum(1 for r in rows if r["showing"])
    for r in rows:
        assert r["label"] and r["where"] and r["detail"]
        for h, s in r["stats"].items():
            assert 0.0 <= s["arrival_rate"] <= 1.0
            assert s["n"] <= s["setups_considered"]


def test_confirmations_are_ordered_by_visibility_not_by_return():
    """Sorting on the best forward number across ~250 comparisons would surface
    noise every day. What is showing now comes first, full stop."""
    df = _ohlcv(n=2000, seed=8)
    idea = evaluate_ticker(df, _bench(df.index), ticker="TEST", with_confirmations=True)
    showing = [r["showing"] for r in idea["confirmations"]]
    assert showing == sorted(showing, reverse=True)


def test_confirmations_absent_unless_requested():
    """The close-only scan path must stay exactly as it was."""
    df = _ohlcv(n=2000)
    idea = evaluate_ticker(df, _bench(df.index), ticker="TEST")
    assert "confirmations" not in idea


def test_direction_drives_which_checklist_is_attached():
    df = _ohlcv(n=2000)
    idea = evaluate_ticker(df, _bench(df.index), ticker="TEST", with_confirmations=True)
    keys = {r["key"] for r in idea["confirmations"]}
    if idea["direction"] == "oversold":
        assert "rsi_up_with_price" in keys and "rsi_down_with_price" not in keys
    else:
        assert "rsi_down_with_price" in keys and "rsi_up_with_price" not in keys


def test_attach_is_safe_on_a_frame_too_thin_for_triggers():
    idea = {"ticker": "T", "direction": "oversold", "rsi_percentile": 5.0}
    thin = pd.DataFrame({"Close": np.linspace(100, 110, 20)},
                        index=pd.bdate_range("2024-01-01", periods=20))
    out = attach_confirmations(idea, thin, pd.Series(100.0, index=thin.index), {5: [1]})
    assert out["confirmations"] == []
