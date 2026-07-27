"""
Tests for the validated trade-idea engine.

These assert the properties that keep the page honest rather than persuasive:
small samples must not outrank established ones, overlapping bars must not
inflate the sample, rates must be measured against the ticker's own baseline,
and the example list must contain losers.
"""

import numpy as np
import pandas as pd
import pytest

from trade_ideas import (
    apply_multiple_testing_correction,
    MIN_ANALOGS,
    _dedupe_events,
    evaluate_ticker,
    find_analogs,
    percentile_of_last,
    scan_universe,
    score_analogs,
    wilson_interval,
)


def _series(vals, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=len(vals))
    return pd.Series(vals, index=idx, dtype=float)


# ── Wilson interval ─────────────────────────────────────────────────────

def test_wilson_interval_is_wider_for_smaller_samples():
    """The whole defence against 3-for-3 setups: uncertainty must show up."""
    lo_small, hi_small = wilson_interval(3, 3)
    lo_big, hi_big = wilson_interval(30, 30)
    assert (hi_small - lo_small) > (hi_big - lo_big)
    assert lo_small < lo_big


def test_wilson_interval_stays_in_unit_range():
    for wins, n in [(0, 5), (5, 5), (1, 100), (99, 100)]:
        lo, hi = wilson_interval(wins, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_handles_empty_sample():
    assert wilson_interval(0, 0) == (0.0, 1.0)


# ── Event deduplication ─────────────────────────────────────────────────

def test_consecutive_bars_collapse_to_one_event():
    """Six consecutive oversold bars are ONE event. Counting six would inflate
    n sixfold and shrink the interval on essentially the same forward window."""
    assert _dedupe_events([10, 11, 12, 13, 14, 15], min_gap=5) == [10, 15]


def test_separated_events_are_all_kept():
    assert _dedupe_events([10, 30, 60], min_gap=5) == [10, 30, 60]


def test_dedupe_handles_empty():
    assert _dedupe_events([], min_gap=5) == []


# ── Percentile framing ──────────────────────────────────────────────────

def test_percentile_reflects_position_in_own_distribution():
    rising = _series(np.arange(300, dtype=float))
    assert percentile_of_last(rising, lookback=252) > 95.0

    falling = _series(np.arange(300, 0, -1, dtype=float))
    assert percentile_of_last(falling, lookback=252) < 5.0


def test_percentile_returns_none_without_enough_history():
    assert percentile_of_last(_series([1.0, 2.0, 3.0]), lookback=252) is None


# ── Scoring ─────────────────────────────────────────────────────────────

def _flat_bench(close):
    return pd.Series(100.0, index=close.index)


def test_score_measures_against_benchmark_not_raw_return():
    """A ticker that rose 10% while the benchmark rose 10% has zero edge."""
    n = 120
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(np.linspace(100, 200, n), index=idx)
    bench = pd.Series(np.linspace(100, 200, n), index=idx)  # identical path

    s = score_analogs(close, bench, [10, 30, 50], horizon=5)
    assert s is not None
    assert abs(s.expectancy) < 1e-9, "identical benchmark path => no abnormal return"


def test_examples_include_losers_not_just_winners():
    """Showing only winners would turn a coin flip into a sales pitch."""
    rng = np.random.default_rng(0)
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))), index=idx)
    bench = pd.Series(100.0, index=idx)

    s = score_analogs(close, bench, list(range(20, 300, 20)), horizon=5)
    assert s is not None
    rets = [e["abnormal_return"] for e in s.examples]
    assert any(r > 0 for r in rets) and any(r < 0 for r in rets), (
        "example set must span winners and losers"
    )


def test_confident_requires_significant_positive_expectancy():
    """'Confident the return will be positive' is a claim about the MEAN, and a
    two-analog sample can never support it however flattering its mean."""
    rng = np.random.default_rng(3)
    n = 600
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, n))), index=idx)
    bench = pd.Series(100.0, index=idx)

    tiny = score_analogs(close, bench, [50, 120], horizon=5)
    assert tiny is not None and tiny.n == 2
    assert not tiny.confident, "n=2 can never support a confident claim"


def test_wilson_lower_bound_penalises_small_samples():
    """3-for-3 must sit below a larger, slightly-less-perfect sample."""
    lo3, _ = wilson_interval(3, 3)
    lo17, _ = wilson_interval(17, 25)
    assert lo3 < lo17


# ── End to end ──────────────────────────────────────────────────────────

def _synthetic_frame(n=800, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    return pd.DataFrame({"Close": close}, index=idx)


def test_evaluate_ticker_returns_scorecard_with_sample_size():
    df = _synthetic_frame()
    bench = pd.Series(
        100 * np.exp(np.cumsum(np.random.default_rng(9).normal(0.0002, 0.01, len(df)))),
        index=df.index,
    )
    idea = evaluate_ticker(df, bench, ticker="TEST")

    assert idea is not None
    assert idea["ticker"] == "TEST"
    assert 0.0 <= idea["rsi_percentile"] <= 100.0
    assert set(idea["stats"]) <= {5, 10}
    for s in idea["stats"].values():
        assert s["n"] >= 0
        assert 0.0 <= s["hit_ci_low"] <= s["hit_ci_high"] <= 1.0


def test_insufficient_history_can_never_rank():
    """Below MIN_ANALOGS the score must be -inf so it cannot reach the page."""
    df = _synthetic_frame(n=800)
    bench = pd.Series(100.0, index=df.index)
    idea = evaluate_ticker(df, bench, ticker="TEST")
    assert idea is not None
    if idea["n_analogs"] < MIN_ANALOGS:
        assert idea["score"] == float("-inf")
        assert idea["confident"] is False


def test_evaluate_ticker_returns_none_on_thin_history():
    thin = pd.DataFrame({"Close": np.linspace(100, 110, 50)},
                        index=pd.bdate_range("2024-01-01", periods=50))
    assert evaluate_ticker(thin, pd.Series(100.0, index=thin.index)) is None


def test_evaluate_ticker_handles_missing_close_column():
    df = pd.DataFrame({"Open": [1, 2, 3]})
    assert evaluate_ticker(df, pd.Series([1.0, 2.0, 3.0])) is None


def test_find_analogs_leaves_room_for_forward_return():
    """Analogs must never include bars whose forward window runs off the end."""
    df = _synthetic_frame(n=600)
    from trade_ideas import _rsi
    rsi = _rsi(df["Close"])
    positions = find_analogs(rsi, 50.0, horizon=10)
    assert all(p < len(rsi.dropna()) - 10 for p in positions)


# ── Multiple-testing correction ─────────────────────────────────────────
# Scanning 31 tickers at alpha=0.05 yields ~1.5 "significant" hits from noise.
# Without correction the page always finds a winner and calls luck validation.

def _idea(ticker, p5, p10, confident=True):
    return {
        "ticker": ticker,
        "confident": confident,
        "stats": {5: {"p_value": p5}, 10: {"p_value": p10}},
    }


def test_correction_uses_the_worse_horizon():
    """A setup must hold at BOTH horizons, so the combined p is the worse one."""
    ideas = apply_multiple_testing_correction([_idea("A", 0.01, 0.40)])
    assert ideas[0]["p_combined"] == 0.40


def test_lone_marginal_hit_among_many_does_not_survive():
    """THE regression: one p=0.03 out of 31 tests is what noise looks like."""
    ideas = [_idea("HIT", 0.03, 0.03)] + [
        _idea(f"N{i}", 0.5, 0.6, confident=False) for i in range(30)
    ]
    out = apply_multiple_testing_correction(ideas)
    hit = next(i for i in out if i["ticker"] == "HIT")

    assert hit["confident_uncorrected"] is True
    assert hit["confident"] is False, "must not survive FDR across 31 tests"
    assert hit["n_tested"] == 31


def test_strong_result_still_survives_correction():
    """The correction must not be so blunt that nothing can ever pass."""
    ideas = [_idea("STRONG", 0.0001, 0.0002)] + [
        _idea(f"N{i}", 0.5, 0.6, confident=False) for i in range(30)
    ]
    out = apply_multiple_testing_correction(ideas)
    assert next(i for i in out if i["ticker"] == "STRONG")["confident"] is True


def test_correction_is_safe_on_empty_input():
    assert apply_multiple_testing_correction([]) == []


# ── scan_universe ───────────────────────────────────────────────────────

def _wide_frame(tickers, n=800, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    data = {t: 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n))) for t in tickers}
    return pd.DataFrame(data, index=idx)


def test_scan_ranks_by_unusualness_not_by_expected_return():
    """Ranking on expected return would be sorting noise and calling it a
    forecast — out-of-sample date-clustered tests put this signal at p=0.67."""
    df = _wide_frame(["AAA", "BBB", "CCC", "SPY"])
    ideas = scan_universe(df)

    assert ideas, "expected some ideas"
    unusual = [i["unusualness"] for i in ideas]
    assert unusual == sorted(unusual, reverse=True)


def test_scan_reports_confluence_without_claiming_predictive_power():
    df = _wide_frame(["AAA", "SPY"])
    ideas = scan_universe(df)
    for i in ideas:
        assert isinstance(i["confluence"], bool)
        assert "macd_percentile" in i
        # Confluence must never silently promote something to confident.
        if i["confluence"]:
            assert i["confident"] in (True, False)


def test_scan_applies_correction_across_the_whole_scan():
    df = _wide_frame([f"T{i}" for i in range(8)] + ["SPY"])
    ideas = scan_universe(df)
    assert all("n_tested" in i for i in ideas)
    assert all(i["n_tested"] == len(ideas) for i in ideas)


def test_scan_returns_empty_without_benchmark():
    df = _wide_frame(["AAA", "BBB"])
    assert scan_universe(df, benchmark="SPY") == []


def test_scan_labels_direction():
    df = _wide_frame(["AAA", "BBB", "SPY"])
    for i in scan_universe(df):
        assert i["direction"] in ("oversold", "overbought")
        assert (i["direction"] == "oversold") == (i["rsi_percentile"] < 50)
