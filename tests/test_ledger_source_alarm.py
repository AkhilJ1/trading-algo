"""
Tests for the ledger source alarm.

The bug this guards against is not hypothetical: between 2026-06 and 2026-07
every recorded forecast came from yfinance while the pipeline believed it was on
Schwab, and nothing complained because the monitoring watched token age instead
of what actually got written. These tests assert the alarm fires on that exact
shape, and — just as importantly — does not fire on the benign ones.
"""

import pandas as pd

from ledger_source_alarm import check_ledger


def _ledger(rows):
    """rows = [(date, chain_source), ...]"""
    return pd.DataFrame(
        [{"date": d, "chain_source": s, "ticker": "SPY"} for d, s in rows]
    )


def test_alarms_when_every_recent_session_fell_back():
    """THE regression: recording continues, but from the wrong backend."""
    df = _ledger([
        ("2026-07-20", "yfinance"),
        ("2026-07-21", "yfinance"),
        ("2026-07-22", "yfinance"),
    ])
    problem, _ = check_ledger(df, "PinForecasts", "schwab", 3)

    assert problem is not None
    assert "schwab" in problem
    assert "never" in problem, "no good session ever -> should say so"


def test_reports_the_last_good_session_so_the_outage_can_be_dated():
    df = _ledger([
        ("2026-06-18", "schwab"),
        ("2026-07-20", "yfinance"),
        ("2026-07-21", "yfinance"),
        ("2026-07-22", "yfinance"),
    ])
    problem, _ = check_ledger(df, "PinForecasts", "schwab", 3)

    assert problem is not None
    assert "2026-06-18" in problem


def test_healthy_when_recent_sessions_are_on_source():
    df = _ledger([
        ("2026-07-20", "schwab"),
        ("2026-07-21", "schwab"),
        ("2026-07-22", "schwab"),
    ])
    problem, detail = check_ledger(df, "PinForecasts", "schwab", 3)

    assert problem is None
    assert "ok" in detail


def test_single_fallback_session_is_tolerated():
    """A thin pre-open chain or a transient 5xx must not page anyone."""
    df = _ledger([
        ("2026-07-20", "schwab"),
        ("2026-07-21", "yfinance"),
        ("2026-07-22", "schwab"),
    ])
    problem, _ = check_ledger(df, "PinForecasts", "schwab", 3)
    assert problem is None


def test_mixed_rows_in_one_session_count_as_on_source():
    """Recorders write several rows a day; one bad intraday run is not an
    outage as long as that session also produced a good row."""
    df = _ledger([
        ("2026-07-20", "yfinance"),
        ("2026-07-20", "schwab"),
        ("2026-07-21", "yfinance"),
        ("2026-07-21", "schwab"),
        ("2026-07-22", "schwab"),
    ])
    problem, _ = check_ledger(df, "PinForecasts", "schwab", 3)
    assert problem is None


def test_sessions_not_rows_define_the_window():
    """Three rows on ONE day must not satisfy a three-SESSION window — that is
    how a row-based check silently shrinks to a single trading day."""
    df = _ledger([
        ("2026-07-22", "yfinance"),
        ("2026-07-22", "yfinance"),
        ("2026-07-22", "yfinance"),
    ])
    problem, detail = check_ledger(df, "PinForecasts", "schwab", 3)

    assert problem is None, "one session is not enough evidence to alarm"
    assert "too early" in detail


def test_too_few_sessions_does_not_alarm():
    """A freshly reset ledger must not page anyone on day one."""
    df = _ledger([("2026-07-22", "yfinance")])
    problem, detail = check_ledger(df, "PinForecasts", "schwab", 3)

    assert problem is None
    assert "too early" in detail


def test_empty_ledger_is_not_a_problem():
    problem, detail = check_ledger(pd.DataFrame(), "PinForecasts", "schwab", 3)
    assert problem is None
    assert "nothing to check" in detail


def test_missing_chain_source_column_is_not_a_problem():
    """Rows predating provenance columns must not be read as an outage."""
    df = pd.DataFrame([{"date": "2026-07-22", "ticker": "SPY"}])
    problem, _ = check_ledger(df, "Predictions", "schwab", 3)
    assert problem is None


def test_source_comparison_is_case_and_whitespace_insensitive():
    df = _ledger([
        ("2026-07-20", " Schwab "),
        ("2026-07-21", "SCHWAB"),
        ("2026-07-22", "schwab"),
    ])
    problem, _ = check_ledger(df, "PinForecasts", "schwab", 3)
    assert problem is None
