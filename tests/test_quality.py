"""Tests for providers.quality — the single source of truth for 'is this data
good enough to trust' (Requirement 3)."""

import numpy as np
import pandas as pd

from providers.quality import (
    price_is_usable, chain_is_usable, _usable_count,
    MIN_USABLE_STRIKES, MIN_USABLE_IV,
)
from tests.conftest import make_chain, make_price_df


# ── price_is_usable ────────────────────────────────────────────────────────
def test_price_usable_positive_close():
    assert price_is_usable(make_price_df([100.0, 105.0])) is True


def test_price_unusable_when_none():
    assert price_is_usable(None) is False


def test_price_unusable_when_empty():
    assert price_is_usable(pd.DataFrame()) is False


def test_price_unusable_without_close_column():
    df = pd.DataFrame({"Open": [1.0], "High": [2.0]})
    assert price_is_usable(df) is False


def test_price_unusable_when_last_close_zero_or_negative():
    assert price_is_usable(make_price_df([100.0, 0.0])) is False
    assert price_is_usable(make_price_df([100.0, -5.0])) is False


def test_price_unusable_when_last_close_nan():
    df = make_price_df([100.0, 101.0])
    df.iloc[-1, df.columns.get_loc("Close")] = np.nan
    assert price_is_usable(df) is False


# ── chain_is_usable ────────────────────────────────────────────────────────
def test_chain_usable_with_enough_real_strikes():
    # 12 + 12 = 24 usable rows >= MIN_USABLE_STRIKES (20)
    assert chain_is_usable(make_chain(12), make_chain(12)) is True


def test_chain_unusable_when_too_few_strikes():
    # 5 + 5 = 10 < 20
    assert chain_is_usable(make_chain(5), make_chain(5)) is False


def test_chain_unusable_when_iv_below_threshold():
    # Plenty of rows but IV at/below the 5% floor => stale/missing => unusable.
    low_iv = make_chain(30, iv=MIN_USABLE_IV)  # not strictly greater than floor
    assert chain_is_usable(low_iv, make_chain(0)) is False


def test_chain_unusable_when_open_interest_zero():
    no_oi = make_chain(30, oi=0)
    assert chain_is_usable(no_oi, make_chain(0)) is False


def test_chain_unusable_when_either_side_none():
    assert chain_is_usable(None, make_chain(30)) is False
    assert chain_is_usable(make_chain(30), None) is False


def test_chain_unusable_when_both_empty():
    assert chain_is_usable(pd.DataFrame(), pd.DataFrame()) is False


def test_usable_count_handles_missing_columns():
    df = pd.DataFrame({"strike": [1, 2, 3]})  # no IV/OI columns
    assert _usable_count(df) == 0


def test_usable_count_coerces_garbage_to_zero():
    df = make_chain(10)
    # Cast to object so we can inject non-numeric garbage without a pandas
    # dtype-incompatibility warning; the point is that quality coerces it to 0.
    df["impliedVolatility"] = df["impliedVolatility"].astype(object)
    df["openInterest"] = df["openInterest"].astype(object)
    df.loc[0, "impliedVolatility"] = "not-a-number"
    df.loc[1, "openInterest"] = None
    # 10 rows, two corrupted -> 8 usable
    assert _usable_count(df) == 8


def test_chain_counts_split_across_both_sides():
    # 19 + 1 = 20 -> exactly at the threshold, should be usable.
    assert chain_is_usable(make_chain(19), make_chain(1)) is True
    # 19 + 0 -> 19 < 20, not usable.
    assert chain_is_usable(make_chain(19), make_chain(0)) is False
