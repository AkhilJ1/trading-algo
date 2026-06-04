"""Tests for the provider factory (providers.get_provider).

The factory is the one switch that decides which backend the whole pipeline
talks to (Requirement 5). It must: build yfinance by default, build the
Schwab->yfinance fallback when asked, reject unknown names, and hand back a
singleton per backend so we don't re-load Schwab tokens on every fetch.
"""

import pytest

import providers
from providers import get_provider
from providers.base import DataProvider


@pytest.fixture(autouse=True)
def _clear_instance_cache():
    # The factory memoizes instances in a module global; isolate each test.
    providers._INSTANCES.clear()
    yield
    providers._INSTANCES.clear()


def test_yfinance_built_by_name():
    p = get_provider("yfinance")
    assert isinstance(p, DataProvider)
    assert p.name == "yfinance"


def test_schwab_name_builds_fallback_wrapper():
    # "schwab" must NOT return a bare SchwabProvider — it returns the fallback
    # wrapper so the pipeline degrades to yfinance across the re-auth gap.
    p = get_provider("schwab")
    assert p.name == "schwab->yfinance"


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_provider("bloomberg")


def test_singleton_per_backend():
    a = get_provider("yfinance")
    b = get_provider("yfinance")
    assert a is b


def test_distinct_backends_are_distinct_instances():
    y = get_provider("yfinance")
    s = get_provider("schwab")
    assert y is not s
    assert y.name == "yfinance"
    assert s.name == "schwab->yfinance"


def test_name_is_case_insensitive():
    a = get_provider("YFINANCE")
    b = get_provider("yfinance")
    # Both normalize to the same cache key -> same singleton.
    assert a is b
