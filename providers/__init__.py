"""
Provider factory.

`get_provider()` returns the active DataProvider based on config.DATA_PROVIDER
(which itself honors the DATA_PROVIDER env var). Imports are lazy so the
yfinance path never requires schwab-py to be installed, and vice-versa.

Phase 1 default is "yfinance" — zero behavior change. Flip DATA_PROVIDER to
"schwab" (env or config) once Schwab credentials exist.
"""

from typing import Optional

from .base import DataProvider

try:
    from config import DATA_PROVIDER as _CONFIG_DEFAULT
except Exception:  # pragma: no cover - config should always import
    _CONFIG_DEFAULT = "yfinance"

# Cache one instance per backend name so we don't re-create providers (and,
# for Schwab, re-load tokens) on every fetch.
_INSTANCES: dict = {}


def _build(name: str) -> DataProvider:
    if name == "yfinance":
        from .yfinance_provider import YFinanceProvider
        return YFinanceProvider()
    if name == "schwab":
        # Schwab is primary, yfinance is the automatic safety net: if a Schwab
        # call raises (expired token) or returns data that fails the quality
        # gate, the fetch transparently degrades to yfinance. This is what keeps
        # the pipeline autonomous across the weekly Schwab re-auth window.
        from .schwab_provider import SchwabProvider
        from .yfinance_provider import YFinanceProvider
        from .fallback_provider import FallbackProvider
        return FallbackProvider(SchwabProvider(), YFinanceProvider())
    raise ValueError(
        f"Unknown DATA_PROVIDER {name!r}. Expected 'yfinance' or 'schwab'."
    )


def get_provider(name: Optional[str] = None) -> DataProvider:
    """
    Return the active data provider (singleton per backend).

    `name` overrides config when given; otherwise config.DATA_PROVIDER decides.
    """
    chosen = (name or _CONFIG_DEFAULT or "yfinance").lower()
    if chosen not in _INSTANCES:
        _INSTANCES[chosen] = _build(chosen)
    return _INSTANCES[chosen]


__all__ = ["DataProvider", "get_provider"]
