"""
FallbackProvider — a primary data source with automatic degrade to a secondary.

Used when DATA_PROVIDER=schwab: serves Schwab when its data passes the quality
gate, otherwise transparently falls back to yfinance so the pipeline never breaks
(Requirement 2 — full autonomy even across the weekly Schwab re-auth gap).

It records which backend actually served each call type in `last_source`, so the
data-health check and the option-chain meta can report the true source
(Requirement 3 — know where every number came from).
"""

from typing import Callable, List, Optional, Tuple

import pandas as pd

from .base import DataProvider
from .quality import price_is_usable, chain_is_usable


class FallbackProvider(DataProvider):
    def __init__(self, primary: DataProvider, secondary: DataProvider):
        self.primary = primary
        self.secondary = secondary
        self.name = f"{primary.name}->{secondary.name}"
        # {'price'|'expirations'|'chain': '<backend name>' or None}
        self.last_source: dict = {}

    def _try(self, key: str, fn_name: str, validator: Callable, *args, **kwargs):
        """Try primary; if it raises or fails the validator, use secondary."""
        try:
            res = getattr(self.primary, fn_name)(*args, **kwargs)
            if validator(res):
                self.last_source[key] = self.primary.name
                return res
        except Exception:
            pass
        try:
            res = getattr(self.secondary, fn_name)(*args, **kwargs)
            self.last_source[key] = self.secondary.name
            return res
        except Exception:
            self.last_source[key] = None
            raise

    def get_quote(self, ticker: str) -> Optional[float]:
        """Live quote from primary, degrading to secondary, then None.

        A usable quote is a positive float. Unlike the other routes this never
        raises — a missing live quote is a soft miss (the caller falls back to
        the daily close), not a pipeline failure.
        """
        def _usable(px) -> bool:
            try:
                return px is not None and float(px) > 0
            except (TypeError, ValueError):
                return False

        try:
            res = self.primary.get_quote(ticker)
            if _usable(res):
                self.last_source["quote"] = self.primary.name
                return float(res)
        except Exception:
            pass
        try:
            res = self.secondary.get_quote(ticker)
            if _usable(res):
                self.last_source["quote"] = self.secondary.name
                return float(res)
        except Exception:
            pass
        self.last_source["quote"] = None
        return None

    def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        return self._try(
            "price", "get_price_history", price_is_usable, ticker, period, interval
        )

    def get_expirations(self, ticker: str) -> List[str]:
        return self._try("expirations", "get_expirations", lambda x: bool(x), ticker)

    def get_option_chain(self, ticker: str, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        def _ok(res) -> bool:
            try:
                calls, puts = res
            except Exception:
                return False
            return chain_is_usable(calls, puts)

        return self._try("chain", "get_option_chain", _ok, ticker, expiry)
