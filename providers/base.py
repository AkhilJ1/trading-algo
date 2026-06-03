"""
Data provider abstraction.

A provider is the *only* thing that talks to an external market-data source.
Everything above it (caching, column normalization, futures-proxy resolution,
NaN handling, meta construction) lives in data_fetcher.py / options_fetcher.py
and is provider-agnostic.

The contract is deliberately thin and modeled on what yfinance already returns,
so that swapping in a different backend (e.g. Charles Schwab) is a drop-in change
with zero downstream impact:

  * get_price_history(...) -> a DataFrame shaped like yf.Ticker.history():
        DatetimeIndex, columns include at least Open/High/Low/Close/Volume
        (any capitalization — the caller title-cases them).
  * get_expirations(...)   -> list of 'YYYY-MM-DD' expiry strings.
  * get_option_chain(...)  -> (calls_df, puts_df) shaped like
        yfinance's chain.calls / chain.puts, each carrying at least:
        strike, openInterest, volume, impliedVolatility, bid, ask, lastPrice.

Providers must NOT do futures→ETF proxy mapping or symbol-specific rewrites that
the rest of the app expects to control; the one exception is a backend that uses
a different symbology (e.g. Schwab's '$VIX'), in which case the provider
translates *its own* wire symbols internally and still accepts the app's symbols
('^VIX') at the boundary.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd


class DataProvider(ABC):
    """Abstract market-data backend."""

    #: short identifier, e.g. "yfinance" or "schwab"
    name: str = "abstract"

    @abstractmethod
    def get_price_history(
        self,
        ticker: str,
        period: str,
        interval: str,
    ) -> pd.DataFrame:
        """
        Return OHLCV history for `ticker`.

        Shape: a pandas DataFrame indexed by date/datetime with columns that
        (case-insensitively) include Open, High, Low, Close, Volume. Returning
        an empty DataFrame signals "no data" — callers handle that gracefully.
        """
        raise NotImplementedError

    @abstractmethod
    def get_expirations(self, ticker: str) -> List[str]:
        """Return available option expiration dates as 'YYYY-MM-DD' strings."""
        raise NotImplementedError

    @abstractmethod
    def get_option_chain(
        self,
        ticker: str,
        expiry: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return (calls_df, puts_df) for `ticker` at `expiry`.

        Each frame must carry at least these columns:
            strike, openInterest, volume, impliedVolatility, bid, ask, lastPrice
        Missing volume/openInterest may be NaN — the caller fills them.
        """
        raise NotImplementedError
