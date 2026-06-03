"""
yfinance-backed DataProvider.

This is a faithful 1:1 wrapper around the exact yfinance calls that
data_fetcher.py and options_fetcher.py used to make inline. It performs NO
normalization, NaN-filling, caching, or proxy resolution — those stay in the
fetchers so behavior is byte-for-byte identical to the pre-refactor code.
"""

from typing import List, Tuple

import pandas as pd
import yfinance as yf

from .base import DataProvider


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def get_price_history(
        self,
        ticker: str,
        period: str,
        interval: str,
    ) -> pd.DataFrame:
        # Mirrors: yf.Ticker(ticker).history(period=period, interval=interval)
        return yf.Ticker(ticker).history(period=period, interval=interval)

    def get_expirations(self, ticker: str) -> List[str]:
        # Mirrors: list(yf.Ticker(resolved).options)
        try:
            return list(yf.Ticker(ticker).options)
        except Exception:
            return []

    def get_option_chain(
        self,
        ticker: str,
        expiry: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Mirrors: chain = t.option_chain(expiry); chain.calls, chain.puts
        chain = yf.Ticker(ticker).option_chain(expiry)
        return chain.calls.copy(), chain.puts.copy()
