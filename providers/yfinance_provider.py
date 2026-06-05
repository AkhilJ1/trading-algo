"""
yfinance-backed DataProvider.

This is a faithful 1:1 wrapper around the exact yfinance calls that
data_fetcher.py and options_fetcher.py used to make inline. It performs NO
normalization, NaN-filling, caching, or proxy resolution — those stay in the
fetchers so behavior is byte-for-byte identical to the pre-refactor code.
"""

from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf

from .base import DataProvider


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def get_quote(self, ticker: str) -> Optional[float]:
        """Latest trade price including pre/post-market.

        Pulls 1-minute bars with prepost=True and returns the last close — in
        the pre-open window this is the most recent pre-market print, which is
        a far better spot anchor than yesterday's settled daily close. Returns
        None on any failure so the caller degrades to the daily close.
        """
        try:
            hist = yf.Ticker(ticker).history(
                period="1d", interval="1m", prepost=True
            )
            if hist is None or hist.empty or "Close" not in hist.columns:
                return None
            px = float(hist["Close"].dropna().iloc[-1])
            return px if px > 0 else None
        except Exception:
            return None

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
