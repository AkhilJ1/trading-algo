import os
from datetime import date

import pandas as pd

from config import CACHE_DIR, DATA_INTERVAL, DATA_PERIOD
from providers import get_provider


def fetch_stock_data(
    ticker: str,
    period: str = DATA_PERIOD,
    interval: str = DATA_INTERVAL,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV daily data for a ticker via yfinance.
    Results are cached per-day so repeated calls within the same day are instant.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    today = date.today().isoformat()
    cache_file = os.path.join(CACHE_DIR, f"{ticker}_{period}_{interval}_{today}.csv")

    if use_cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Ensure index is a timezone-naive DatetimeIndex after CSV round-trip
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        elif df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        # The per-day cache key keeps yesterday's file from being reused today,
        # but a file first written EARLIER today still holds today's *partial*
        # bar — e.g. an intraday low (742) that the rest of the session blew
        # through (real low 738, close 739). Serving that stale candle makes
        # the chart's low never reach the true session low. If the newest
        # cached bar is dated today, fall through and re-fetch so the current
        # day's OHLC reflects the latest prints; completed prior days never
        # change, so every other day still hits the cache instantly.
        cache_has_today = (
            isinstance(df.index, pd.DatetimeIndex)
            and len(df) > 0
            and df.index[-1].date() >= date.today()
        )
        if not cache_has_today:
            return df

    df = get_provider().get_price_history(ticker, period=period, interval=interval)
    if df.empty:
        return pd.DataFrame()

    # Normalize column names (yfinance is inconsistent across versions)
    df.columns = [c.strip().title() for c in df.columns]
    # Strip timezone before caching so CSV round-trips cleanly
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # Drop rows with NaN Close (yfinance returns NaN for today before market close)
    df = df.dropna(subset=['Close'])
    df.to_csv(cache_file)
    return df


def fetch_multiple(
    tickers: list,
    period: str = DATA_PERIOD,
    interval: str = DATA_INTERVAL,
) -> dict:
    """Fetch data for a list of tickers; returns {ticker: DataFrame}."""
    results = {}
    for ticker in tickers:
        df = fetch_stock_data(ticker, period, interval)
        if not df.empty:
            results[ticker] = df
    return results
