"""
Charles Schwab–backed DataProvider (Trader API / Market Data Production).

DORMANT BY DEFAULT. Nothing imports this unless DATA_PROVIDER=schwab. It is a
faithful translation of Schwab's JSON into the exact shapes the rest of the app
already consumes from the yfinance provider:

  * get_price_history -> DataFrame[Open,High,Low,Close,Volume] on a DatetimeIndex
  * get_expirations   -> ['YYYY-MM-DD', ...]
  * get_option_chain  -> (calls_df, puts_df) each with:
        strike, openInterest, volume, impliedVolatility, bid, ask, lastPrice

Key normalizations so downstream code (GEX, walls, IV model) needs no changes:
  - Schwab reports implied vol as a PERCENT (e.g. 18.5); we divide by 100 to
    match yfinance's decimal fraction (0.185).
  - Schwab uses -999.0 as a "not available" sentinel; we coerce it to NaN.
  - Index symbols: the app uses '^VIX' / '^VIX3M'; Schwab uses '$VIX' / '$VIX3M'.
    Translation happens at the boundary so callers keep using '^' symbols.

Credentials (read from the environment so nothing secret lives in the repo):
  SCHWAB_API_KEY      - app key from developer.schwab.com
  SCHWAB_APP_SECRET   - app secret
  SCHWAB_TOKEN_PATH   - path to the OAuth token json (default: schwab_token.json)

The token is created/refreshed by schwab_auth.py. The access token auto-refreshes
for ~7 days; after that a human must re-run schwab_auth.py (browser login).

This module is intentionally untested in CI — it cannot run without live Schwab
credentials. The yfinance provider remains the safety net (see DATA_PROVIDER).
"""

import os
import re
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from .base import DataProvider


# ── symbol translation ────────────────────────────────────────────────────
def _translate_symbol(ticker: str) -> str:
    """Map the app's symbology to Schwab's.

    '^VIX' -> '$VIX', '^VIX3M' -> '$VIX3M'. Equities pass through unchanged.
    (Futures like 'ES=F' never reach here — options_fetcher resolves them to
    ETF proxies before calling the provider.)
    """
    t = ticker.strip().upper()
    if t.startswith('^'):
        return '$' + t[1:]
    return t


# ── period parsing (yfinance-style strings -> a start datetime) ────────────
_PERIOD_RE = re.compile(r'^\s*(\d+)\s*(mo|wk|d|y)\s*$', re.IGNORECASE)


def _period_to_start(period: str) -> datetime:
    """Translate a yfinance period string ('1y','5d','2d','1mo','6mo','2y',
    'max') into a start datetime relative to now."""
    now = datetime.now()
    p = (period or '').strip().lower()
    if p in ('max', 'ytd'):
        return datetime(now.year, 1, 1) if p == 'ytd' else now - timedelta(days=365 * 20)
    m = _PERIOD_RE.match(p)
    if not m:
        # Unknown -> default to one year (matches config.DATA_PERIOD).
        return now - timedelta(days=365)
    n = int(m.group(1))
    unit = m.group(2)
    days = {'d': 1, 'wk': 7, 'mo': 30, 'y': 365}[unit]
    return now - timedelta(days=n * days)


def _clean_iv(raw) -> float:
    """Schwab volatility is a percent with -999 as 'N/A'. Return a decimal
    fraction or NaN."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return float('nan')
    if v <= 0 or v >= 999:  # -999.0 sentinel or nonsensical
        return float('nan')
    return v / 100.0


class SchwabProvider(DataProvider):
    name = "schwab"

    def __init__(self):
        # Lazy: the client is built on first use so merely importing this
        # module (or constructing the provider) never requires credentials.
        self._client = None

    # ── client construction ────────────────────────────────────────────
    def _get_client(self):
        if self._client is not None:
            return self._client

        api_key = os.environ.get('SCHWAB_API_KEY')
        app_secret = os.environ.get('SCHWAB_APP_SECRET')
        token_path = os.environ.get('SCHWAB_TOKEN_PATH', 'schwab_token.json')

        if not api_key or not app_secret:
            raise RuntimeError(
                "Schwab credentials missing. Set SCHWAB_API_KEY and "
                "SCHWAB_APP_SECRET (and optionally SCHWAB_TOKEN_PATH)."
            )
        if not os.path.exists(token_path):
            raise RuntimeError(
                f"Schwab token file not found at {token_path!r}. "
                "Run `python schwab_auth.py` to authenticate first."
            )

        # Lazy import so the yfinance path never requires schwab-py installed.
        from schwab.auth import client_from_token_file
        self._client = client_from_token_file(token_path, api_key, app_secret)
        return self._client

    # ── live quote (incl. extended hours) ──────────────────────────────
    def get_quote(self, ticker: str) -> Optional[float]:
        """Freshest trade price for `ticker`, including pre/post-market.

        Schwab's /quotes endpoint carries an extended-hours-aware ``lastPrice``,
        so a pre-open call returns the live pre-market print rather than the
        prior daily close. Best-effort: any failure returns None and the
        FallbackProvider degrades to yfinance, which degrades to the daily
        close — the pin is still recorded either way.
        """
        symbol = _translate_symbol(ticker)
        try:
            client = self._get_client()
            resp = client.get_quote(symbol)
            data = resp.json()
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        # Response shape: {'SPY': {'quote': {...}, 'regular': {...}}, ...}
        node = data.get(symbol) or (next(iter(data.values())) if data else None)
        if not isinstance(node, dict):
            return None
        quote = node.get("quote", node) if isinstance(node.get("quote", None), dict) else node

        for key in ("lastPrice", "mark", "regularMarketLastPrice", "closePrice"):
            val = quote.get(key)
            try:
                px = float(val)
            except (TypeError, ValueError):
                continue
            if px > 0:
                return px
        # Last resort: bid/ask midpoint if both are present and sane.
        try:
            bid = float(quote.get("bidPrice"))
            ask = float(quote.get("askPrice"))
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
        except (TypeError, ValueError):
            pass
        return None

    # ── price history ──────────────────────────────────────────────────
    def get_price_history(
        self,
        ticker: str,
        period: str,
        interval: str,
    ) -> pd.DataFrame:
        symbol = _translate_symbol(ticker)
        client = self._get_client()
        start = _period_to_start(period)
        end = datetime.now()

        # The app only ever requests daily candles ('1d'). Use Schwab's daily
        # convenience endpoint, which sets the correct period/frequency types.
        try:
            resp = client.get_price_history_every_day(
                symbol,
                start_datetime=start,
                end_datetime=end,
            )
            data = resp.json()
        except Exception:
            return pd.DataFrame()

        candles = data.get('candles') if isinstance(data, dict) else None
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        # Schwab fields: open, high, low, close, volume, datetime (epoch ms).
        if 'datetime' not in df.columns:
            return pd.DataFrame()
        idx = pd.to_datetime(df['datetime'], unit='ms')
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume',
        })
        keep = [c for c in ('Open', 'High', 'Low', 'Close', 'Volume') if c in df.columns]
        out = df[keep].copy()
        out.index = idx
        out.index.name = 'Date'
        return out

    # ── expirations ────────────────────────────────────────────────────
    def get_expirations(self, ticker: str) -> List[str]:
        symbol = _translate_symbol(ticker)
        try:
            client = self._get_client()
            resp = client.get_option_expiration_chain(symbol)
            data = resp.json()
        except Exception:
            return []

        exp_list = data.get('expirationList', []) if isinstance(data, dict) else []
        out = []
        for item in exp_list:
            d = item.get('expirationDate')
            if d:
                out.append(str(d)[:10])  # 'YYYY-MM-DD'
        return out

    # ── option chain ───────────────────────────────────────────────────
    def get_option_chain(
        self,
        ticker: str,
        expiry: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        symbol = _translate_symbol(ticker)
        client = self._get_client()

        exp_date = self._parse_date(expiry)
        resp = client.get_option_chain(
            symbol,
            from_date=exp_date,
            to_date=exp_date,
        )
        data = resp.json()
        if not isinstance(data, dict):
            return pd.DataFrame(), pd.DataFrame()

        calls = self._exp_map_to_df(data.get('callExpDateMap', {}), expiry)
        puts = self._exp_map_to_df(data.get('putExpDateMap', {}), expiry)
        return calls, puts

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _parse_date(expiry: str) -> date:
        return datetime.strptime(str(expiry)[:10], '%Y-%m-%d').date()

    @staticmethod
    def _exp_map_to_df(exp_map: dict, expiry: str) -> pd.DataFrame:
        """Flatten Schwab's {'YYYY-MM-DD:DTE': {'strike': [contract,...]}} into
        a flat DataFrame with the 7-column contract the app expects."""
        want_date = str(expiry)[:10]
        rows = []
        for exp_key, strikes in (exp_map or {}).items():
            # exp_key looks like '2026-06-03:0' — keep only the requested expiry.
            if str(exp_key)[:10] != want_date:
                continue
            for _strike, contracts in (strikes or {}).items():
                for c in contracts:
                    rows.append({
                        'strike': c.get('strikePrice'),
                        'openInterest': c.get('openInterest'),
                        'volume': c.get('totalVolume'),
                        'impliedVolatility': _clean_iv(c.get('volatility')),
                        'bid': c.get('bid'),
                        'ask': c.get('ask'),
                        'lastPrice': c.get('last'),
                    })
        if not rows:
            return pd.DataFrame(
                columns=['strike', 'openInterest', 'volume',
                         'impliedVolatility', 'bid', 'ask', 'lastPrice']
            )
        df = pd.DataFrame(rows)
        return df.sort_values('strike').reset_index(drop=True)
