# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (Python 3.9+)
pip3 install -r requirements.txt

# Run the daily scanner (full watchlist)
python scanner.py

# Run the scanner on specific tickers
python scanner.py AAPL TSLA NVDA

# Run backtests (defaults to SPY, AAPL, NVDA)
python backtest.py

# Backtest specific tickers
python backtest.py MSFT COIN AMD
```

## Architecture

```
trading-algo/
├── config.py               # Watchlist, strategy parameters (edit this to customize)
├── data_fetcher.py         # yfinance wrapper; caches per-day to data/cache/
├── strategies/
│   ├── ma_crossover.py     # SMA(50/200) golden/death cross signals
│   └── rsi_bollinger.py    # RSI(14) + lower Bollinger Band wick detection
├── scanner.py              # Daily trade ideas script (main entry point)
├── backtest.py             # MA crossover backtester with P&L, win rate, drawdown
└── data/cache/             # Auto-generated daily CSV cache (gitignore this)
```

## Strategy Logic

**Buy signal** (`scanner.py`) requires both conditions on 1-year daily chart:
1. RSI(14) < 30 — stock is oversold
2. At least one of the last 5 candles has a wick (low) at or below the lower Bollinger Band (20-period, 2 std dev)

Signal strength (0–100) blends how far RSI is below 30 with how many recent wicks touched the lower BB.

**MA crossover** (`backtest.py`) uses SMA(50) vs SMA(200):
- Golden cross = buy; death cross = sell
- Requires ≥200 bars of data — tickers with less are skipped

**Data caching**: Each fetch writes `data/cache/{TICKER}_{period}_{interval}_{date}.csv`. Old dates are ignored automatically. Pass `use_cache=False` to `fetch_stock_data()` to force a fresh pull.

## Customization

- Edit `WATCHLIST` in `config.py` to change tickers.
- Adjust `RSI_OVERSOLD`, `BB_WICK_LOOKBACK`, `MA_SHORT`, `MA_LONG` in `config.py` to tune strategy parameters.
