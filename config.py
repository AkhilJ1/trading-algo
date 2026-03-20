# Watchlist of stocks to scan daily
WATCHLIST = [
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    # High-volatility / popular
    'NFLX', 'COIN', 'PLTR', 'SOFI', 'RIVN', 'NIO', 'MSTR', 'RBLX',
    # Financials
    'JPM', 'BAC', 'GS', 'MS',
    # ETFs
    'SPY', 'QQQ', 'ARKK',
    # Semis
    'INTC', 'MU', 'QCOM', 'AVGO', 'TSM',
    # Other
    'DIS', 'BA', 'F', 'GM',
]

# Moving average crossover (golden cross / death cross)
MA_SHORT = 50
MA_LONG = 200

# RSI settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30       # Buy threshold: RSI < 30

# Bollinger Band settings
BB_PERIOD = 20
BB_STD = 2.0
BB_WICK_LOOKBACK = 5    # Number of recent candles to check for lower-BB wick touches

# Data fetch settings
DATA_PERIOD = '1y'      # 1 year of daily history
DATA_INTERVAL = '1d'    # Daily candles

# Cache directory (relative to project root)
CACHE_DIR = 'data/cache'

# ── Fractal & Options Analysis ────────────────────────────────────────────
FRACTAL_PERIOD = 5          # Bill Williams fractal: N bars each side
FRACTAL_DIM_WINDOW = 30     # Window for fractal dimension (Hurst) calculation

# Futures → ETF proxy for options data (futures have limited options in yfinance)
FUTURES_PROXY = {
    'ES=F': 'SPY',
    'NQ=F': 'QQQ',
    'YM=F': 'DIA',
}

# Bias voting weights (sum to 1.0) — controls directional bias only, NOT floor/ceiling
SIGNAL_WEIGHTS = {
    'options_walls':    0.25,
    'gex_levels':       0.20,
    'iv_range':         0.20,
    'fractals':         0.15,
    'put_call_ratio':   0.10,
    'iv_skew':          0.05,
    'max_pain':         0.05,
}

# ── Evidence-Based Range Engine ────────────────────────────────────────
# Variance Risk Premium (IV systematically overstates realized vol)
PARKINSON_WINDOW = 20           # Trading days for Parkinson vol estimator
VRP_LOOKBACK_DAYS = 63          # ~3 months for IV/RV ratio
VRP_MIN_RATIO = 0.70            # Floor: never shrink IV below 70%
VRP_MAX_RATIO = 1.00            # Ceiling: never inflate IV

# Regime-based range scaling
REGIME_SCALE = {
    'trending':      1.20,      # Wider range in trending markets
    'transitional':  1.00,      # Baseline
    'choppy':        0.85,      # Tighter range in mean-reverting markets
}
VIX_CONTANGO_SHRINK = 0.95      # Contango (normal): slight range tightening
VIX_BACKWARDATION_EXPAND = 1.10 # Backwardation (fear): range widening

# Dealer hedging boundary blend (how much weight to give GEX/walls vs IV model)
DEALER_BOUND_BLEND = 0.30       # 30% dealer levels, 70% IV-based
WALL_CLUSTER_WIDTH = 0.02       # 2% of spot for OI strike clustering

# Multi-sigma confidence levels for iron condor strike selection
CONFIDENCE_SIGMAS = {
    '1sigma':   1.0,            # ~68.3% probability
    '1_5sigma': 1.5,            # ~86.6% probability
    '2sigma':   2.0,            # ~95.4% probability
}

# ── Google Sheets Config ─────────────────────────────────────────────────
GSHEET_SPREADSHEET_NAME = "TradingAlgoPredictions"
GSHEET_PREDICTIONS_SHEET = "Predictions"
GSHEET_WEIGHTS_SHEET = "Weights"

# ── Auto-Retune Safety Rails ─────────────────────────────────────────────
RETUNE_MIN_DAYS = 60            # Minimum prediction history before retuning
RETUNE_ROLLING_WINDOW = 90     # Days of data used for learning
RETUNE_MAX_WEIGHT_CHANGE = 0.15 # Max 15% change per weight per cycle
RETUNE_MIN_ACCURACY = 0.65     # Guard rail: revert if accuracy drops below 65%
RETUNE_HOLDOUT_FRACTION = 0.20 # Last 20% of data reserved for validation
RETUNE_MIN_WEIGHT = 0.02       # No signal weight below 2%
RETUNE_MAX_WEIGHT = 0.40       # No signal weight above 40%
