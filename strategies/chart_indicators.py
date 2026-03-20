"""
Institutional-grade chart indicators for the Stock Chart and Scanner pages.
All functions are pure computation — no Streamlit dependencies.
"""

import numpy as np
import pandas as pd

from config import (
    VWAP_WINDOW,
    ZSCORE_WINDOW,
    KELTNER_EMA,
    KELTNER_ATR,
    KELTNER_MULT,
    ATR_STOP_PERIOD,
    ATR_STOP_MULT,
    SCANNER_WEIGHTS,
)


# ── ATR helper (shared) ──────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── VWAP ──────────────────────────────────────────────────────────────────

def compute_vwap(df: pd.DataFrame, window: int = VWAP_WINDOW) -> pd.Series:
    """Rolling VWAP over `window` bars."""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_vol = tp * df['Volume']
    return tp_vol.rolling(window).sum() / df['Volume'].rolling(window).sum()


def compute_anchored_vwap(df: pd.DataFrame, anchor_date) -> pd.Series:
    """VWAP anchored from a specific date forward."""
    anchor_date = pd.Timestamp(anchor_date)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_vol = tp * df['Volume']

    result = pd.Series(np.nan, index=df.index)
    mask = df.index >= anchor_date
    cum_tp_vol = tp_vol[mask].cumsum()
    cum_vol = df['Volume'][mask].cumsum()
    result[mask] = cum_tp_vol / cum_vol
    return result


def find_anchor_events(df: pd.DataFrame) -> list:
    """
    Auto-detect significant anchor dates:
    - 52-week high/low
    - Gap days (|Open - prev Close| > 2 * ATR)
    """
    if len(df) < 20:
        return []

    events = []
    atr = _compute_atr(df, 14)

    # 52-week high and low (use last 252 bars or all available)
    lookback = min(252, len(df))
    recent = df.iloc[-lookback:]

    high_idx = recent['High'].idxmax()
    low_idx = recent['Low'].idxmin()
    events.append({'date': high_idx, 'event': '52W High', 'price': float(recent.loc[high_idx, 'High'])})
    events.append({'date': low_idx, 'event': '52W Low', 'price': float(recent.loc[low_idx, 'Low'])})

    # Gap days
    prev_close = df['Close'].shift(1)
    gaps = (df['Open'] - prev_close).abs()
    for i in range(1, len(df)):
        idx = df.index[i]
        atr_val = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0
        if atr_val > 0 and gaps.iloc[i] > 2 * atr_val:
            events.append({'date': idx, 'event': 'Gap', 'price': float(df['Open'].iloc[i])})

    # Deduplicate and limit to most recent 5
    seen = set()
    unique = []
    for e in events:
        key = str(e['date'])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return sorted(unique, key=lambda x: x['date'], reverse=True)[:5]


# ── Volume Profile ────────────────────────────────────────────────────────

def compute_volume_profile(df: pd.DataFrame, n_bins: int = 50) -> dict:
    """
    Price-binned volume histogram.
    Returns POC (Point of Control), VAH (Value Area High), VAL (Value Area Low).
    """
    if df.empty or len(df) < 5:
        return {'bin_centers': [], 'volumes': [], 'poc': 0, 'vah': 0, 'val': 0}

    price_min = float(df['Low'].min())
    price_max = float(df['High'].max())
    if price_max <= price_min:
        return {'bin_centers': [], 'volumes': [], 'poc': price_min, 'vah': price_max, 'val': price_min}

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign each bar's volume to the bin of its close
    bin_idx = np.digitize(df['Close'].values, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    vol_hist = np.zeros(n_bins)
    np.add.at(vol_hist, bin_idx, df['Volume'].values.astype(float))

    # POC: price with highest volume
    poc_idx = int(np.argmax(vol_hist))
    poc = float(bin_centers[poc_idx])

    # Value Area: expand from POC until 70% of volume captured
    total_vol = vol_hist.sum()
    if total_vol == 0:
        return {'bin_centers': bin_centers.tolist(), 'volumes': vol_hist.tolist(),
                'poc': poc, 'vah': float(bin_edges[-1]), 'val': float(bin_edges[0])}

    target = total_vol * 0.70
    va_vol = vol_hist[poc_idx]
    lo, hi = poc_idx, poc_idx

    while va_vol < target and (lo > 0 or hi < n_bins - 1):
        expand_lo = vol_hist[lo - 1] if lo > 0 else 0
        expand_hi = vol_hist[hi + 1] if hi < n_bins - 1 else 0
        if expand_lo >= expand_hi and lo > 0:
            lo -= 1
            va_vol += vol_hist[lo]
        elif hi < n_bins - 1:
            hi += 1
            va_vol += vol_hist[hi]
        else:
            lo -= 1
            va_vol += vol_hist[lo]

    return {
        'bin_centers': bin_centers.tolist(),
        'volumes': vol_hist.tolist(),
        'poc': poc,
        'vah': float(bin_edges[hi + 1]),
        'val': float(bin_edges[lo]),
    }


# ── Z-Score ───────────────────────────────────────────────────────────────

def compute_zscore(df: pd.DataFrame, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Z-score: (Close - SMA) / rolling_std."""
    ma = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    return (df['Close'] - ma) / std


# ── Keltner Channels ──────────────────────────────────────────────────────

def compute_keltner_channels(
    df: pd.DataFrame,
    ema_period: int = KELTNER_EMA,
    atr_period: int = KELTNER_ATR,
    multiplier: float = KELTNER_MULT,
) -> tuple:
    """Returns (upper, mid, lower) Keltner Channel Series."""
    mid = df['Close'].ewm(span=ema_period, adjust=False).mean()
    atr = _compute_atr(df, atr_period)
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr
    return upper, mid, lower


# ── ATR Trailing Stop ─────────────────────────────────────────────────────

def compute_atr_stops(
    df: pd.DataFrame,
    period: int = ATR_STOP_PERIOD,
    multiplier: float = ATR_STOP_MULT,
) -> pd.Series:
    """Chandelier-style ATR trailing stop: Close - ATR * multiplier."""
    atr = _compute_atr(df, period)
    return df['Close'] - atr * multiplier


# ── Confluence Signals ────────────────────────────────────────────────────

def compute_confluence_signals(df: pd.DataFrame, fractal_levels: dict = None) -> pd.DataFrame:
    """
    Multi-signal confluence scoring per bar.
    Returns DataFrame with signal_score (-5 to +5) and component booleans.
    """
    result = pd.DataFrame(index=df.index)
    result['signal_score'] = 0

    # 1. RSI
    from strategies.rsi_bollinger import calculate_rsi
    rsi = calculate_rsi(df)
    result['rsi_buy'] = rsi < 30
    result['rsi_sell'] = rsi > 70
    result['signal_score'] += result['rsi_buy'].astype(int) - result['rsi_sell'].astype(int)

    # 2. Z-Score
    zscore = compute_zscore(df)
    result['zscore_buy'] = zscore <= -2.0
    result['zscore_sell'] = zscore >= 2.0
    result['signal_score'] += result['zscore_buy'].astype(int) - result['zscore_sell'].astype(int)

    # 3. Keltner Channels
    k_upper, k_mid, k_lower = compute_keltner_channels(df)
    result['keltner_buy'] = df['Close'] <= k_lower
    result['keltner_sell'] = df['Close'] >= k_upper
    result['signal_score'] += result['keltner_buy'].astype(int) - result['keltner_sell'].astype(int)

    # 4. Fractal S/R proximity (within 1% of level)
    result['fractal_buy'] = False
    result['fractal_sell'] = False
    if fractal_levels:
        for _, level in fractal_levels.get('support_levels', []):
            near_support = (df['Close'] - level).abs() / df['Close'] < 0.01
            result['fractal_buy'] = result['fractal_buy'] | near_support
        for _, level in fractal_levels.get('resistance_levels', []):
            near_resist = (df['Close'] - level).abs() / df['Close'] < 0.01
            result['fractal_sell'] = result['fractal_sell'] | near_resist
    result['signal_score'] += result['fractal_buy'].astype(int) - result['fractal_sell'].astype(int)

    # 5. Volume surge (>1.5x 20-day avg) in direction of bar
    vol_avg = df['Volume'].rolling(20).mean()
    vol_surge = df['Volume'] > 1.5 * vol_avg
    bar_dir = np.sign(df['Close'] - df['Open'])  # +1 up, -1 down
    result['vol_surge'] = vol_surge
    result['signal_score'] += (vol_surge.astype(int) * bar_dir).fillna(0).astype(int)

    # Label
    labels = []
    for s in result['signal_score']:
        if s >= 2:
            labels.append('strong_buy')
        elif s == 1:
            labels.append('buy')
        elif s == 0:
            labels.append('neutral')
        elif s == -1:
            labels.append('sell')
        else:
            labels.append('strong_sell')
    result['signal_label'] = labels

    return result


# ── Multi-Factor Scanner Score ────────────────────────────────────────────

def compute_multi_factor_score(df: pd.DataFrame) -> dict:
    """
    Compute a multi-factor composite score (0-100) for scanner ranking.
    Higher = stronger setup.
    """
    if len(df) < 60:
        return {'composite': 0, 'rsi_score': 0, 'zscore_score': 0,
                'volume_score': 0, 'regime_score': 0, 'atr_score': 0,
                'zscore': 0, 'volume_surge': False, 'regime': 'unknown', 'atr_state': 'normal'}

    from strategies.rsi_bollinger import calculate_rsi
    from strategies.fractal_indicators import calculate_fractal_dimension, classify_regime

    close = df['Close']
    rsi = calculate_rsi(df)
    zscore = compute_zscore(df)
    atr = _compute_atr(df, 14)

    current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50
    current_z = float(zscore.iloc[-1]) if not zscore.dropna().empty else 0
    current_vol = float(df['Volume'].iloc[-1])
    avg_vol = float(df['Volume'].rolling(20).mean().iloc[-1]) if len(df) >= 20 else current_vol

    # Fractal dimension / regime
    fd_series = calculate_fractal_dimension(df)
    current_fd = float(fd_series.dropna().iloc[-1]) if not fd_series.dropna().empty else 1.5
    regime = classify_regime(current_fd)

    # ATR contraction (current ATR / 60-day ATR average)
    atr_60_avg = float(atr.rolling(60).mean().iloc[-1]) if len(atr.dropna()) >= 60 else float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 1
    current_atr = float(atr.iloc[-1]) if not atr.dropna().empty else 1
    atr_ratio = current_atr / atr_60_avg if atr_60_avg > 0 else 1.0

    # Score components (each scaled to their max)
    # RSI (0-25): oversold = high score
    rsi_score = max(0, min(25, (30 - current_rsi) * 25 / 30)) if current_rsi < 30 else 0

    # Z-score (0-25): more negative = higher score
    zscore_score = max(0, min(25, abs(current_z) * 12.5)) if current_z < -1.0 else 0

    # Volume surge (0-15)
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
    volume_score = max(0, min(15, (vol_ratio - 1.0) * 15)) if vol_ratio > 1.0 else 0
    volume_surge = vol_ratio > 1.5

    # Regime (0-20): trending = best for entries
    regime_scores = {'trending': 20, 'transitional': 10, 'choppy': 3}
    regime_score = regime_scores.get(regime, 5)

    # ATR contraction (0-15): lower ratio = more coiled = higher score
    atr_score = max(0, min(15, (1.0 - atr_ratio) * 30)) if atr_ratio < 1.0 else 0
    atr_state = 'contracted' if atr_ratio < 0.7 else 'expanded' if atr_ratio > 1.3 else 'normal'

    # Weighted composite
    w = SCANNER_WEIGHTS
    raw = (rsi_score / 25 * w['rsi'] +
           zscore_score / 25 * w['zscore'] +
           volume_score / 15 * w['volume'] +
           regime_score / 20 * w['regime'] +
           atr_score / 15 * w['atr'])
    composite = round(raw * 100, 1)

    return {
        'composite': composite,
        'rsi_score': round(rsi_score, 1),
        'zscore_score': round(zscore_score, 1),
        'volume_score': round(volume_score, 1),
        'regime_score': round(regime_score, 1),
        'atr_score': round(atr_score, 1),
        'zscore': round(current_z, 2),
        'volume_surge': volume_surge,
        'regime': regime,
        'atr_state': atr_state,
    }


# ── Sell Signal Detection ────────────────────────────────────────────────

def compute_sell_signals(df: pd.DataFrame) -> dict:
    """
    Detect overbought / sell conditions — mirror of buy signal detection.
    Returns dict with sell_signal (bool), strength (0-100), and component details.
    """
    if len(df) < 60:
        return {'sell_signal': False, 'sell_strength': 0, 'components': {},
                'rsi': 50, 'zscore': 0}

    from strategies.rsi_bollinger import calculate_rsi

    close = df['Close']
    rsi = calculate_rsi(df)
    zscore = compute_zscore(df)
    k_upper, _, _ = compute_keltner_channels(df)

    # Bollinger upper
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std

    current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50
    current_z = float(zscore.iloc[-1]) if not zscore.dropna().empty else 0
    current_close = float(close.iloc[-1])
    current_k_upper = float(k_upper.iloc[-1]) if not k_upper.dropna().empty else float('inf')
    current_bb_upper = float(bb_upper.iloc[-1]) if not bb_upper.dropna().empty else float('inf')

    components = {
        'rsi_overbought': current_rsi > 70,
        'zscore_extended': current_z > 2.0,
        'above_keltner': current_close >= current_k_upper,
        'above_bb_upper': current_close >= current_bb_upper,
    }

    # Count how many sell conditions are active
    active = sum(components.values())
    sell_signal = active >= 2  # Need at least 2 conditions for a sell signal

    # Strength: weighted scoring (0-100)
    strength = 0.0
    if components['rsi_overbought']:
        strength += min(30, (current_rsi - 70) * 30 / 30)  # 0-30 pts
    if components['zscore_extended']:
        strength += min(30, (current_z - 2.0) * 15)  # 0-30 pts
    if components['above_keltner']:
        strength += 20
    if components['above_bb_upper']:
        strength += 20
    strength = min(100, strength)

    return {
        'sell_signal': sell_signal,
        'sell_strength': round(strength, 1),
        'components': components,
        'active_count': active,
        'rsi': round(current_rsi, 1),
        'zscore': round(current_z, 2),
        'bb_upper': round(current_bb_upper, 2),
        'keltner_upper': round(current_k_upper, 2),
    }


# ── Opportunity Scanner ──────────────────────────────────────────────────

def compute_opportunity_scan(df: pd.DataFrame) -> list:
    """
    Scan for actionable setups regardless of direction.
    Returns list of opportunity dicts with tier, direction, confidence, and reason.
    """
    if len(df) < 60:
        return []

    from strategies.rsi_bollinger import calculate_rsi
    from strategies.fractal_indicators import calculate_fractal_dimension, classify_regime

    close = df['Close']
    rsi = calculate_rsi(df)
    zscore = compute_zscore(df)
    atr = _compute_atr(df, 14)

    current_rsi = float(rsi.iloc[-1]) if not rsi.empty else 50
    current_z = float(zscore.iloc[-1]) if not zscore.dropna().empty else 0
    current_close = float(close.iloc[-1])
    current_vol = float(df['Volume'].iloc[-1])
    avg_vol = float(df['Volume'].rolling(20).mean().iloc[-1]) if len(df) >= 20 else current_vol
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

    # Regime
    fd_series = calculate_fractal_dimension(df)
    current_fd = float(fd_series.dropna().iloc[-1]) if not fd_series.dropna().empty else 1.5
    regime = classify_regime(current_fd)

    # ATR ratio
    atr_60_avg = float(atr.rolling(60).mean().iloc[-1]) if len(atr.dropna()) >= 60 else float(atr.dropna().iloc[-1]) if not atr.dropna().empty else 1
    current_atr = float(atr.iloc[-1]) if not atr.dropna().empty else 1
    atr_ratio = current_atr / atr_60_avg if atr_60_avg > 0 else 1.0

    # Bollinger bandwidth
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_bandwidth = (bb_std * 4) / bb_mid  # Total band width as % of price
    current_bw = float(bb_bandwidth.iloc[-1]) if not bb_bandwidth.dropna().empty else 0
    bw_20_min = float(bb_bandwidth.rolling(20).min().iloc[-1]) if len(bb_bandwidth.dropna()) >= 20 else current_bw

    # SMA 50/200
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else current_close
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else current_close

    opportunities = []

    # ── Tier 1: Proven (Statistical) ──────────────────────────────────

    # Z-Score approaching extreme (mean reversion setup)
    if current_z <= -1.5:
        conf = min(90, 50 + abs(current_z) * 20)
        opportunities.append({
            'setup': 'Z-Score Mean Reversion',
            'direction': 'long',
            'tier': 1,
            'confidence': round(conf),
            'reason': f'Z-Score at {current_z:.2f} — statistically mean-reverting (< -1.5σ)',
        })
    elif current_z >= 1.5:
        conf = min(90, 50 + abs(current_z) * 20)
        opportunities.append({
            'setup': 'Z-Score Mean Reversion',
            'direction': 'short',
            'tier': 1,
            'confidence': round(conf),
            'reason': f'Z-Score at {current_z:+.2f} — statistically extended (> +1.5σ)',
        })

    # RSI approaching extreme
    if 30 <= current_rsi <= 35:
        opportunities.append({
            'setup': 'RSI Approaching Oversold',
            'direction': 'long',
            'tier': 1,
            'confidence': round(60 + (35 - current_rsi) * 6),
            'reason': f'RSI at {current_rsi:.1f} — approaching oversold threshold (< 30)',
        })
    elif 65 <= current_rsi <= 70:
        opportunities.append({
            'setup': 'RSI Approaching Overbought',
            'direction': 'short',
            'tier': 1,
            'confidence': round(60 + (current_rsi - 65) * 6),
            'reason': f'RSI at {current_rsi:.1f} — approaching overbought threshold (> 70)',
        })

    # Bollinger Band squeeze (BB bandwidth at 20-bar low)
    if current_bw > 0 and current_bw <= bw_20_min * 1.05:
        opportunities.append({
            'setup': 'Bollinger Squeeze',
            'direction': 'long' if current_z < 0 else 'short',
            'tier': 1,
            'confidence': 65,
            'reason': 'BB bandwidth at 20-bar low — volatility compression precedes expansion',
        })

    # ── Tier 2: Validated (Backtested) ────────────────────────────────

    # Momentum setup: trending regime + above key MAs
    if regime == 'trending' and current_close > sma50 > sma200:
        opportunities.append({
            'setup': 'Trend Continuation',
            'direction': 'long',
            'tier': 2,
            'confidence': 70,
            'reason': f'Trending regime (FD={current_fd:.2f}) with price above SMA50/200 — momentum favored',
        })
    elif regime == 'trending' and current_close < sma50 < sma200:
        opportunities.append({
            'setup': 'Trend Continuation',
            'direction': 'short',
            'tier': 2,
            'confidence': 70,
            'reason': f'Trending regime (FD={current_fd:.2f}) with price below SMA50/200 — downtrend continuation',
        })

    # Volume-confirmed breakout/breakdown
    if vol_ratio > 1.5:
        pct_change = (current_close - float(close.iloc[-2])) / float(close.iloc[-2]) * 100
        if pct_change > 0.5:
            opportunities.append({
                'setup': 'Volume Breakout',
                'direction': 'long',
                'tier': 2,
                'confidence': round(55 + min(20, vol_ratio * 5)),
                'reason': f'Price up {pct_change:+.1f}% on {vol_ratio:.1f}x avg volume — volume-confirmed move',
            })
        elif pct_change < -0.5:
            opportunities.append({
                'setup': 'Volume Breakdown',
                'direction': 'short',
                'tier': 2,
                'confidence': round(55 + min(20, vol_ratio * 5)),
                'reason': f'Price down {pct_change:+.1f}% on {vol_ratio:.1f}x avg volume — volume-confirmed move',
            })

    # ATR contraction (coiled spring)
    if atr_ratio < 0.7:
        opportunities.append({
            'setup': 'ATR Contraction (Coiled Spring)',
            'direction': 'long' if current_z < 0 else 'short',
            'tier': 2,
            'confidence': 60,
            'reason': f'ATR {atr_ratio:.0%} of 60-day avg — volatility compressed, breakout likely',
        })

    # Mean reversion in choppy regime
    if regime == 'choppy' and abs(current_z) > 1.0:
        direction = 'long' if current_z < 0 else 'short'
        opportunities.append({
            'setup': 'Regime Mean Reversion',
            'direction': direction,
            'tier': 2,
            'confidence': 65,
            'reason': f'Choppy regime (FD={current_fd:.2f}) + Z-Score {current_z:+.2f} — mean reversion favored',
        })

    # ── Tier 3: Speculative ───────────────────────────────────────────

    # SMA crossover proximity (within 1%)
    if sma50 > 0 and sma200 > 0:
        sma_diff = (sma50 - sma200) / sma200
        if abs(sma_diff) < 0.01:
            cross_type = 'Golden Cross' if sma50 > sma200 else 'Death Cross'
            opportunities.append({
                'setup': f'{cross_type} Imminent',
                'direction': 'long' if sma50 > sma200 else 'short',
                'tier': 3,
                'confidence': 45,
                'reason': f'SMA50/200 within 1% — {cross_type.lower()} forming',
            })

    # Sort by tier then confidence descending
    opportunities.sort(key=lambda x: (x['tier'], -x['confidence']))
    return opportunities


# ── Strategy Consensus ───────────────────────────────────────────────────

def compute_strategy_consensus(df: pd.DataFrame) -> dict:
    """
    Run all available strategies and return their latest signal.
    Returns dict of strategy_name → signal (+1, -1, 0) and summary.
    """
    signals = {}

    # 1. MA Crossover
    try:
        from strategies.ma_crossover import current_signal as ma_current_signal
        ma = ma_current_signal(df)
        signals['MA Cross'] = 1 if ma['trend'] == 'bullish' else -1
    except Exception:
        signals['MA Cross'] = 0

    # 2. RSI + BB
    try:
        from strategies.rsi_bollinger import get_buy_signal
        rsi_bb = get_buy_signal(df)
        if rsi_bb['buy_signal']:
            signals['RSI+BB'] = 1
        elif rsi_bb['rsi'] > 70:
            signals['RSI+BB'] = -1
        else:
            signals['RSI+BB'] = 0
    except Exception:
        signals['RSI+BB'] = 0

    # 3. MACD + RSI
    try:
        from strategies.macd_rsi import generate_signals
        sig_df = generate_signals(df)
        signals['MACD+RSI'] = int(sig_df['strategy_signal'].iloc[-1])
    except Exception:
        signals['MACD+RSI'] = 0

    # 4. BB Squeeze
    try:
        from strategies.bb_squeeze import generate_signals
        sig_df = generate_signals(df)
        signals['BB Squeeze'] = int(sig_df['strategy_signal'].iloc[-1])
    except Exception:
        signals['BB Squeeze'] = 0

    # 5. TSMOM
    try:
        from strategies.tsmom import generate_signals
        sig_df = generate_signals(df)
        signals['TSMOM'] = int(sig_df['strategy_signal'].iloc[-1])
    except Exception:
        signals['TSMOM'] = 0

    # 6. Turtle
    try:
        from strategies.turtle import generate_signals
        sig_df = generate_signals(df)
        signals['Turtle'] = int(sig_df['strategy_signal'].iloc[-1])
    except Exception:
        signals['Turtle'] = 0

    # Summary
    bullish = sum(1 for v in signals.values() if v > 0)
    bearish = sum(1 for v in signals.values() if v < 0)
    total = len(signals)

    if bullish >= 3:
        consensus = 'bullish'
    elif bearish >= 3:
        consensus = 'bearish'
    else:
        consensus = 'mixed'

    return {
        'signals': signals,
        'bullish_count': bullish,
        'bearish_count': bearish,
        'neutral_count': total - bullish - bearish,
        'consensus': consensus,
        'total': total,
    }
