"""
Fractal & Options Analysis Engine
-----------------------------------
Combines fractal market structure with options flow analytics to produce
institutional-grade support/resistance levels, directional bias, and
probabilistic range estimates.

Signals:
  1. Options walls (OI-based support/resistance)        25%
  2. GEX (gamma exposure pivot levels)                   20%
  3. IV expected range (probabilistic daily/weekly)       20%
  4. Fractal structure (Bill Williams swing levels)       15%
  5. Put/Call ratio (sentiment)                           10%
  6. IV skew (institutional hedging bias)                  5%
  7. Max pain (expiration magnet)                          5%
"""

import math
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import SIGNAL_WEIGHTS
from options_fetcher import fetch_options_chain, fetch_expiration_dates
from strategies.fractal_indicators import (
    add_williams_fractals,
    get_recent_fractal_levels,
    calculate_fractal_dimension,
    classify_regime,
)
from data_fetcher import fetch_stock_data


# ── Black-Scholes helpers ─────────────────────────────────────────────────

def _bs_d1(S, K, T, r, sigma):
    """Black-Scholes d1."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma for a single option."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


# ── Options Analytics ─────────────────────────────────────────────────────

def compute_gex_profile(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    expiry_str: str,
    r: float = 0.045,
) -> pd.DataFrame:
    """
    Compute Gamma Exposure (GEX) per strike.

    Dealers are short options, so:
      Call GEX = +gamma * OI * 100 * spot  (positive — dealers buy on rise)
      Put GEX  = -gamma * OI * 100 * spot  (negative — dealers sell on drop)

    Returns DataFrame: strike, call_gex, put_gex, net_gex
    """
    today = date.today()
    try:
        expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        expiry_date = today
    T = max((expiry_date - today).days, 1) / 365.0

    # Filter to strikes within +/- 15% of spot
    lo, hi = spot * 0.85, spot * 1.15
    c = calls[(calls['strike'] >= lo) & (calls['strike'] <= hi)].copy()
    p = puts[(puts['strike'] >= lo) & (puts['strike'] <= hi)].copy()

    rows = []
    all_strikes = sorted(set(c['strike'].tolist() + p['strike'].tolist()))

    for strike in all_strikes:
        call_row = c[c['strike'] == strike]
        put_row = p[p['strike'] == strike]

        call_gex = 0.0
        if not call_row.empty:
            iv = float(call_row['impliedVolatility'].iloc[0])
            oi = float(call_row['openInterest'].iloc[0])
            if iv > 0.05 and oi > 0:  # require at least 5% IV
                g = _bs_gamma(spot, strike, T, r, iv)
                call_gex = g * oi * 100 * spot

        put_gex = 0.0
        if not put_row.empty:
            iv = float(put_row['impliedVolatility'].iloc[0])
            oi = float(put_row['openInterest'].iloc[0])
            if iv > 0.05 and oi > 0:  # require at least 5% IV
                g = _bs_gamma(spot, strike, T, r, iv)
                put_gex = -g * oi * 100 * spot

        rows.append({
            'strike': strike,
            'call_gex': round(call_gex, 2),
            'put_gex': round(put_gex, 2),
            'net_gex': round(call_gex + put_gex, 2),
        })

    return pd.DataFrame(rows)


def compute_options_walls(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    top_n: int = 10,
) -> dict:
    """
    Identify high-activity strikes as support/resistance walls.

    Uses openInterest when available, falls back to volume.
    Call walls (resistance): highest activity strikes above spot.
    Put walls (support): highest activity strikes below spot.
    """
    col = _pick_activity_col(pd.concat([calls, puts], ignore_index=True))

    # Call walls — above spot
    above = calls[calls['strike'] >= spot]
    if col in above.columns and above[col].sum() > 0:
        above = above.nlargest(top_n, col)
        call_walls = [(float(r['strike']), int(r[col]))
                      for _, r in above.iterrows() if r[col] > 0]
    else:
        call_walls = []

    # Put walls — below spot
    below = puts[puts['strike'] <= spot]
    if col in below.columns and below[col].sum() > 0:
        below = below.nlargest(top_n, col)
        put_walls = [(float(r['strike']), int(r[col]))
                     for _, r in below.iterrows() if r[col] > 0]
    else:
        put_walls = []

    strongest_call = call_walls[0][0] if call_walls else spot * 1.05
    strongest_put = put_walls[0][0] if put_walls else spot * 0.95

    return {
        'call_walls': sorted(call_walls, key=lambda x: x[0]),
        'put_walls': sorted(put_walls, key=lambda x: x[0]),
        'strongest_call_wall': strongest_call,
        'strongest_put_wall': strongest_put,
        'activity_source': col,
    }


def _pick_activity_col(df: pd.DataFrame) -> str:
    """Pick 'openInterest' if it has data, otherwise fall back to 'volume'."""
    if 'openInterest' in df.columns and df['openInterest'].sum() > 0:
        return 'openInterest'
    if 'volume' in df.columns and df['volume'].sum() > 0:
        return 'volume'
    return 'openInterest'


def compute_max_pain(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float = 0.0,
) -> float:
    """
    Max pain: the strike where total options expire with minimum value.

    Uses openInterest when available, falls back to volume.
    Returns spot price if no activity data is available.
    """
    col = _pick_activity_col(pd.concat([calls, puts], ignore_index=True))

    calls_active = calls[calls[col] > 0] if col in calls.columns else calls
    puts_active = puts[puts[col] > 0] if col in puts.columns else puts

    total_activity = (calls_active[col].sum() if not calls_active.empty else 0) + \
                     (puts_active[col].sum() if not puts_active.empty else 0)

    if total_activity == 0:
        return spot if spot > 0 else 0.0

    # Only consider strikes near spot (±15%) for efficiency and relevance
    lo, hi = spot * 0.85, spot * 1.15
    calls_near = calls_active[(calls_active['strike'] >= lo) & (calls_active['strike'] <= hi)]
    puts_near = puts_active[(puts_active['strike'] >= lo) & (puts_active['strike'] <= hi)]

    all_strikes = sorted(set(calls_near['strike'].tolist() + puts_near['strike'].tolist()))
    if not all_strikes:
        return spot if spot > 0 else 0.0

    min_pain = float('inf')
    max_pain_strike = all_strikes[len(all_strikes) // 2]

    for K in all_strikes:
        call_pain = sum(
            max(0, K - row['strike']) * row[col]
            for _, row in calls_near.iterrows()
        )
        put_pain = sum(
            max(0, row['strike'] - K) * row[col]
            for _, row in puts_near.iterrows()
        )
        total = call_pain + put_pain
        if total < min_pain:
            min_pain = total
            max_pain_strike = K

    return float(max_pain_strike)


def compute_iv_expected_move(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    expiry_str: str,
    vix_fallback: float = None,
) -> dict:
    """
    IV-based expected price range using ATM straddle implied volatility.

    1-sigma range (~68%): spot +/- (spot * IV * sqrt(T))
    2-sigma range (~95%): spot +/- 2 * (spot * IV * sqrt(T))

    Falls back to VIX/100 as IV proxy when option IVs are unavailable.
    """
    today = date.today()
    try:
        expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        expiry_date = today
    dte = max((expiry_date - today).days, 1)
    T = dte / 365.0

    # Find ATM options (closest strike to spot)
    atm_call = calls.iloc[(calls['strike'] - spot).abs().argsort()[:3]]
    atm_put = puts.iloc[(puts['strike'] - spot).abs().argsort()[:3]]

    # Average ATM IV (filter out junk values — require at least 5% annualized IV)
    ivs = []
    for df_atm in [atm_call, atm_put]:
        for _, row in df_atm.iterrows():
            iv = row.get('impliedVolatility', 0)
            if iv and iv > 0.05:  # require at least 5% IV
                ivs.append(iv)

    iv_source = 'options'
    if ivs:
        atm_iv = np.mean(ivs)
    elif vix_fallback and vix_fallback > 0:
        # VIX represents annualized 1-sigma expected move of S&P 500 in %
        atm_iv = vix_fallback / 100.0
        iv_source = 'VIX'
    else:
        atm_iv = 0.20
        iv_source = 'default'

    # Expected move to expiry
    move_1s = spot * atm_iv * math.sqrt(T)
    move_2s = 2 * move_1s

    # Daily expected move
    daily_move = spot * atm_iv * math.sqrt(1 / 365.0)

    return {
        'iv_used': round(atm_iv, 4),
        'iv_source': iv_source,
        'days_to_expiry': dte,
        'expected_move_1sigma': round(move_1s, 2),
        'expected_move_2sigma': round(move_2s, 2),
        'range_low_1sigma': round(spot - move_1s, 2),
        'range_high_1sigma': round(spot + move_1s, 2),
        'range_low_2sigma': round(spot - move_2s, 2),
        'range_high_2sigma': round(spot + move_2s, 2),
        'daily_expected_move': round(daily_move, 2),
        'daily_range_low': round(spot - daily_move, 2),
        'daily_range_high': round(spot + daily_move, 2),
    }


def compute_put_call_ratios(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
) -> dict:
    """Put/Call ratios for OI and volume. Uses volume as primary when OI is unavailable."""
    call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
    put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
    call_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
    put_vol = puts['volume'].sum() if 'volume' in puts.columns else 0

    pc_oi = put_oi / call_oi if call_oi > 0 else 1.0
    pc_vol = put_vol / call_vol if call_vol > 0 else 1.0

    # Use volume ratio as primary bias when OI is unreliable
    has_oi = (call_oi + put_oi) > 100

    def _bias(ratio):
        if ratio < 0.7:
            return 'bullish'
        elif ratio > 1.0:
            return 'bearish'
        return 'neutral'

    primary_bias = _bias(pc_oi) if has_oi else _bias(pc_vol)

    return {
        'pc_ratio_oi': round(pc_oi, 3),
        'pc_ratio_volume': round(pc_vol, 3),
        'call_oi_total': int(call_oi),
        'put_oi_total': int(put_oi),
        'call_volume_total': int(call_vol),
        'put_volume_total': int(put_vol),
        'oi_bias': primary_bias,
        'volume_bias': _bias(pc_vol),
    }


def compute_iv_skew(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
) -> dict:
    """
    IV skew: compare OTM put IV vs OTM call IV.

    Higher put IV = institutions hedging downside = bearish sentiment.
    """
    # OTM puts: strike < spot, OTM calls: strike > spot
    otm_puts = puts[puts['strike'] < spot * 0.97].copy()
    otm_calls = calls[calls['strike'] > spot * 1.03].copy()

    put_iv = otm_puts['impliedVolatility'].mean() if not otm_puts.empty else 0.20
    call_iv = otm_calls['impliedVolatility'].mean() if not otm_calls.empty else 0.20

    skew = put_iv / call_iv if call_iv > 0 else 1.0

    if skew > 1.15:
        bias = 'bearish'
    elif skew < 0.90:
        bias = 'bullish'
    else:
        bias = 'neutral'

    return {
        'skew_ratio': round(skew, 3),
        'skew_bias': bias,
        'otm_put_iv': round(put_iv, 4),
        'otm_call_iv': round(call_iv, 4),
    }


# ── Composite Scoring ────────────────────────────────────────────────────

def _score_signal(name, value, bias, evidence, weights=None):
    w = weights if weights is not None else SIGNAL_WEIGHTS
    return {
        'name': name,
        'value': round(value, 4) if isinstance(value, float) else value,
        'bias': bias,
        'weight': w.get(name, 0.0),
        'evidence': evidence,
    }


def _aggregate_signals(signals, spot, walls, fractal_levels, iv_range, max_pain, weights=None):
    """
    Aggregate all signals into floor, ceiling, bias, confidence.

    Floor = weighted average of support levels (put wall, fractal support, IV low, max pain).
    Ceiling = weighted average of resistance levels (call wall, fractal resistance, IV high).
    Bias = weighted vote of all signals.
    Confidence = degree of agreement among signals (0-100).
    """
    # ── Floor (support) ──
    floor_candidates = []
    w = weights if weights is not None else SIGNAL_WEIGHTS

    floor_candidates.append((walls['strongest_put_wall'], w['options_walls']))
    floor_candidates.append((iv_range.get('daily_range_low', spot * 0.99), w['iv_range']))

    if fractal_levels['support_levels']:
        nearest_sup = fractal_levels['support_levels'][0][1]
        floor_candidates.append((nearest_sup, w['fractals']))

    if max_pain < spot:
        floor_candidates.append((max_pain, w['max_pain']))

    total_w = sum(c[1] for c in floor_candidates)
    floor = sum(c[0] * c[1] for c in floor_candidates) / total_w if total_w > 0 else spot * 0.98

    # ── Ceiling (resistance) ──
    ceil_candidates = []
    ceil_candidates.append((walls['strongest_call_wall'], w['options_walls']))
    ceil_candidates.append((iv_range.get('daily_range_high', spot * 1.01), w['iv_range']))

    if fractal_levels['resistance_levels']:
        nearest_res = fractal_levels['resistance_levels'][0][1]
        ceil_candidates.append((nearest_res, w['fractals']))

    if max_pain > spot:
        ceil_candidates.append((max_pain, w['max_pain']))

    total_w = sum(c[1] for c in ceil_candidates)
    ceiling = sum(c[0] * c[1] for c in ceil_candidates) / total_w if total_w > 0 else spot * 1.02

    # ── Bias (weighted vote) ──
    bullish_w, bearish_w, neutral_w = 0.0, 0.0, 0.0
    for sig in signals:
        sw = sig['weight']
        if sig['bias'] == 'bullish':
            bullish_w += sw
        elif sig['bias'] == 'bearish':
            bearish_w += sw
        else:
            neutral_w += sw

    if bullish_w > bearish_w and bullish_w > neutral_w:
        bias = 'BULLISH'
    elif bearish_w > bullish_w and bearish_w > neutral_w:
        bias = 'BEARISH'
    else:
        bias = 'NEUTRAL'

    # Confidence = how dominant the winning bias is
    total = bullish_w + bearish_w + neutral_w
    if total > 0:
        dominant = max(bullish_w, bearish_w, neutral_w)
        confidence = (dominant / total) * 100
    else:
        confidence = 0.0

    return floor, ceiling, bias, confidence


def compute_composite_analysis(
    ticker: str,
    expiry: Optional[str] = None,
    weights: dict = None,
) -> dict:
    """
    Master analysis function. Runs all sub-analyses and produces
    the composite result for the dashboard.
    """
    # 1. Fetch options chain — skip expiries with no usable OI/IV data
    calls, puts, meta = fetch_options_chain(ticker, expiry)
    if calls.empty:
        return {'error': f'No options data for {ticker}'}

    spot = meta['spot_price']
    resolved = meta['resolved_ticker']
    proxy_used = meta['proxy_used']
    price_ratio = meta.get('price_ratio', 1.0)
    expiry_used = meta['expiry']

    # Check if this expiry has usable activity data (OI or volume)
    _MIN_ACTIVITY = 1000
    total_oi = calls['openInterest'].sum() + puts['openInterest'].sum()
    total_vol = calls['volume'].sum() + puts['volume'].sum()
    if total_oi < _MIN_ACTIVITY and total_vol < _MIN_ACTIVITY:
        # Try next expiries until we find one with meaningful activity
        all_expiries = fetch_expiration_dates(ticker)
        for alt_exp in all_expiries[1:10]:
            alt_calls, alt_puts, alt_meta = fetch_options_chain(ticker, alt_exp)
            if alt_calls.empty:
                continue
            alt_oi = alt_calls['openInterest'].sum() + alt_puts['openInterest'].sum()
            alt_vol = alt_calls['volume'].sum() + alt_puts['volume'].sum()
            if alt_oi >= _MIN_ACTIVITY or alt_vol >= _MIN_ACTIVITY:
                calls, puts, meta = alt_calls, alt_puts, alt_meta
                expiry_used = alt_meta['expiry']
                break

    # 2. Fetch price data — use proxy ticker so prices match options chain
    price_ticker = resolved if proxy_used else ticker
    df = fetch_stock_data(price_ticker, period='1y')
    if df.empty:
        return {'error': f'No price data for {ticker}'}
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # 2b. Fetch VIX for IV fallback
    vix_value = None
    try:
        vix_df = fetch_stock_data('^VIX', period='5d')
        if not vix_df.empty:
            vix_value = float(vix_df['Close'].iloc[-1])
    except Exception:
        pass

    # 3. Fractal indicators on price data
    df = add_williams_fractals(df)
    fd_series = calculate_fractal_dimension(df)
    df['fractal_dimension'] = fd_series
    current_fd = float(fd_series.dropna().iloc[-1]) if not fd_series.dropna().empty else 1.5
    regime = classify_regime(current_fd)
    fractal_levels = get_recent_fractal_levels(df)

    # 4. Options analytics
    gex_df = compute_gex_profile(calls, puts, spot, expiry_used)
    walls = compute_options_walls(calls, puts, spot)
    max_pain_strike = compute_max_pain(calls, puts, spot)
    iv_range = compute_iv_expected_move(calls, puts, spot, expiry_used, vix_fallback=vix_value)
    pc_ratios = compute_put_call_ratios(calls, puts)
    skew = compute_iv_skew(calls, puts, spot)

    # 5. Build individual signal scores
    active_w = weights  # None falls through to SIGNAL_WEIGHTS in _score_signal
    signals = []

    # Options walls
    wall_mid = (walls['strongest_put_wall'] + walls['strongest_call_wall']) / 2
    wall_bias = 'bullish' if spot < wall_mid else 'bearish'
    signals.append(_score_signal(
        'options_walls', 0.0, wall_bias,
        f"Put wall ${walls['strongest_put_wall']:.0f} | Call wall ${walls['strongest_call_wall']:.0f}",
        weights=active_w,
    ))

    # GEX
    net_gex = gex_df['net_gex'].sum() if not gex_df.empty else 0
    gex_bias = 'bullish' if net_gex > 0 else 'bearish'
    gex_label = 'positive (sticky/mean-reverting)' if net_gex > 0 else 'negative (slippery/trending)'
    signals.append(_score_signal('gex_levels', net_gex, gex_bias, f"Net GEX: {gex_label}", weights=active_w))

    # IV range (directionally neutral)
    signals.append(_score_signal(
        'iv_range', iv_range['iv_used'], 'neutral',
        f"Daily 1-sigma: ${iv_range['daily_range_low']:.2f} — ${iv_range['daily_range_high']:.2f}",
        weights=active_w,
    ))

    # Fractal structure
    nearest_sup = fractal_levels['support_levels'][0][1] if fractal_levels['support_levels'] else spot * 0.95
    nearest_res = fractal_levels['resistance_levels'][0][1] if fractal_levels['resistance_levels'] else spot * 1.05
    dist_to_sup = spot - nearest_sup
    dist_to_res = nearest_res - spot
    frac_bias = 'bullish' if dist_to_sup < dist_to_res else 'bearish'
    signals.append(_score_signal(
        'fractals', current_fd, frac_bias,
        f"{regime.title()} regime | Support ${nearest_sup:.2f} | Resistance ${nearest_res:.2f}",
        weights=active_w,
    ))

    # Put/Call ratio
    signals.append(_score_signal(
        'put_call_ratio', pc_ratios['pc_ratio_oi'], pc_ratios['oi_bias'],
        f"P/C OI: {pc_ratios['pc_ratio_oi']:.2f} | P/C Vol: {pc_ratios['pc_ratio_volume']:.2f}",
        weights=active_w,
    ))

    # IV skew
    signals.append(_score_signal(
        'iv_skew', skew['skew_ratio'], skew['skew_bias'],
        f"Skew: {skew['skew_ratio']:.2f} (OTM put IV {skew['otm_put_iv']*100:.1f}% vs call {skew['otm_call_iv']*100:.1f}%)",
        weights=active_w,
    ))

    # Max pain
    mp_bias = 'bullish' if spot < max_pain_strike else 'bearish'
    signals.append(_score_signal(
        'max_pain', max_pain_strike, mp_bias,
        f"Max pain ${max_pain_strike:.0f} — spot {'below' if spot < max_pain_strike else 'above'}",
        weights=active_w,
    ))

    # 6. Composite aggregation
    floor, ceiling, bias, confidence = _aggregate_signals(
        signals, spot, walls, fractal_levels, iv_range, max_pain_strike,
        weights=active_w,
    )

    # Scale floor/ceiling for futures proxy
    display_spot = meta.get('futures_spot', spot)
    if proxy_used and price_ratio > 1:
        display_floor = floor * price_ratio
        display_ceiling = ceiling * price_ratio
    else:
        display_floor = floor
        display_ceiling = ceiling
        display_spot = spot

    return {
        'ticker': ticker.upper(),
        'resolved_ticker': resolved,
        'proxy_used': proxy_used,
        'spot_price': round(display_spot, 2),
        'proxy_spot': round(spot, 2),
        'price_ratio': round(price_ratio, 4),
        'timestamp': date.today().isoformat(),
        'expiry': expiry_used,
        'floor': round(display_floor, 2),
        'ceiling': round(display_ceiling, 2),
        'bias': bias,
        'confidence': round(confidence, 1),
        'iv_range': iv_range,
        'gex_profile': gex_df,
        'options_walls': walls,
        'max_pain': round(max_pain_strike, 2),
        'put_call_ratios': pc_ratios,
        'iv_skew': skew,
        'fractal_levels': fractal_levels,
        'fractal_dimension': round(current_fd, 3),
        'market_regime': regime,
        'signals': signals,
        'signal_summary': (
            f"{bias} ({confidence:.0f}% confidence) | "
            f"Floor: ${display_floor:.2f} | Ceiling: ${display_ceiling:.2f}"
        ),
        'price_df': df,
        'gex_df': gex_df,
    }
