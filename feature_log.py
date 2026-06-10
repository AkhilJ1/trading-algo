"""
Feature Log — the ML-ready, point-in-time research ledger.

Both scheduled runs append one wide row of model inputs (and, post-close,
realized outcomes) to the FeatureLog sheet, so every prediction the system
ever grades can later be explained — and a better model trained — from data
captured AT THE MOMENT THE CALL WAS MADE, never reconstructed after the fact
(the Lopez de Prado point-in-time rule: log the inputs with the prediction,
or leakage will quietly poison every backtest).

What gets logged, and the research behind each block:

  * Dealer positioning — net GEX, gamma strength/magnet, max pain, walls,
    put/call ratios, spot→max-pain distance. Pinning is real and measurable
    (Ni, Pearson & Poteshman 2005: optionable stocks cluster at strikes on
    expiry, ~16.5bp average alteration), and the GEX regime sets whether
    moves damp or extend (SqueezeMetrics; Bouchaud & Bonart 2018).
  * Volatility state — IV used, expected moves, VRP (IV systematically
    overstates RV and the gap predicts returns: Bollerslev, Tauchen & Zhou
    2009), Parkinson RV, VIX spot/3M and the term-structure ratio (slope
    carries incremental return predictability: Johnson 2017).
  * Overnight session — prior close, the gap %, overnight high/low/range.
    Overnight returns are economically large and tied to prior-close order
    imbalances (Boyarchenko, Larsen & Whelan, NY Fed "The Overnight Drift"),
    and the OVN range is a load-bearing level set in the Milk RCG framework.
  * Trend & structure — fractal dimension, regime, distance to SMA20/50/200,
    multi-horizon momentum signs (Moskowitz, Ooi & Pedersen TSMOM), RSI,
    prior-day return/range/close-location.
  * Post-close outcomes — OHLCV, overnight vs intraday return split, range
    vs the morning's implied expected move (the calibration that grades the
    vol model), close location value, volume vs 20d, close→max-pain gap
    (the pin outcome itself), first/last-half-hour color when available
    (intraday momentum: Gao, Han, Li & Zhou 2018).

One row per (date, ticker, session) where session is 'pre_open' (6:25am PT)
or 'post_close' (1:16pm PT). Everything is best-effort: a missing feature
logs blank, and a feature-log failure must never block the forecast itself.
"""

import math

import numpy as np
import pandas as pd


def _num(v, nd=4):
    """Round a finite number, else None (serialized to blank by the logger)."""
    try:
        f = float(v)
        return round(f, nd) if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _pct(a, b, nd=4):
    """(a/b - 1) * 100, None-safe."""
    try:
        a, b = float(a), float(b)
        if b == 0 or not (math.isfinite(a) and math.isfinite(b)):
            return None
        return round((a / b - 1.0) * 100.0, nd)
    except (TypeError, ValueError):
        return None


def _mom_sign(close: pd.Series, months: int):
    """Sign of trailing momentum. The 3/6/12-month horizons skip the most
    recent month (TSMOM short-term-reversal convention); the 1-month horizon
    cannot skip a month or it would measure nothing."""
    lb, skip = months * 21, (21 if months > 1 else 1)
    if len(close) <= lb:
        return None
    try:
        ret = float(close.iloc[-skip]) / float(close.iloc[-lb]) - 1.0
        return 1 if ret > 0 else (-1 if ret < 0 else 0)
    except (IndexError, ZeroDivisionError, TypeError):
        return None


def _rsi14(close: pd.Series):
    if len(close) < 15:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    val = 100 - 100 / (1 + rs.iloc[-1])
    return _num(val, 2)


def build_feature_row(result: dict, session: str,
                      vix_val=None, vix_regime=None,
                      overnight: dict = None) -> dict:
    """
    Flatten one analysis result (+ daily history inside it) into the
    FeatureLog row. Pure dict/DataFrame in → dict out; unit-testable offline.
    `overnight` optionally carries {'ovn_high','ovn_low','prior_close'} from
    a pre-market extended-hours fetch.
    """
    overnight = overnight or {}
    spot = _num(result.get('spot_price'), 2)

    # ── dealer positioning ────────────────────────────────────────────
    gex_df = result.get('gex_df')
    gex_net = None
    if gex_df is not None and hasattr(gex_df, 'empty') and not gex_df.empty \
            and 'net_gex' in gex_df:
        gex_net = _num(gex_df['net_gex'].sum(), 0)
    pin = result.get('estimated_close') or {}
    pin = pin if isinstance(pin, dict) else {}
    walls = result.get('options_walls') or {}
    gexb = result.get('gex_boundaries') or {}
    pcr = result.get('put_call_ratios') or {}
    skew = result.get('iv_skew') or {}
    max_pain = _num(result.get('max_pain'), 2)
    dist_max_pain_pct = _pct(max_pain, spot) if (max_pain and spot) else None

    # ── volatility state ──────────────────────────────────────────────
    ivr = result.get('iv_range') or {}
    vrp = result.get('vrp') or {}
    vts = result.get('vix_term_structure') or {}
    daily_em = _num(ivr.get('daily_expected_move'), 2)

    # ── trend & structure (from the daily bars the analysis used) ─────
    hist = result.get('price_df')
    feats_trend = {}
    prior = {}
    if hist is not None and hasattr(hist, 'empty') and not hist.empty:
        close = hist['Close'].astype(float)
        last = hist.iloc[-1]
        for w in (20, 50, 200):
            if len(close) >= w:
                feats_trend[f'dist_sma{w}_pct'] = _pct(close.iloc[-1],
                                                       close.rolling(w).mean().iloc[-1])
        for m in (1, 3, 6, 12):
            feats_trend[f'mom_{m}m_sign'] = _mom_sign(close, m)
        feats_trend['rsi14'] = _rsi14(close)
        rng = float(last['High']) - float(last['Low'])
        prior = {
            'last_bar_date': str(hist.index[-1])[:10],
            'last_close': _num(last['Close'], 2),
            'last_ret_pct': _pct(close.iloc[-1], close.iloc[-2]) if len(close) > 1 else None,
            'last_range_pct': _pct(float(last['High']), float(last['Low'])),
            # Close location value: where the close landed inside the bar's
            # range (1 = at the high, 0 = at the low).
            'last_clv': _num((float(last['Close']) - float(last['Low'])) / rng, 3)
                        if rng > 0 else None,
            'last_volume': _num(last.get('Volume'), 0),
            'vol_vs_20d': _num(float(last.get('Volume', np.nan))
                               / float(hist['Volume'].tail(20).mean()), 3)
                          if 'Volume' in hist and len(hist) >= 20
                          and float(hist['Volume'].tail(20).mean() or 0) > 0 else None,
        }
        # Realized range vs the morning's implied expected move — grades the
        # vol model every single day (range/EM ≈ 1 means well calibrated).
        if daily_em and rng > 0:
            prior['range_vs_em'] = _num(rng / daily_em, 3)

    # ── overnight / gap (pre-open only has these) ─────────────────────
    prior_close = _num(overnight.get('prior_close'), 2) or prior.get('last_close')
    gap_pct = _pct(spot, prior_close) if (session == 'pre_open' and spot and prior_close) else None
    ovn_hi, ovn_lo = _num(overnight.get('ovn_high'), 2), _num(overnight.get('ovn_low'), 2)
    ovn_range_pct = _pct(ovn_hi, ovn_lo) if (ovn_hi and ovn_lo) else None

    row = {
        'session': session,
        'ticker': result.get('ticker', ''),
        'expiry': result.get('expiry', ''),
        'dte': ivr.get('days_to_expiry'),
        'spot': spot,
        'spot_source': result.get('spot_source', ''),
        'chain_source': result.get('source', ''),
        'stale': bool(result.get('stale', False)),
        # forecast levels
        'floor': _num(result.get('floor'), 2),
        'ceiling': _num(result.get('ceiling'), 2),
        'bias': result.get('bias', ''),
        'confidence': _num(result.get('confidence'), 1),
        'est_close': _num(pin.get('estimated_close'), 2),
        'pin_target': _num(pin.get('pin_target'), 2),
        # dealer positioning
        'gex_net': gex_net,
        'gamma_strength': _num(pin.get('gamma_strength'), 3),
        'gamma_pin_strike': _num(pin.get('gamma_pin_strike'), 2),
        'pull_fraction': _num(pin.get('pull_fraction'), 3),
        'gex_floor': _num(gexb.get('gex_floor'), 2),
        'gex_ceiling': _num(gexb.get('gex_ceiling'), 2),
        'max_pain': max_pain,
        'dist_max_pain_pct': dist_max_pain_pct,
        'call_wall': _num(walls.get('strongest_call_wall'), 2),
        'put_wall': _num(walls.get('strongest_put_wall'), 2),
        'pcr_oi': _num(pcr.get('pc_ratio_oi'), 3),
        'pcr_volume': _num(pcr.get('pc_ratio_volume'), 3),
        'skew_ratio': _num(skew.get('skew_ratio'), 3),
        # volatility state
        'iv_used': _num(ivr.get('iv_used'), 4),
        'daily_em': daily_em,
        'em_1sigma': _num(ivr.get('expected_move_1sigma'), 2),
        'vrp_pct': _num(vrp.get('vrp_pct'), 2),
        'parkinson_rv': _num(result.get('parkinson_rv'), 4),
        'vix': _num(vix_val, 2) or _num(vts.get('vix_spot'), 2),
        'vix_3m': _num(vts.get('vix_3m'), 2),
        'vix_term_ratio': _num(vts.get('ratio'), 3),
        'vix_regime': vix_regime or vts.get('structure', ''),
        # trend & structure
        'fractal_dim': _num(result.get('fractal_dimension'), 3),
        'regime': result.get('market_regime', ''),
        'dist_sma20_pct': feats_trend.get('dist_sma20_pct'),
        'dist_sma50_pct': feats_trend.get('dist_sma50_pct'),
        'dist_sma200_pct': feats_trend.get('dist_sma200_pct'),
        'mom_1m_sign': feats_trend.get('mom_1m_sign'),
        'mom_3m_sign': feats_trend.get('mom_3m_sign'),
        'mom_6m_sign': feats_trend.get('mom_6m_sign'),
        'mom_12m_sign': feats_trend.get('mom_12m_sign'),
        'rsi14': feats_trend.get('rsi14'),
        # overnight / gap (pre-open) — Boyarchenko et al. overnight drift,
        # Milk's OVN levels
        'prior_close': prior_close,
        'gap_pct': gap_pct,
        'ovn_high': ovn_hi,
        'ovn_low': ovn_lo,
        'ovn_range_pct': ovn_range_pct,
        # latest realized bar (post-close: today's session; pre-open: prior day)
        'bar_date': prior.get('last_bar_date'),
        'bar_close': prior.get('last_close'),
        'bar_ret_pct': prior.get('last_ret_pct'),
        'bar_range_pct': prior.get('last_range_pct'),
        'bar_clv': prior.get('last_clv'),
        'bar_volume': prior.get('last_volume'),
        'vol_vs_20d': prior.get('vol_vs_20d'),
        'range_vs_em': prior.get('range_vs_em'),
    }
    return row


def fetch_overnight_context(ticker: str) -> dict:
    """
    Pre-market extended-hours context: prior RTH close + overnight high/low
    from prepost minute bars. Best-effort; {} on any failure.
    """
    try:
        import yfinance as yf
        bars = yf.Ticker(ticker).history(period='2d', interval='5m', prepost=True)
        if bars is None or bars.empty:
            return {}
        if bars.index.tz is not None:
            bars = bars.tz_convert('America/New_York')
        days = sorted(set(bars.index.date))
        if len(days) < 2:
            return {}
        prev_day, today = days[-2], days[-1]
        prev_rth = bars[(bars.index.date == prev_day)
                        & (bars.index.hour * 60 + bars.index.minute >= 570)
                        & (bars.index.hour < 16)]
        prior_close = float(prev_rth['Close'].iloc[-1]) if not prev_rth.empty else None
        # Overnight = after prior 16:00 ET through this morning's bars so far.
        cutoff = pd.Timestamp(f'{prev_day} 16:00').tz_localize(bars.index.tz)
        ovn = bars[bars.index > cutoff]
        if ovn.empty:
            return {'prior_close': prior_close}
        # Use bar CLOSES, not High/Low wicks: yfinance pre/post-market bars
        # carry occasional junk tick spikes in the wicks (e.g. a 696 print on
        # a 739 close) that would corrupt the overnight range feature.
        closes = ovn['Close'].dropna()
        if closes.empty:
            return {'prior_close': prior_close}
        return {
            'prior_close': prior_close,
            'ovn_high': float(closes.max()),
            'ovn_low': float(closes.min()),
        }
    except Exception:
        return {}


def log_features(result: dict, session: str, sheets_ok: bool,
                 vix_val=None, vix_regime=None) -> bool:
    """Build + append one feature row. Never raises; returns logged-or-not."""
    try:
        from sheets_logger import log_feature_row, log_feature_row_csv
        overnight = (fetch_overnight_context(result.get('ticker', ''))
                     if session == 'pre_open' else {})
        row = build_feature_row(result, session, vix_val=vix_val,
                                vix_regime=vix_regime, overnight=overnight)
        if sheets_ok and log_feature_row(row):
            return True
        return log_feature_row_csv(row)
    except Exception as e:
        print(f"  [feature_log] failed ({e}) — forecast unaffected")
        return False
