"""
Streamlit Dashboard for trading-algo
--------------------------------------
3-page sidebar navigation:
  1. Daily Scanner  — RSI + Bollinger Band signal scanner
  2. Stock Chart    — Interactive chart with period controls + volume
  3. Backtest       — MA crossover backtest with custom date ranges

Launch:
    python3 -m streamlit run dashboard.py

Compatible with streamlit 1.12.0 (Python 3.9.7 environment).
"""

import sys
import os
import json
from datetime import date, timedelta

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Schwab credentials bridge — MUST run before any project module imports config
# ---------------------------------------------------------------------------
# On Streamlit Cloud the Schwab credentials live in st.secrets, but the data
# layer (providers/) reads them from os.environ, and config.DATA_PROVIDER is
# captured at import time. So we copy the secrets into the environment here,
# before `from config import ...` below. When all three Schwab secrets are
# present we materialize the OAuth token file and flip DATA_PROVIDER to
# "schwab", so the WHOLE dashboard runs Schwab-primary with yfinance as the
# automatic per-call fallback. When they're absent (local dev, no Schwab) this
# is a no-op and the app stays yfinance-only — "only Schwab when available".
def _bridge_schwab_secrets() -> None:
    try:
        secrets = st.secrets
    except Exception:
        return  # no secrets.toml configured at all → yfinance-only

    def _get(key):
        try:
            return secrets[key] if key in secrets else None
        except Exception:
            return None

    api_key = _get("SCHWAB_API_KEY")
    app_secret = _get("SCHWAB_APP_SECRET")
    token_blob = _get("SCHWAB_TOKEN")
    # Without the full credential set Schwab cannot authenticate — stay on
    # yfinance rather than half-configuring and crashing every fetch.
    if not (api_key and app_secret and token_blob):
        return

    os.environ.setdefault("SCHWAB_API_KEY", str(api_key))
    os.environ.setdefault("SCHWAB_APP_SECRET", str(app_secret))

    token_path = os.environ.get("SCHWAB_TOKEN_PATH") or str(
        _get("SCHWAB_TOKEN_PATH") or "schwab_token.json"
    )
    try:
        # Serialize the secret to the exact JSON schwab-py expects. If
        # SCHWAB_TOKEN was pasted into secrets.toml as a bare JSON object (no
        # surrounding quotes), Streamlit parses it into a dict-like — and str()
        # on that emits a PYTHON repr (single quotes, True/None) that schwab-py
        # cannot load, so every Schwab call silently fell back to yfinance.
        # json.dumps() restores valid JSON; a string secret is written verbatim.
        if isinstance(token_blob, str):
            token_str = token_blob
        else:
            token_str = json.dumps(token_blob, default=lambda o: dict(o))
        # Always (re)write from the secret on cold start so re-running
        # schwab_auth.py and pasting a fresh SCHWAB_TOKEN actually takes effect.
        # This bridge runs only at process import (cold start), never on a warm
        # rerun, so it cannot clobber the access token schwab-py refreshes in
        # place during the container's life; the 7-day refresh token does not
        # rotate, so writing the secret's copy never loses a newer one.
        with open(token_path, "w") as fh:
            fh.write(token_str)
    except Exception:
        return  # couldn't persist the token → don't flip to a broken Schwab

    os.environ["SCHWAB_TOKEN_PATH"] = token_path
    # All set: prefer Schwab. setdefault so an explicit env override still wins.
    os.environ.setdefault("DATA_PROVIDER", "schwab")


_bridge_schwab_secrets()

from config import WATCHLIST, MA_SHORT, MA_LONG, RSI_OVERSOLD, BB_WICK_LOOKBACK, SIGNAL_WEIGHTS
from screener import discover_candidates
from data_fetcher import fetch_stock_data
from scanner import scan_ticker
from backtest import backtest_ma_crossover
from strategies.rsi_bollinger import get_buy_signal, add_indicators
from strategies.ma_crossover import generate_signals, current_signal

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📈 Trading Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["📡 Daily Scanner", "📊 Stock Chart", "🔁 Backtest", "🔬 Fractal & Options"],
)
st.sidebar.markdown(
    """
    <style>
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:first-of-type * {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Watchlist: {len(WATCHLIST)} tickers")
st.sidebar.caption(f"Strategy: RSI<{RSI_OVERSOLD} + BB wick | MA({MA_SHORT}/{MA_LONG})")

# Active data backend — so you can SEE at a glance whether Schwab is wired up.
# The credentials bridge above flips DATA_PROVIDER to "schwab" only when all
# three Schwab secrets parse and the token file is written; the Schwab
# FallbackProvider then names itself "schwab->yfinance". If the secrets are
# missing or misnamed this stays "yfinance" and you know the entry didn't take.
# NOTE: this shows what's CONFIGURED. Whether a given chain actually came from
# Schwab is shown per-analysis in the Fractal & Options tab ("Chain via: …"),
# because Schwab can authenticate yet still fall back to yfinance on a bad call.
try:
    from providers import get_provider as _get_provider
    _backend = _get_provider().name
except Exception:
    _backend = os.environ.get("DATA_PROVIDER", "yfinance")
if "schwab" in _backend:
    st.sidebar.caption("🟢 Data backend: **Schwab** primary · yfinance fallback")
else:
    st.sidebar.caption("⚪ Data backend: **yfinance** only — Schwab not detected")


# ── Schwab connection check (on-demand) ─────────────────────────────────────
# The banner above only reflects what's CONFIGURED. When it says Schwab yet
# chains still arrive via yfinance, the cause is a token the Schwab client can't
# read/validate — and the FallbackProvider swallows that exception. This expander
# calls Schwab DIRECTLY with the error uncaught so the true reason is visible.
# It never prints secret VALUES — only whether each secret is present.
with st.sidebar.expander("🔧 Schwab connection check"):
    st.caption("Diagnose why chains may be on yfinance. Never shows secret values.")
    if st.button("Run check", key="schwab_diag_btn"):
        from datetime import datetime as _dt, timezone as _tz, date as _date
        _lines = []
        for _k in ("SCHWAB_API_KEY", "SCHWAB_APP_SECRET", "SCHWAB_TOKEN_PATH",
                   "DATA_PROVIDER"):
            _lines.append(f"{_k}: {'set' if os.environ.get(_k) else 'missing'}")

        _tp = os.environ.get("SCHWAB_TOKEN_PATH", "schwab_token.json")
        if not os.path.exists(_tp):
            _lines.append(f"token file: MISSING at {_tp!r}")
        else:
            try:
                with open(_tp) as _fh:
                    _tok = json.load(_fh)
                _ct = _tok.get("creation_timestamp") or os.path.getmtime(_tp)
                _age = (_dt.now(_tz.utc).timestamp() - float(_ct)) / 86400.0
                _left = 7.0 - _age
                _lines.append(
                    f"token JSON parses: yes (keys={sorted(_tok.keys())})")
                _lines.append(
                    f"token age: {_age:.2f}d — "
                    f"{'EXPIRED, re-auth now' if _left < 0 else f'{_left:.1f}d left of 7d'}")
            except Exception as _e:
                _lines.append(
                    f"token file: UNREADABLE/not-JSON → {type(_e).__name__}: {_e}")

        try:
            from providers.schwab_provider import SchwabProvider
            from providers.quality import chain_is_usable as _ciu
            _sp = SchwabProvider()
            try:
                _sp._get_client()
                _lines.append("client build: OK")
            except Exception as _e:
                _lines.append(f"client build: FAILED → {type(_e).__name__}: {_e}")
            try:
                _exps = _sp.get_expirations("SPY")
                _lines.append(
                    f"get_expirations('SPY'): {len(_exps)} dates, first={_exps[:3]}")
            except Exception as _e:
                _exps = []
                _lines.append(f"get_expirations: RAISED → {type(_e).__name__}: {_e}")

            # Raw HTTP probe: schwab-py does NOT raise on 4xx/5xx, so an empty
            # expirationList hides the real status (a 401/403 entitlement reject
            # looks identical to a genuinely empty list). Surface status + body.
            # The quote call is the discriminator: a 401/403 there means the app
            # lacks Market Data Production entitlement (token still loads fine).
            try:
                _cl = _sp._get_client()
                try:
                    _r = _cl.get_option_expiration_chain("SPY")
                    _lines.append(
                        f"raw expiration-chain: HTTP {_r.status_code} "
                        f"body[:300]={_r.text[:300]!r}")
                except Exception as _e:
                    _lines.append(
                        f"raw expiration-chain RAISED → {type(_e).__name__}: {_e}")
                try:
                    _rq = _cl.get_quote("SPY")
                    _lines.append(
                        f"raw quote('SPY'): HTTP {_rq.status_code} "
                        f"body[:200]={_rq.text[:200]!r}")
                except Exception as _e:
                    _lines.append(f"raw quote RAISED → {type(_e).__name__}: {_e}")
            except Exception as _e:
                _lines.append(f"raw probe setup failed: {type(_e).__name__}: {_e}")

            def _pd(s):
                try:
                    return _dt.strptime(str(s)[:10], "%Y-%m-%d").date()
                except Exception:
                    return None
            _future = [e for e in _exps if (_pd(e) and _pd(e) >= _date.today())]
            if _future:
                _exp = _future[0]
                try:
                    _c, _p = _sp.get_option_chain("SPY", _exp)
                    _nc = 0 if _c is None else len(_c)
                    _np = 0 if _p is None else len(_p)
                    _lines.append(
                        f"get_option_chain('SPY',{_exp}): calls={_nc} puts={_np} "
                        f"usable={_ciu(_c, _p)}")
                except Exception as _e:
                    _lines.append(
                        f"get_option_chain('SPY',{_exp}): RAISED → "
                        f"{type(_e).__name__}: {_e}")
            elif _exps:
                _lines.append("no non-expired expiry available to probe a chain")
        except Exception as _e:
            _lines.append(f"schwab probe import failed: {type(_e).__name__}: {_e}")

        st.code("\n".join(_lines) or "(no output)")
        st.caption("Healthy = 'client build: OK' AND get_option_chain "
                   "'usable=True'. Any RAISED/FAILED line is the real cause.")


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_full_data(ticker: str) -> pd.DataFrame:
    """Fetch 5Y of OHLCV + all indicators. Cached 1 hour."""
    df = fetch_stock_data(ticker, period='5y')
    if df.empty:
        return pd.DataFrame()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = add_indicators(df)
    df = generate_signals(df)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_range_calibration(ticker: str = 'SPY', period: str = '2y') -> dict:
    """
    Engine-true calibration of the evidence-based floor/ceiling range, cached 1h.

    Replays the *actual* range engine over history (VIX as the point-in-time IV
    proxy), grades realized next-session coverage per confidence sigma against
    its Gaussian target, and runs the anchored out-of-sample width sweep. Returns
    {'summary', 'sweep'} (both plain dicts) or {} on insufficient data.
    """
    import range_calibration as rc
    price = fetch_stock_data(ticker, period=period)
    if price is None or price.empty:
        return {}
    vix = fetch_stock_data('^VIX', period=period)
    vix3m = fetch_stock_data('^VIX3M', period=period)
    vix_s = vix['Close'] if vix is not None and not vix.empty else pd.Series(dtype=float)
    v3_s = vix3m['Close'] if vix3m is not None and not vix3m.empty else None
    res = rc.replay_and_summarize(price, vix_s, v3_s)
    sweep = rc.sweep_parameters(price, vix_s, v3_s)
    return {'summary': res['summary'], 'sweep': sweep}


# ---------------------------------------------------------------------------
# Date-range helpers
# ---------------------------------------------------------------------------
PERIOD_LABELS = ['1W', '1M', '3M', '6M', 'YTD', '1Y', '2Y', '5Y']

def period_to_dates(df: pd.DataFrame, period: str):
    """Return (start_date, end_date) as date objects for a given period label."""
    last = df.index[-1].date()
    if period == '1W':
        start = last - timedelta(weeks=1)
    elif period == '1M':
        start = (df.index[-1] - pd.DateOffset(months=1)).date()
    elif period == '3M':
        start = (df.index[-1] - pd.DateOffset(months=3)).date()
    elif period == '6M':
        start = (df.index[-1] - pd.DateOffset(months=6)).date()
    elif period == 'YTD':
        start = date(last.year, 1, 1)
    elif period == '1Y':
        start = (df.index[-1] - pd.DateOffset(years=1)).date()
    elif period == '2Y':
        start = (df.index[-1] - pd.DateOffset(years=2)).date()
    elif period == '5Y':
        start = (df.index[-1] - pd.DateOffset(years=5)).date()
    else:
        start = df.index[0].date()
    start = max(start, df.index[0].date())
    return start, last


def filter_df(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df[(df.index.date >= start) & (df.index.date <= end)]


# ---------------------------------------------------------------------------
# Plotly chart builder
# ---------------------------------------------------------------------------
def build_chart(df: pd.DataFrame, ticker: str, chart_type: str = 'Candlestick') -> go.Figure:
    """3-subplot chart: Price (+ BB + SMAs) | Volume | RSI."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.60, 0.15, 0.25],
        vertical_spacing=0.02,
        subplot_titles=(f"{ticker}", "Volume", "RSI (14)"),
    )

    # ── Price subplot ──────────────────────────────────────────────────────
    if chart_type == 'Candlestick':
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['Close'],
                name='Close',
                line=dict(color='#26a69a', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(38,166,154,0.07)',
            ),
            row=1, col=1,
        )

    # BB bands
    if 'BB_upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_upper'],
                name='BB Upper',
                line=dict(color='rgba(100,149,237,0.5)', width=1),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_lower'],
                name='BB Lower',
                line=dict(color='rgba(100,149,237,0.5)', width=1),
                fill='tonexty',
                fillcolor='rgba(100,149,237,0.07)',
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_mid'],
                name='BB Mid',
                line=dict(color='rgba(100,149,237,0.6)', width=1, dash='dot'),
            ),
            row=1, col=1,
        )

    # SMAs
    sma_s = f'SMA_{MA_SHORT}'
    sma_l = f'SMA_{MA_LONG}'
    if sma_s in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[sma_s], name=f'SMA {MA_SHORT}',
                       line=dict(color='orange', width=1.5)),
            row=1, col=1,
        )
    if sma_l in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df[sma_l], name=f'SMA {MA_LONG}',
                       line=dict(color='#ff69b4', width=1.5)),
            row=1, col=1,
        )

    # BB wick-touch markers
    if 'BB_lower' in df.columns and chart_type == 'Candlestick':
        wicks = df[df['Low'] <= df['BB_lower']]
        if not wicks.empty:
            fig.add_trace(
                go.Scatter(
                    x=wicks.index, y=wicks['Low'] * 0.998,
                    mode='markers', name='BB Wick Touch',
                    marker=dict(symbol='triangle-up', color='lime', size=9),
                ),
                row=1, col=1,
            )

    # ── Volume subplot ─────────────────────────────────────────────────────
    if 'Volume' in df.columns:
        vol_colors = [
            '#26a69a' if c >= o else '#ef5350'
            for c, o in zip(df['Close'], df['Open'])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df['Volume'],
                marker_color=vol_colors,
                name='Volume',
                showlegend=False,
            ),
            row=2, col=1,
        )

    # ── RSI subplot ────────────────────────────────────────────────────────
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['RSI'],
                name='RSI',
                line=dict(color='#9c27b0', width=1.5),
            ),
            row=3, col=1,
        )
        for level, color, label in [(30, '#26a69a', 'Oversold'), (70, '#ef5350', 'Overbought')]:
            fig.add_hline(
                y=level,
                line=dict(color=color, width=1, dash='dash'),
                row=3, col=1,
                annotation_text=label,
                annotation_position='right',
            )

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=50, b=20, l=0, r=80),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
    )
    fig.update_xaxes(gridcolor='#1e2130', showgrid=True)
    fig.update_yaxes(gridcolor='#1e2130', showgrid=True)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    return fig


# ===========================================================================
# Shared scanner helpers
# ===========================================================================
def _run_scan(tickers: list) -> dict:
    """Scan a list of tickers with full signal detection; return {buy, sell, watch, opportunities, all, errors} dict."""
    from scanner import scan_ticker_full
    buy, sell, watch, all_results, errors = [], [], [], [], []
    status = st.empty()
    bar = st.progress(0)
    for i, ticker in enumerate(tickers):
        status.text(f'Scanning {ticker}… ({i+1}/{len(tickers)})')
        bar.progress((i + 1) / len(tickers))
        try:
            result = scan_ticker_full(ticker)
            if result is None:
                errors.append((ticker, 'Insufficient data'))
            else:
                all_results.append(result)
                if result['buy_signal']:
                    buy.append(result)
                elif result.get('sell_signal'):
                    sell.append(result)
                elif result['rsi_oversold']:
                    watch.append(result)
        except Exception as e:
            errors.append((ticker, str(e)))
    status.empty()
    bar.empty()
    buy.sort(key=lambda x: x.get('composite_score', x['strength']), reverse=True)
    sell.sort(key=lambda x: x.get('sell_strength', 0), reverse=True)
    watch.sort(key=lambda x: x['rsi'])
    all_results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)

    # Collect all opportunities across tickers
    all_opps = []
    for r in all_results:
        for opp in r.get('opportunities', []):
            opp['ticker'] = r['ticker']
            opp['price'] = r['price']
            all_opps.append(opp)
    all_opps.sort(key=lambda x: (x['tier'], -x['confidence']))

    return {'buy': buy, 'sell': sell, 'watch': watch, 'opportunities': all_opps,
            'all': all_results, 'errors': errors}


def _tier_badge(tier: int) -> str:
    """Return HTML badge for signal confidence tier."""
    if tier == 1:
        return '<span style="background:#1b5e20;color:#a5d6a7;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold;">PROVEN</span>'
    elif tier == 2:
        return '<span style="background:#e65100;color:#ffcc80;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold;">VALIDATED</span>'
    else:
        return '<span style="background:#424242;color:#bdbdbd;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold;">SPECULATIVE</span>'


def _render_scan_results(results: dict, total: int) -> None:
    """Render buy signals, sell signals, opportunities, consensus, watch list, and errors."""
    buy_signals   = results['buy']
    sell_signals  = results.get('sell', [])
    watch_only    = results['watch']
    opportunities = results.get('opportunities', [])
    all_results   = results.get('all', [])
    errors        = results['errors']

    # ── Summary metrics ───────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric('Scanned', total)
    c2.metric('Buy Signals', len(buy_signals))
    c3.metric('Sell Signals', len(sell_signals))
    c4.metric('Opportunities', len(opportunities))
    c5.metric('Watching', len(watch_only))
    c6.metric('Errors', len(errors))
    st.markdown('---')

    # ── BUY SIGNALS ───────────────────────────────────────────────────
    if buy_signals:
        st.subheader('Buy Signals')
        for sig in buy_signals:
            xo = sig['ma_last_crossover']
            xo_str = ''
            if xo:
                d = xo['date'].strftime('%Y-%m-%d') if hasattr(xo['date'], 'strftime') else str(xo['date'])[:10]
                xo_str = f"Last {xo['type'].replace('_', ' ')} on {d} @ ${xo['price']:.2f}"
            wick_dates = ', '.join(sig['wick_dates'][-3:]) if sig['wick_dates'] else 'none'
            factors = sig.get('factors')
            composite = sig.get('composite_score', sig['strength'])
            with st.expander(
                f"**{sig['ticker']}**  —  ${sig['price']:.2f}  |  Score {composite:.0f}/100",
                expanded=True,
            ):
                st.markdown(_tier_badge(1) + '&nbsp; RSI + Bollinger Band buy signal', unsafe_allow_html=True)
                r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                r1c1.metric('Price', f"${sig['price']:.2f}")
                r1c2.metric('RSI', f"{sig['rsi']:.1f}")
                r1c3.metric('Strength', f"{sig['strength']:.0f}/100")
                r1c4.metric('Composite', f"{composite:.0f}/100")
                if factors:
                    def _dot(val, label):
                        color = '#26a69a' if val > 0 else '#666'
                        return f'<span style="color:{color};font-weight:bold;">{label}</span>'
                    dots = ' · '.join([
                        _dot(factors['rsi_score'], f"RSI {factors['rsi_score']:.0f}"),
                        _dot(factors['zscore_score'], f"Z {factors['zscore_score']:.0f}"),
                        _dot(factors['volume_score'], f"Vol {factors['volume_score']:.0f}"),
                        _dot(factors['regime_score'], f"Rgm {factors['regime_score']:.0f}"),
                        _dot(factors['atr_score'], f"ATR {factors['atr_score']:.0f}"),
                    ])
                    st.markdown(dots, unsafe_allow_html=True)
                    st.caption(f"Z-Score: {factors['zscore']:.2f}  |  Regime: {factors['regime'].title()}  |  ATR: {factors['atr_state']}")
                st.markdown(
                    f"**MA Trend:** {sig['ma_trend'].upper()}  |  "
                    f"**Wick touches:** {sig['wick_touches']} → {wick_dates}"
                )
                if xo_str:
                    st.caption(xo_str)
                # Strategy consensus inline
                cons = sig.get('consensus')
                if cons:
                    _render_consensus_inline(cons)

    # ── SELL SIGNALS ──────────────────────────────────────────────────
    if sell_signals:
        st.subheader('Sell Signals')
        for sig in sell_signals:
            sell_str = sig.get('sell_strength', 0)
            comps = sig.get('sell_components', {})
            with st.expander(
                f"**{sig['ticker']}**  —  ${sig['price']:.2f}  |  Sell Strength {sell_str:.0f}/100",
                expanded=True,
            ):
                # Determine tier based on active count
                tier = 1 if sig.get('sell_active_count', 0) >= 3 else 2
                st.markdown(_tier_badge(tier) + '&nbsp; Overbought / sell signal', unsafe_allow_html=True)
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric('Price', f"${sig['price']:.2f}")
                sc2.metric('RSI', f"{sig['rsi']:.1f}")
                sc3.metric('Sell Strength', f"{sell_str:.0f}/100")
                sc4.metric('Conditions Met', f"{sig.get('sell_active_count', 0)}/4")
                # Component breakdown
                comp_parts = []
                if comps.get('rsi_overbought'):
                    comp_parts.append('RSI > 70')
                if comps.get('zscore_extended'):
                    comp_parts.append('Z-Score > +2.0')
                if comps.get('above_keltner'):
                    comp_parts.append('Above Keltner Upper')
                if comps.get('above_bb_upper'):
                    comp_parts.append('Above BB Upper')
                if comp_parts:
                    st.markdown('**Active conditions:** ' + ' · '.join(comp_parts))
                st.caption(f"MA Trend: {sig['ma_trend'].upper()}")
                cons = sig.get('consensus')
                if cons:
                    _render_consensus_inline(cons)

    # ── OPPORTUNITIES ─────────────────────────────────────────────────
    if opportunities:
        st.subheader('Opportunities')
        st.caption('Actionable setups detected across your watchlist — grouped by confidence tier')

        # Detect conflicting signals per ticker
        _opp_dirs = {}
        for opp in opportunities:
            _opp_dirs.setdefault(opp['ticker'], set()).add(opp['direction'])
        conflicted_tickers = {t for t, dirs in _opp_dirs.items() if len(dirs) > 1}

        if conflicted_tickers:
            tickers_str = ', '.join(sorted(conflicted_tickers))
            st.warning(
                f"**Conflicting signals for {tickers_str}** — "
                "different timeframes disagree. A short-term mean reversion (bounce) "
                "can coexist with a longer-term downtrend. Higher-tier signals carry more weight; "
                "consider your holding period when choosing which to act on."
            )

        # Group by tier
        tier_groups = {1: [], 2: [], 3: []}
        for opp in opportunities:
            tier_groups[opp['tier']].append(opp)

        for tier_num, tier_label in [(1, 'Proven (Statistical Basis)'), (2, 'Validated (Backtested)'), (3, 'Speculative')]:
            items = tier_groups[tier_num]
            if not items:
                continue
            st.markdown(f"#### {tier_label}")
            for opp in items:
                dir_icon = '↑' if opp['direction'] == 'long' else '↓'
                dir_color = '#26a69a' if opp['direction'] == 'long' else '#ef5350'
                # Conflict indicator
                conflict_tag = ''
                if opp['ticker'] in conflicted_tickers:
                    conflict_tag = '&nbsp;&nbsp;<span style="color:#ffd600;font-size:11px;" title="This ticker has opposing signals at different timeframes">CONFLICTED</span>'
                html = (
                    f'<div style="background:#1a1f2e;padding:10px 14px;border-radius:6px;'
                    f'border-left:3px solid {dir_color};margin:4px 0;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<div>'
                    f'<span style="font-weight:bold;font-size:14px;color:#fafafa;">{opp["ticker"]}</span>'
                    f'&nbsp;&nbsp;{_tier_badge(tier_num)}'
                    f'&nbsp;&nbsp;<span style="color:{dir_color};font-weight:bold;">{dir_icon} {opp["direction"].upper()}</span>'
                    f'&nbsp;&nbsp;<span style="color:#90caf9;font-size:13px;">{opp["setup"]}</span>'
                    f'{conflict_tag}'
                    f'</div>'
                    f'<div style="color:#fff;font-weight:bold;font-size:13px;">'
                    f'${opp["price"]:.2f} &nbsp;|&nbsp; Confidence: {opp["confidence"]}%</div>'
                    f'</div>'
                    f'<div style="color:#aaa;font-size:12px;margin-top:4px;">{opp["reason"]}</div>'
                    f'</div>'
                )
                st.markdown(html, unsafe_allow_html=True)
    elif not buy_signals and not sell_signals:
        st.info('No actionable signals or opportunities detected. All tickers are in neutral territory.')

    # ── STRATEGY CONSENSUS ────────────────────────────────────────────
    consensus_tickers = [r for r in all_results if r.get('consensus')]
    if consensus_tickers:
        with st.expander('Strategy Consensus (all tickers)', expanded=False):
            st.caption('Latest signal from each strategy: +1 bullish, -1 bearish, 0 neutral')
            cons_rows = []
            for r in consensus_tickers:
                cons = r['consensus']
                row = {'Ticker': r['ticker']}
                for strat, val in cons['signals'].items():
                    row[strat] = val
                row['Bullish'] = cons['bullish_count']
                row['Bearish'] = cons['bearish_count']
                row['Consensus'] = cons['consensus'].title()
                cons_rows.append(row)
            if cons_rows:
                df_cons = pd.DataFrame(cons_rows)
                st.dataframe(df_cons, width='stretch')

    # ── WATCH LIST ────────────────────────────────────────────────────
    if watch_only:
        st.subheader('Watch — RSI Oversold, Awaiting BB Wick Touch')
        watch_data = []
        for s in watch_only:
            row = {
                'Ticker': s['ticker'],
                'Price': f"${s['price']:.2f}",
                'RSI': round(s['rsi'], 1),
                'BB Lower': f"${s['bb_lower']:.2f}",
                'BB Upper': f"${s['bb_upper']:.2f}",
                'MA Trend': s['ma_trend'].upper(),
            }
            if s.get('factors'):
                row['Regime'] = s['factors']['regime'].title()
                row['Z-Score'] = s['factors']['zscore']
                row['Score'] = s.get('composite_score', 0)
            watch_data.append(row)
        st.dataframe(pd.DataFrame(watch_data), width='stretch')

    # ── FACTOR BREAKDOWN ──────────────────────────────────────────────
    if all_results:
        with st.expander('Factor Breakdown (all scanned tickers)'):
            factor_rows = []
            for r in all_results:
                f = r.get('factors')
                row = {
                    'Ticker': r['ticker'],
                    'Price': f"${r['price']:.2f}",
                    'RSI': round(r['rsi'], 1),
                    'Signal': 'BUY' if r['buy_signal'] else ('SELL' if r.get('sell_signal') else ('WATCH' if r['rsi_oversold'] else '—')),
                }
                if f:
                    row.update({
                        'RSI Score': f['rsi_score'],
                        'Z-Score': f['zscore'],
                        'Z Score': f['zscore_score'],
                        'Vol Score': f['volume_score'],
                        'Regime': f['regime'].title(),
                        'Regime Score': f['regime_score'],
                        'ATR Score': f['atr_score'],
                        'Composite': r.get('composite_score', 0),
                    })
                factor_rows.append(row)
            st.dataframe(pd.DataFrame(factor_rows), width='stretch')

    if errors:
        with st.expander(f'Errors ({len(errors)})'):
            for ticker, err in errors:
                st.warning(f'**{ticker}**: {err}')


def _render_consensus_inline(cons: dict) -> None:
    """Render strategy consensus as a compact inline display."""
    signals = cons['signals']
    parts = []
    for strat, val in signals.items():
        if val > 0:
            parts.append(f'<span style="color:#26a69a;">+{strat}</span>')
        elif val < 0:
            parts.append(f'<span style="color:#ef5350;">-{strat}</span>')
        else:
            parts.append(f'<span style="color:#666;">{strat}</span>')
    label = cons['consensus'].title()
    color = '#26a69a' if label == 'Bullish' else '#ef5350' if label == 'Bearish' else '#90caf9'
    st.markdown(
        f'<div style="font-size:12px;margin-top:4px;">'
        f'<span style="color:{color};font-weight:bold;">Consensus: {label}</span> '
        f'({cons["bullish_count"]}B / {cons["bearish_count"]}S) &nbsp;|&nbsp; '
        f'{" · ".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def _candidate_card_html(r: dict) -> str:
    """Return an HTML card for a dynamic screener candidate."""
    ticker = r['ticker']
    if r['buy_signal']:
        border = '#26a69a'
        badge  = f'<span style="color:#26a69a;font-weight:bold;">● BUY &nbsp;{r["strength"]:.0f}/100</span>'
    elif r['rsi_oversold']:
        border = '#ffd600'
        badge  = '<span style="color:#ffd600;font-weight:bold;">● WATCH</span>'
    else:
        border = '#4a90d9'
        badge  = '<span style="color:#4a90d9;font-weight:bold;">● Near oversold</span>'

    return (
        f'<div style="background:#1a1f2e;padding:12px;border-radius:8px;'
        f'border-left:3px solid {border};margin:3px 0;">'
        f'<div style="font-size:15px;font-weight:bold;color:#fafafa;">{ticker}</div>'
        f'<div style="margin-top:5px;">{badge}</div>'
        f'<div style="margin-top:6px;color:#ccc;font-size:13px;">'
        f'<strong>${r["price"]:.2f}</strong> &nbsp;|&nbsp; RSI {r["rsi"]:.1f} &nbsp;|&nbsp; Wicks {r["wick_touches"]}</div>'
        f'<div style="margin-top:3px;color:#888;font-size:11px;">'
        f'Score {r["score"]:.0f} &nbsp;·&nbsp; Win rate {r["win_rate"]:.0f}% ({r["num_trades"]} trades)</div>'
        f'</div>'
    )


# ===========================================================================
# PAGE 1: Daily Scanner
# ===========================================================================
if page == '📡 Daily Scanner':
    st.markdown("<h1 style='color: #ffffff;'>📡 Daily Scanner</h1>", unsafe_allow_html=True)
    st.caption('Multi-factor scanner: buy signals, sell signals, and opportunities across your watchlist')

    # ── Session state ──────────────────────────────────────────────────────
    if 'dynamic_results' not in st.session_state:
        st.session_state.dynamic_results = None
        st.session_state.dynamic_time = None
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = None
        st.session_state.scan_time = None
        st.session_state.scan_total = 0

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — Dynamic Candidates (S&P 500 weekly screen)
    # ══════════════════════════════════════════════════════════════════════
    dyn_hdr, dyn_time_col, dyn_btn_col = st.columns([3, 2, 1])
    with dyn_hdr:
        st.subheader('🎯 Dynamic Candidates')
        st.caption('Scans S&P 500 weekly charts · ranked by signal strength + backtest win rate')
    with dyn_time_col:
        if st.session_state.dynamic_time:
            st.write('')
            st.caption(f'Last run: {st.session_state.dynamic_time}')
    with dyn_btn_col:
        st.write('')
        st.write('')
        discover_btn = st.button('Discover')

    if discover_btn:
        prog_bar = st.progress(0.0)
        prog_text = st.empty()

        def _dyn_progress(pct, msg):
            prog_bar.progress(min(pct, 1.0))
            prog_text.text(msg)

        results = discover_candidates(progress_callback=_dyn_progress, top_n=15)
        prog_bar.empty()
        prog_text.empty()

        st.session_state.dynamic_results = results
        from datetime import datetime
        st.session_state.dynamic_time = datetime.now().strftime('%H:%M:%S')

    if st.session_state.dynamic_results is not None:
        candidates = st.session_state.dynamic_results
        if not candidates:
            st.info('No candidates met the weekly RSI threshold right now. Markets may not be oversold.')
        else:
            st.caption(f'{len(candidates)} candidates found — weekly chart analysis + backtest ranked')
            CARDS_PER_ROW = 5
            for row_start in range(0, len(candidates), CARDS_PER_ROW):
                chunk = candidates[row_start:row_start + CARDS_PER_ROW]
                cols = st.columns(CARDS_PER_ROW)
                for j, r in enumerate(chunk):
                    cols[j].markdown(_candidate_card_html(r), unsafe_allow_html=True)
    else:
        st.info('Click **Discover** to scan S&P 500 weekly charts and surface the highest-conviction setups. Takes ~1–2 minutes.')

    st.markdown('---')

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — Daily Watchlist Scanner (auto-runs on page load)
    # ══════════════════════════════════════════════════════════════════════
    scan_hdr, scan_time_col, rescan_col = st.columns([3, 2, 1])
    with scan_hdr:
        st.subheader('🔍 Daily Watchlist Scan')
    with scan_time_col:
        if st.session_state.scan_time:
            st.write('')
            st.caption(f'Last scanned: {st.session_state.scan_time}')
    with rescan_col:
        st.write('')
        rescan_btn = st.button('Re-scan')

    custom_input = st.text_input(
        'Override tickers (comma-separated) — leave blank to scan full watchlist',
        placeholder='e.g. AAPL, NVDA, TSLA',
    )

    # Auto-run on first page load; re-run on button click
    needs_scan = (st.session_state.scan_results is None) or rescan_btn
    if needs_scan:
        tickers = (
            [t.strip().upper() for t in custom_input.split(',') if t.strip()]
            if custom_input.strip() else WATCHLIST
        )
        results = _run_scan(tickers)
        st.session_state.scan_results = results
        st.session_state.scan_total = len(tickers)
        from datetime import datetime
        st.session_state.scan_time = datetime.now().strftime('%H:%M:%S')

    if st.session_state.scan_results:
        _render_scan_results(
            st.session_state.scan_results,
            st.session_state.scan_total,
        )


# ===========================================================================
# PAGE 2: Stock Chart
# ===========================================================================
elif page == '📊 Stock Chart':
    st.title('📊 Stock Chart')

    # ── Session state init ─────────────────────────────────────────────────
    if 'chart_ticker' not in st.session_state:
        st.session_state.chart_ticker = ''
    if 'chart_period' not in st.session_state:
        st.session_state.chart_period = '1Y'
    if 'chart_custom' not in st.session_state:
        st.session_state.chart_custom = False

    # ── Ticker input ───────────────────────────────────────────────────────
    t_col, b_col, ct_col = st.columns([3, 1, 2])
    with t_col:
        ticker_input = st.text_input('Ticker', value='AAPL')
    with b_col:
        st.write('')
        load_btn = st.button('Load Chart')
    with ct_col:
        chart_type = st.selectbox('Chart type', ['Candlestick', 'Line'])

    if load_btn:
        st.session_state.chart_ticker = ticker_input.strip().upper()
        st.session_state.chart_custom = False

    ticker = st.session_state.chart_ticker or ticker_input.strip().upper()
    if not ticker:
        st.info('Enter a ticker and click Load Chart.')
        st.stop()

    # ── Load data ──────────────────────────────────────────────────────────
    with st.spinner(f'Loading {ticker}…'):
        df_full = load_full_data(ticker)

    if df_full.empty:
        st.error(f'No data for **{ticker}**. Check the ticker symbol.')
        st.stop()

    # ── Period quick-select buttons ────────────────────────────────────────
    st.markdown('**Time range**')
    btn_cols = st.columns(len(PERIOD_LABELS) + 1)
    for i, p in enumerate(PERIOD_LABELS):
        label = f'[ {p} ]' if (st.session_state.chart_period == p and not st.session_state.chart_custom) else p
        if btn_cols[i].button(label, key=f'cp_{p}'):
            st.session_state.chart_period = p
            st.session_state.chart_custom = False
    if btn_cols[-1].button('Custom'):
        st.session_state.chart_custom = True

    # ── Date range ─────────────────────────────────────────────────────────
    if st.session_state.chart_custom:
        default_start, default_end = period_to_dates(df_full, '1Y')
        d1, d2 = st.columns(2)
        custom_start = d1.date_input('From', value=default_start,
                                     min_value=df_full.index[0].date(),
                                     max_value=df_full.index[-1].date())
        custom_end = d2.date_input('To', value=default_end,
                                   min_value=df_full.index[0].date(),
                                   max_value=df_full.index[-1].date())
        range_start, range_end = custom_start, custom_end
    else:
        range_start, range_end = period_to_dates(df_full, st.session_state.chart_period)

    df_display = filter_df(df_full, range_start, range_end)
    if df_display.empty:
        st.warning('No data in selected range.')
        st.stop()

    # ── Overlay controls ─────────────────────────────────────────────────
    from strategies.chart_indicators import (
        compute_vwap, compute_anchored_vwap, find_anchor_events,
        compute_volume_profile, compute_zscore, compute_keltner_channels,
        compute_atr_stops, compute_confluence_signals,
    )
    from strategies.fractal_indicators import (
        add_williams_fractals, calculate_fractal_dimension,
        classify_regime, get_recent_fractal_levels,
    )

    with st.expander('Chart Overlays', expanded=False):
        ov1, ov2, ov3, ov4 = st.columns(4)
        show_vwap = ov1.checkbox('VWAP', value=True)
        show_keltner = ov2.checkbox('Keltner Channels', value=False)
        show_vol_profile = ov3.checkbox('Volume Profile', value=True)
        show_fractal_sr = ov4.checkbox('Fractal S/R', value=True)
        ov5, ov6, ov7, ov8 = st.columns(4)
        show_zscore = ov5.checkbox('Z-Score Subplot', value=False)
        show_atr_stop = ov6.checkbox('ATR Stop', value=True)
        show_anchored_vwap = ov7.checkbox('Anchored VWAP', value=False)
        show_entries = ov8.checkbox('Entry/Exit Markers', value=True)

    # Strategy for entry/exit markers
    if show_entries:
        STRAT_OPTIONS = ['MA Crossover', 'MACD + RSI', 'BB Squeeze', 'TSMOM', 'Turtle', 'Fractal', 'Ensemble']
        chart_strategy = st.selectbox('Signal strategy for entries/exits', STRAT_OPTIONS, index=5, key='chart_strat')

    # ── Compute indicators on full data ───────────────────────────────────
    try:
        signal = get_buy_signal(df_full)
        ma_info = current_signal(df_full)
    except Exception as e:
        st.error(f'Signal error: {e}')
        st.stop()

    # Fractal levels (for S/R lines and confluence)
    df_frac = add_williams_fractals(df_full.copy())
    fd_series = calculate_fractal_dimension(df_frac)
    current_fd = float(fd_series.dropna().iloc[-1]) if not fd_series.dropna().empty else 1.5
    regime = classify_regime(current_fd)
    fractal_levels = get_recent_fractal_levels(df_frac)

    # Confluence signals
    confluence = compute_confluence_signals(df_full, fractal_levels)
    latest_score = int(confluence['signal_score'].iloc[-1])
    latest_label = confluence['signal_label'].iloc[-1]

    # ── Confluence signal banner ──────────────────────────────────────────
    score_details = []
    last_row = confluence.iloc[-1]
    if last_row.get('rsi_buy', False):
        score_details.append('RSI oversold')
    if last_row.get('rsi_sell', False):
        score_details.append('RSI overbought')
    if last_row.get('zscore_buy', False):
        score_details.append('Z-Score extreme low')
    if last_row.get('zscore_sell', False):
        score_details.append('Z-Score extreme high')
    if last_row.get('keltner_buy', False):
        score_details.append('Below Keltner')
    if last_row.get('keltner_sell', False):
        score_details.append('Above Keltner')
    if last_row.get('fractal_buy', False):
        score_details.append('Near fractal support')
    if last_row.get('fractal_sell', False):
        score_details.append('Near fractal resistance')
    if last_row.get('vol_surge', False):
        score_details.append('Volume surge')
    detail_str = ' + '.join(score_details) if score_details else 'No signals firing'

    if latest_score >= 2:
        st.success(f"STRONG BUY  —  Confluence {latest_score}/5  |  {detail_str}")
    elif latest_score == 1:
        st.success(f"BUY  —  Confluence {latest_score}/5  |  {detail_str}")
    elif latest_score == 0:
        st.info(f"NEUTRAL  —  {detail_str}")
    elif latest_score == -1:
        st.warning(f"SELL  —  Confluence {latest_score}/5  |  {detail_str}")
    else:
        st.error(f"STRONG SELL  —  Confluence {latest_score}/5  |  {detail_str}")

    # ── Metrics rows ──────────────────────────────────────────────────────
    vwap_val = float(compute_vwap(df_full).iloc[-1]) if len(df_full) >= 20 else 0
    zscore_val = float(compute_zscore(df_full).iloc[-1]) if len(df_full) >= 50 else 0
    atr_stop_val = float(compute_atr_stops(df_full).iloc[-1]) if len(df_full) >= 14 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric('Price', f"${signal['close']:.2f}")
    m2.metric('VWAP', f"${vwap_val:.2f}")
    m3.metric('Z-Score', f"{zscore_val:.2f}")
    m4.metric('ATR Stop', f"${atr_stop_val:.2f}")
    m5.metric('Confluence', f"{latest_score}/5")

    n1, n2, n3, n4, n5 = st.columns(5)
    n1.metric('RSI', f"{signal['rsi']:.1f}")
    n2.metric('BB Lower/Upper', f"${signal['bb_lower']:.0f} / ${signal['bb_upper']:.0f}")
    n3.metric(f'SMA {MA_SHORT}/{MA_LONG}', f"${ma_info.get(f'sma_{MA_SHORT}', 0):.0f} / ${ma_info.get(f'sma_{MA_LONG}', 0):.0f}")
    n4.metric('Regime', regime.title())
    n5.metric('MA Trend', ma_info['trend'].upper())

    # ── Build enhanced chart ──────────────────────────────────────────────
    n_rows = 3 + (1 if show_zscore else 0)
    row_heights = [0.55, 0.15, 0.20] + ([0.10] if show_zscore else [])
    subtitles = [ticker, 'Volume', 'RSI (14)'] + (['Z-Score'] if show_zscore else [])

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.02,
        subplot_titles=subtitles,
    )

    # Price
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=df_display.index, open=df_display['Open'], high=df_display['High'],
            low=df_display['Low'], close=df_display['Close'], name='Price',
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df_display.index, y=df_display['Close'], name='Close',
            line=dict(color='#26a69a', width=1.5),
            fill='tozeroy', fillcolor='rgba(38,166,154,0.07)',
        ), row=1, col=1)

    # BB bands
    if 'BB_upper' in df_display.columns:
        fig.add_trace(go.Scatter(x=df_display.index, y=df_display['BB_upper'], name='BB Upper',
                                  line=dict(color='rgba(100,149,237,0.5)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_display.index, y=df_display['BB_lower'], name='BB Lower',
                                  line=dict(color='rgba(100,149,237,0.5)', width=1),
                                  fill='tonexty', fillcolor='rgba(100,149,237,0.07)'), row=1, col=1)

    # SMAs
    sma_s, sma_l = f'SMA_{MA_SHORT}', f'SMA_{MA_LONG}'
    if sma_s in df_display.columns:
        fig.add_trace(go.Scatter(x=df_display.index, y=df_display[sma_s], name=f'SMA {MA_SHORT}',
                                  line=dict(color='orange', width=1.5)), row=1, col=1)
    if sma_l in df_display.columns:
        fig.add_trace(go.Scatter(x=df_display.index, y=df_display[sma_l], name=f'SMA {MA_LONG}',
                                  line=dict(color='#ff69b4', width=1.5)), row=1, col=1)

    # VWAP overlay
    if show_vwap:
        vwap_series = compute_vwap(df_display)
        fig.add_trace(go.Scatter(x=df_display.index, y=vwap_series, name='VWAP',
                                  line=dict(color='#ffeb3b', width=1.5, dash='dot')), row=1, col=1)

    # Keltner Channels
    if show_keltner:
        k_u, k_m, k_l = compute_keltner_channels(df_display)
        fig.add_trace(go.Scatter(x=df_display.index, y=k_u, name='Keltner Upper',
                                  line=dict(color='rgba(255,152,0,0.5)', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_display.index, y=k_l, name='Keltner Lower',
                                  line=dict(color='rgba(255,152,0,0.5)', width=1),
                                  fill='tonexty', fillcolor='rgba(255,152,0,0.05)'), row=1, col=1)

    # ATR Stop
    if show_atr_stop:
        atr_stop = compute_atr_stops(df_display)
        fig.add_trace(go.Scatter(x=df_display.index, y=atr_stop, name='ATR Stop',
                                  line=dict(color='#ef5350', width=1, dash='dot')), row=1, col=1)

    # Fractal S/R levels
    if show_fractal_sr and fractal_levels:
        for _, level in fractal_levels.get('support_levels', [])[:3]:
            fig.add_hline(y=level, line=dict(color='#26a69a', width=1, dash='dash'),
                          annotation_text=f'S ${level:.0f}', annotation_position='right',
                          annotation=dict(font_size=10, font_color='#26a69a'), row=1, col=1)
        for _, level in fractal_levels.get('resistance_levels', [])[:3]:
            fig.add_hline(y=level, line=dict(color='#ef5350', width=1, dash='dash'),
                          annotation_text=f'R ${level:.0f}', annotation_position='right',
                          annotation=dict(font_size=10, font_color='#ef5350'), row=1, col=1)

    # Anchored VWAP
    if show_anchored_vwap:
        events = find_anchor_events(df_full)
        avwap_colors = ['#00bcd4', '#ff5722', '#8bc34a', '#e91e63', '#9c27b0']
        for i, evt in enumerate(events[:3]):
            avwap = compute_anchored_vwap(df_display, evt['date'])
            fig.add_trace(go.Scatter(
                x=df_display.index, y=avwap,
                name=f"AVWAP {evt['event']}",
                line=dict(color=avwap_colors[i % len(avwap_colors)], width=1, dash='dashdot'),
            ), row=1, col=1)

    # Volume Profile (horizontal bars anchored to right edge of chart)
    if show_vol_profile:
        vp = compute_volume_profile(df_display)
        if vp['volumes']:
            max_vol = max(vp['volumes']) if max(vp['volumes']) > 0 else 1
            # Use date-based x coordinates: bars extend leftward from the last date
            n_bars_display = len(df_display)
            x_end = df_display.index[-1]
            # Each bar extends leftward by up to 20% of the visible date range
            if n_bars_display > 1:
                date_span = (df_display.index[-1] - df_display.index[0])
                max_bar_span = date_span * 0.20
            else:
                max_bar_span = pd.Timedelta(days=5)

            bar_width = (vp['bin_centers'][1] - vp['bin_centers'][0]) * 0.8 if len(vp['bin_centers']) > 1 else 1
            for center, vol in zip(vp['bin_centers'], vp['volumes']):
                if vol <= 0:
                    continue
                bar_span = max_bar_span * (vol / max_vol)
                x_start = x_end - bar_span
                fig.add_shape(
                    type='rect',
                    x0=x_start, x1=x_end,
                    y0=center - bar_width / 2, y1=center + bar_width / 2,
                    fillcolor='rgba(66,165,245,0.15)',
                    line=dict(width=0),
                    row=1, col=1,
                )
            # POC line
            fig.add_hline(y=vp['poc'], line=dict(color='#42a5f5', width=1, dash='dot'),
                          annotation_text=f"POC ${vp['poc']:.0f}",
                          annotation=dict(font_size=10, font_color='#42a5f5',
                                          xanchor='left', x=0.01), row=1, col=1)

    # Entry/Exit markers
    if show_entries:
        try:
            from backtest import _get_strategy_fn
            sig_fn, sig_col, defaults = _get_strategy_fn(chart_strategy)
            df_signals = sig_fn(df_full.copy(), **defaults)
            if sig_col == 'ma_signal' and 'ma_signal' in df_signals.columns:
                df_signals['strategy_signal'] = df_signals['ma_signal']
                sig_col = 'strategy_signal'
            df_sig_display = filter_df(df_signals, range_start, range_end)
            buys = df_sig_display[df_sig_display[sig_col] == 1]
            sells = df_sig_display[df_sig_display[sig_col] == -1]
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=buys.index, y=buys['Low'] * 0.995, mode='markers', name='BUY',
                    marker=dict(symbol='triangle-up', color='#00e676', size=12, line=dict(width=1, color='white')),
                ), row=1, col=1)
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=sells.index, y=sells['High'] * 1.005, mode='markers', name='SELL',
                    marker=dict(symbol='triangle-down', color='#ff1744', size=12, line=dict(width=1, color='white')),
                ), row=1, col=1)
        except Exception:
            pass

    # Volume subplot
    if 'Volume' in df_display.columns:
        vol_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_display['Close'], df_display['Open'])]
        fig.add_trace(go.Bar(x=df_display.index, y=df_display['Volume'],
                              marker_color=vol_colors, name='Volume', showlegend=False), row=2, col=1)

    # RSI subplot
    if 'RSI' in df_display.columns:
        fig.add_trace(go.Scatter(x=df_display.index, y=df_display['RSI'], name='RSI',
                                  line=dict(color='#9c27b0', width=1.5)), row=3, col=1)
        for level, color, label in [(30, '#26a69a', 'Oversold'), (70, '#ef5350', 'Overbought')]:
            fig.add_hline(y=level, line=dict(color=color, width=1, dash='dash'),
                          row=3, col=1)

    # Z-Score subplot
    if show_zscore:
        zs = compute_zscore(df_display)
        fig.add_trace(go.Scatter(x=df_display.index, y=zs, name='Z-Score',
                                  line=dict(color='#29b6f6', width=1.5)), row=4, col=1)
        fig.add_hline(y=-2, line=dict(color='#26a69a', width=1, dash='dash'), row=4, col=1)
        fig.add_hline(y=2, line=dict(color='#ef5350', width=1, dash='dash'), row=4, col=1)
        fig.add_hline(y=0, line=dict(color='#666', width=0.5), row=4, col=1)

    fig.update_layout(
        height=800 + (120 if show_zscore else 0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=11)),
        margin=dict(t=50, b=20, l=60, r=50),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        font=dict(color='#fafafa', size=12),
    )
    fig.update_xaxes(gridcolor='#1e2130', showgrid=True)
    fig.update_yaxes(gridcolor='#1e2130', showgrid=True)

    # Price axis: clean tick format, controlled density
    fig.update_yaxes(tickprefix='$', tickformat='.0f', nticks=8,
                     title_text='', row=1, col=1)
    # Volume axis: abbreviated format
    fig.update_yaxes(tickformat='.2s', nticks=4,
                     title_text='', row=2, col=1)
    # RSI axis: fixed 0-100 range, few ticks
    fig.update_yaxes(range=[0, 100], nticks=5, tickvals=[0, 30, 50, 70, 100],
                     title_text='', row=3, col=1)
    # Z-Score axis (if visible)
    if show_zscore:
        fig.update_yaxes(nticks=5, title_text='', row=4, col=1)

    st.plotly_chart(fig, width='stretch')

    # ── Volume Profile Key Levels ─────────────────────────────────────────
    if show_vol_profile:
        vp = compute_volume_profile(df_display)
        vp1, vp2, vp3 = st.columns(3)
        vp1.metric('Point of Control', f"${vp['poc']:.2f}")
        vp2.metric('Value Area Low', f"${vp['val']:.2f}")
        vp3.metric('Value Area High', f"${vp['vah']:.2f}")

    # ── Raw data ───────────────────────────────────────────────────────────
    with st.expander('Raw data (last 30 rows)'):
        cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume',
                             'RSI', 'BB_upper', 'BB_mid', 'BB_lower',
                             f'SMA_{MA_SHORT}', f'SMA_{MA_LONG}'] if c in df_display.columns]
        st.dataframe(df_display[cols].tail(30).round(2))


# ===========================================================================
# PAGE 3: Backtest
# ===========================================================================
elif page == '🔁 Backtest':
    st.title('🔁 Backtest')

    from backtest import backtest_strategy

    STRATEGIES = ['MA Crossover', 'MACD + RSI', 'BB Squeeze', 'TSMOM', 'Turtle', 'Fractal', 'Ensemble']
    STRATEGY_DESC = {
        'MA Crossover': f'SMA({MA_SHORT}/{MA_LONG}) golden/death cross  ·  classic trend-following',
        'MACD + RSI':   'MACD crossover confirmed by RSI < 45 entry  ·  documented 73-86% win rate',
        'BB Squeeze':   'Bollinger Band squeeze breakout above upper band  ·  volatility contraction → expansion',
        'TSMOM':        'Time-Series Momentum (AQR/Man AHL style)  ·  12-month trailing return direction',
        'Turtle':       'Donchian 55-day breakout (System 2)  ·  Richard Dennis turtle rules',
        'Fractal':      'Multi-TF Williams fractals + fractal dimension regime filter  ·  institutional structure',
        'Ensemble':     'Signal voting across all 6 strategies  ·  buy when 3+ agree, higher conviction',
    }

    # ── Session state init ─────────────────────────────────────────────────
    if 'bt_period' not in st.session_state:
        st.session_state.bt_period = '1Y'
    if 'bt_custom' not in st.session_state:
        st.session_state.bt_custom = False
    if 'bt_strategy' not in st.session_state:
        st.session_state.bt_strategy = 'MA Crossover'

    # ── Strategy selector ──────────────────────────────────────────────────
    st.markdown('**Strategy**')
    strat_cols = st.columns(len(STRATEGIES))
    for i, s in enumerate(STRATEGIES):
        label = f'[ {s} ]' if st.session_state.bt_strategy == s else s
        if strat_cols[i].button(label, key=f'strat_{i}'):
            st.session_state.bt_strategy = s

    selected_strategy = st.session_state.bt_strategy
    st.caption(STRATEGY_DESC[selected_strategy])
    st.markdown('---')

    # ── Inputs ────────────────────────────────────────────────────────────
    inp1, inp2, inp3 = st.columns([3, 1, 1])
    with inp1:
        bt_tickers_input = st.text_input('Tickers (comma-separated)', value='SPY, AAPL, NVDA')
    with inp2:
        initial_capital = st.number_input('Capital ($)', min_value=1000,
                                          max_value=10_000_000, value=10_000, step=1000)
    with inp3:
        compare_all = st.selectbox('Mode', ['Single strategy', 'Compare all'])

    # ── Advanced settings ─────────────────────────────────────────────────
    with st.expander('Advanced: Transaction Costs & Position Sizing'):
        adv1, adv2, adv3 = st.columns(3)
        with adv1:
            commission_bps = st.number_input('Commission (bps per trade)', min_value=0,
                                              max_value=100, value=0, step=1)
        with adv2:
            pos_mode = st.selectbox('Position Sizing', ['Full Capital', 'Fixed 10%', 'Volatility-Adjusted'])
        with adv3:
            risk_pct_input = st.number_input('Risk % per trade', min_value=1, max_value=100,
                                              value=10, step=1)
        pos_mode_map = {'Full Capital': 'full', 'Fixed 10%': 'fixed_frac', 'Volatility-Adjusted': 'volatility'}
        bt_pos_mode = pos_mode_map[pos_mode]
        bt_commission = commission_bps / 10000.0
        bt_risk_pct = risk_pct_input / 100.0

    # ── Period quick-select ────────────────────────────────────────────────
    BT_PERIODS = ['6M', 'YTD', '1Y', '2Y', '3Y', '5Y']
    st.markdown('**Date range**')
    bp_cols = st.columns(len(BT_PERIODS) + 1)
    for i, p in enumerate(BT_PERIODS):
        label = f'[ {p} ]' if (st.session_state.bt_period == p and not st.session_state.bt_custom) else p
        if bp_cols[i].button(label, key=f'bt_{p}'):
            st.session_state.bt_period = p
            st.session_state.bt_custom = False
    if bp_cols[-1].button('Custom', key='bt_custom_btn'):
        st.session_state.bt_custom = True

    bt_start, bt_end = None, None
    if st.session_state.bt_custom:
        today = date.today()
        d1, d2 = st.columns(2)
        bt_start = d1.date_input('From', value=date(today.year - 1, today.month, today.day))
        bt_end   = d2.date_input('To',   value=today)
    else:
        today = date.today()
        p = st.session_state.bt_period
        if p == '6M':
            bt_start = (pd.Timestamp(today) - pd.DateOffset(months=6)).date()
        elif p == 'YTD':
            bt_start = date(today.year, 1, 1)
        elif p == '1Y':
            bt_start = date(today.year - 1, today.month, today.day)
        elif p == '2Y':
            bt_start = date(today.year - 2, today.month, today.day)
        elif p == '3Y':
            bt_start = date(today.year - 3, today.month, today.day)
        elif p == '5Y':
            bt_start = date(today.year - 5, today.month, today.day)
        bt_end = today

    st.caption(f'Range: {bt_start}  →  {bt_end}')

    run_bt = st.button('Run Backtest')

    if run_bt:
        tickers = [t.strip().upper() for t in bt_tickers_input.split(',') if t.strip()]
        if not tickers:
            st.warning('Enter at least one ticker.')
            st.stop()

        strategies_to_run = STRATEGIES if compare_all == 'Compare all' else [selected_strategy]
        all_results = []
        status = st.empty()
        bar    = st.progress(0)
        total  = len(tickers) * len(strategies_to_run)
        step   = 0

        for ticker in tickers:
            for strat in strategies_to_run:
                step += 1
                status.text(f'{strat} on {ticker}… ({step}/{total})')
                bar.progress(step / total)
                try:
                    r = backtest_strategy(
                        ticker,
                        strategy=strat,
                        initial_capital=float(initial_capital),
                        start_date=bt_start,
                        end_date=bt_end,
                        commission_pct=bt_commission,
                        position_mode=bt_pos_mode,
                        risk_pct=bt_risk_pct,
                    )
                    if r:
                        all_results.append(r)
                    else:
                        st.warning(f'**{ticker} / {strat}**: insufficient data.')
                except Exception as e:
                    st.warning(f'**{ticker} / {strat}**: {e}')

        status.empty()
        bar.empty()

        if not all_results:
            st.error('No results. Try a wider date range or different tickers.')
            st.stop()

        # ── Summary table ──────────────────────────────────────────────────
        st.subheader('Summary')
        summary_rows = []
        for r in all_results:
            row = {
                'Ticker':    r.get('ticker', ''),
                'Strategy':  r.get('strategy', selected_strategy),
                'Return':    f"{r['total_return_pct']:+.2f}%",
                'B&H':       f"{r['buy_hold_return_pct']:+.2f}%",
                'Alpha':     f"{r['alpha_pct']:+.2f}%",
                'Sharpe':    f"{r.get('sharpe_ratio', 0):.3f}",
                'Sortino':   f"{r.get('sortino_ratio', 0):.3f}",
                'Calmar':    f"{r.get('calmar_ratio', 0):.3f}",
                'Max DD':    f"{r['max_drawdown_pct']:.2f}%",
                'Profit Factor': r.get('profit_factor', '-'),
                'Trades':    r['num_trades'],
                'Win Rate':  f"{r['win_rate_pct']:.1f}%",
                'Avg Duration': f"{r.get('avg_trade_duration', 0):.0f}d",
                'Final ($)': f"${r['final_capital']:,.2f}",
            }
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)

        # Highlight best alpha row
        st.dataframe(summary_df)

        if compare_all == 'Compare all' and len(all_results) > 1:
            best = max(all_results, key=lambda r: r['alpha_pct'])
            st.success(
                f"Best strategy: **{best.get('strategy')}** on {best.get('ticker')} "
                f"— Alpha {best['alpha_pct']:+.2f}%  |  Win rate {best['win_rate_pct']:.1f}%"
            )

        # ── Equity curve chart ─────────────────────────────────────────────
        st.subheader('Equity Curves vs Buy & Hold')
        eq_fig = go.Figure()
        palette = ['#26a69a', '#ef5350', '#42a5f5', '#ff9800', '#ab47bc',
                   '#66bb6a', '#ec407a', '#29b6f6', '#ffa726', '#8d6e63']

        for i, r in enumerate(all_results):
            c     = palette[i % len(palette)]
            label = f"{r.get('ticker','')} · {r.get('strategy', selected_strategy)}"
            eq_fig.add_trace(go.Scatter(
                x=r['equity_dates'], y=r['equity_values'],
                name=label,
                line=dict(color=c, width=2),
            ))
            # B&H only once per ticker (avoid duplicate lines in compare mode)
            if strategies_to_run.index(r.get('strategy', selected_strategy)) == 0 or compare_all != 'Compare all':
                eq_fig.add_trace(go.Scatter(
                    x=r['equity_dates'], y=r['bh_values'],
                    name=f"{r.get('ticker','')} B&H",
                    line=dict(color=c, width=1, dash='dot'),
                    opacity=0.5,
                ))

        eq_fig.update_layout(
            height=420,
            margin=dict(t=30, b=20, l=0, r=20),
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            font=dict(color='#fafafa'),
            yaxis_title='Portfolio Value ($)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        eq_fig.update_xaxes(gridcolor='#1e2130')
        eq_fig.update_yaxes(gridcolor='#1e2130')
        st.plotly_chart(eq_fig)

        # ── Advanced Metrics ──────────────────────────────────────────────
        from backtest import compute_advanced_metrics, monte_carlo_simulation
        from config import MC_SIMULATIONS

        st.subheader('Advanced Metrics')
        adv_rows = []
        for r in all_results:
            adv = compute_advanced_metrics(r)
            adv_rows.append({
                'Ticker': r.get('ticker', ''),
                'Strategy': r.get('strategy', ''),
                'Expectancy': f"{adv['expectancy']:+.2f}%",
                'Payoff Ratio': f"{adv['payoff_ratio']:.2f}",
                'Recovery Factor': f"{adv['recovery_factor']:.2f}",
                'Max DD Duration': f"{adv['max_dd_duration_days']}d",
                'Half-Kelly %': f"{adv['kelly_fraction']*100:.1f}%",
                'Avg Win': f"{adv['avg_win_pct']:+.2f}%",
                'Avg Loss': f"-{adv['avg_loss_pct']:.2f}%",
            })
        st.dataframe(pd.DataFrame(adv_rows))

        # ── Drawdown Analysis ────────────────────────────────────────────
        st.subheader('Drawdown Analysis')
        dd_fig = go.Figure()
        for i, r in enumerate(all_results):
            eq = np.array(r['equity_values'], dtype=float)
            if len(eq) < 2:
                continue
            peak = np.maximum.accumulate(eq)
            drawdown = (eq - peak) / peak * 100
            c = palette[i % len(palette)]
            dd_fig.add_trace(go.Scatter(
                x=r['equity_dates'], y=drawdown,
                fill='tozeroy', fillcolor=f'rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.2)',
                line=dict(color=c, width=1),
                name=f"{r.get('ticker','')} {r.get('strategy','')}",
            ))
        dd_fig.update_layout(
            height=300, margin=dict(t=30, b=20, l=0, r=20),
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            font=dict(color='#fafafa'), yaxis_title='Drawdown (%)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        dd_fig.update_xaxes(gridcolor='#1e2130')
        dd_fig.update_yaxes(gridcolor='#1e2130')
        st.plotly_chart(dd_fig, width='stretch')

        # ── Trade P&L Distribution + Monthly Returns ─────────────────────
        dist_col, heat_col = st.columns(2)

        with dist_col:
            st.subheader('Trade P&L Distribution')
            all_pnl = []
            for r in all_results:
                all_pnl.extend([t['pnl_pct'] for t in r['trades'] if t['type'] == 'sell'])
            if all_pnl:
                colors = ['#26a69a' if p > 0 else '#ef5350' for p in all_pnl]
                hist_fig = go.Figure(go.Histogram(x=all_pnl, nbinsx=25, marker_color='#42a5f5'))
                hist_fig.add_vline(x=0, line=dict(color='white', width=1, dash='dash'))
                hist_fig.update_layout(
                    height=300, margin=dict(t=30, b=20, l=0, r=20),
                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                    font=dict(color='#fafafa'), xaxis_title='P&L (%)', yaxis_title='Count',
                )
                hist_fig.update_xaxes(gridcolor='#1e2130')
                hist_fig.update_yaxes(gridcolor='#1e2130')
                st.plotly_chart(hist_fig, width='stretch')
            else:
                st.info('No completed trades to display.')

        with heat_col:
            st.subheader('Monthly Returns')
            # Use first result for heatmap
            r0 = all_results[0]
            eq_s = pd.Series(r0['equity_values'], index=pd.DatetimeIndex(r0['equity_dates']))
            monthly = eq_s.resample('ME').last().pct_change().dropna() * 100
            if len(monthly) >= 3:
                hm_data = pd.DataFrame({
                    'Year': monthly.index.year,
                    'Month': monthly.index.month,
                    'Return': monthly.values,
                })
                pivot = hm_data.pivot_table(index='Year', columns='Month', values='Return', aggfunc='sum')
                month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                pivot.columns = [month_names[m-1] for m in pivot.columns]
                hm_fig = go.Figure(go.Heatmap(
                    z=pivot.values, x=list(pivot.columns), y=[str(y) for y in pivot.index],
                    colorscale='RdYlGn', zmid=0,
                    text=[[f'{v:.1f}%' if not np.isnan(v) else '' for v in row] for row in pivot.values],
                    texttemplate='%{text}', textfont=dict(size=10),
                ))
                hm_fig.update_layout(
                    height=300, margin=dict(t=30, b=20, l=0, r=20),
                    paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                    font=dict(color='#fafafa'),
                )
                st.plotly_chart(hm_fig, width='stretch')
            else:
                st.info('Need at least 3 months of data for heatmap.')

        # ── Rolling Sharpe ────────────────────────────────────────────────
        with st.expander('Rolling Sharpe Ratio (63-day)'):
            rs_fig = go.Figure()
            for i, r in enumerate(all_results):
                eq = pd.Series(r['equity_values'], index=pd.DatetimeIndex(r['equity_dates']))
                daily_ret = eq.pct_change()
                rolling_sharpe = daily_ret.rolling(63).mean() / daily_ret.rolling(63).std() * np.sqrt(252)
                c = palette[i % len(palette)]
                rs_fig.add_trace(go.Scatter(
                    x=rolling_sharpe.index, y=rolling_sharpe,
                    name=f"{r.get('ticker','')} {r.get('strategy','')}",
                    line=dict(color=c, width=1.5),
                ))
            rs_fig.add_hline(y=0, line=dict(color='#666', width=0.5))
            rs_fig.update_layout(
                height=300, margin=dict(t=30, b=20, l=0, r=20),
                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                font=dict(color='#fafafa'), yaxis_title='Sharpe Ratio',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )
            rs_fig.update_xaxes(gridcolor='#1e2130')
            rs_fig.update_yaxes(gridcolor='#1e2130')
            st.plotly_chart(rs_fig, width='stretch')

        # ── Monte Carlo ───────────────────────────────────────────────────
        with st.expander('Monte Carlo Analysis'):
            for r in all_results:
                if r['num_trades'] < 3:
                    st.info(f"{r.get('ticker','')} {r.get('strategy','')}: Need 3+ trades for Monte Carlo.")
                    continue
                mc = monte_carlo_simulation(r['trades'], n_simulations=MC_SIMULATIONS,
                                             initial_capital=float(initial_capital))
                if mc:
                    st.markdown(f"**{r.get('ticker','')} · {r.get('strategy','')}** ({MC_SIMULATIONS} simulations)")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric('Median Return', f"{mc['median_return']:+.2f}%")
                    mc2.metric('5th-95th Return', f"{mc['p5_return']:+.1f}% to {mc['p95_return']:+.1f}%")
                    mc3.metric('Worst-Case Max DD', f"{mc['p95_max_dd']:.1f}%")
                    mc_fig = go.Figure(go.Histogram(
                        x=mc['final_values'], nbinsx=50, marker_color='#42a5f5',
                    ))
                    mc_fig.add_vline(x=float(initial_capital), line=dict(color='white', width=1, dash='dash'),
                                     annotation_text='Starting Capital')
                    mc_fig.update_layout(
                        height=250, margin=dict(t=30, b=20, l=0, r=20),
                        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                        font=dict(color='#fafafa'), xaxis_title='Final Portfolio Value ($)',
                    )
                    mc_fig.update_xaxes(gridcolor='#1e2130')
                    mc_fig.update_yaxes(gridcolor='#1e2130')
                    st.plotly_chart(mc_fig, width='stretch')

        # ── Walk-Forward Validation + Alpha Scorecard ─────────────────────
        with st.expander('Walk-Forward Validation & Alpha Scorecard'):
            st.caption('Out-of-sample, net of transaction costs: does the edge beat '
                       'buy & hold without being overfit? This is the backtested proof.')
            wf_ticker = st.text_input('Ticker for walk-forward', value=tickers[0], key='wf_ticker')
            wf_strat = st.selectbox('Strategy', STRATEGIES, key='wf_strat',
                                     index=STRATEGIES.index(selected_strategy))
            if st.button('Run Walk-Forward', key='wf_btn'):
                with st.spinner('Running walk-forward analysis (8 folds, net of costs)...'):
                    from walk_forward import build_scorecard
                    card = build_scorecard(wf_ticker, wf_strat)
                    wf = card.get('walk_forward', {})
                if 'error' in wf:
                    st.error(wf['error'])
                else:
                    # Verdict banner — the "is the alpha real?" gate.
                    _cost_bps = card.get('commission_pct', 0.0) * 10_000
                    if card['passed']:
                        st.success(f"✅ {card['verdict']} — beats buy & hold out-of-sample, "
                                   f"net of {_cost_bps:.0f} bps/side costs, without overfitting.")
                    else:
                        st.warning(f"⚠️ {card['verdict']} — "
                                   + "; ".join(card['reasons']))

                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric('OOS Return (net)', f"{card['oos_return']:+.2f}%")
                    sc2.metric('Buy & Hold', f"{card['oos_buy_hold']:+.2f}%")
                    sc3.metric('OOS Alpha', f"{card['oos_alpha']:+.2f}%",
                               delta='beats benchmark' if card['checks']['beats_benchmark'] else 'lags benchmark',
                               delta_color='normal' if card['checks']['beats_benchmark'] else 'inverse')

                    wf1, wf2, wf3 = st.columns(3)
                    wf1.metric('Avg Train Sharpe', f"{wf['avg_train_sharpe']:.3f}")
                    wf2.metric('Avg Test Sharpe', f"{wf['avg_test_sharpe']:.3f}")
                    wf3.metric('OOS Win Rate', f"{wf['oos_win_rate']:.1f}%")

                    overfit = wf['overfit_ratio']
                    if overfit < 1.5:
                        st.success(f"Overfit Ratio: {overfit:.2f}x — Strategy appears robust")
                    elif overfit < 2.5:
                        st.warning(f"Overfit Ratio: {overfit:.2f}x — Moderate overfit risk")
                    else:
                        st.error(f"Overfit Ratio: {overfit:.2f}x — Likely overfit")

                    folds_df = pd.DataFrame(wf['folds'])
                    st.dataframe(folds_df[['fold', 'train_sharpe', 'test_sharpe',
                                           'test_win_rate', 'test_return', 'test_buy_hold',
                                           'test_alpha', 'test_trades']])

        # ── Per-ticker / per-strategy trade logs ───────────────────────────
        with st.expander('Trade Logs'):
            for r in all_results:
                label = f"{r.get('ticker','')}  ·  {r.get('strategy', selected_strategy)}  —  {r['num_trades']} trade(s)"
                st.markdown(f"**{label}**")
                if not r['trades']:
                    st.info('No trades in this range.')
                    continue
                rows = []
                for t in r['trades']:
                    d = t['date'].strftime('%Y-%m-%d') if hasattr(t['date'], 'strftime') else str(t['date'])[:10]
                    rows.append({
                        'Type':   t['type'].upper(),
                        'Date':   d,
                        'Price':  f"${t['price']:.2f}",
                        'Shares': t['shares'],
                        'P&L':    f"{t['pnl_pct']:+.2f}%" if t['type'] == 'sell' else '—',
                    })
                st.dataframe(pd.DataFrame(rows))


# ===========================================================================
# PAGE 4: Fractal & Options Analysis
# ===========================================================================
elif page == '🔬 Fractal & Options':
    st.title('🔬 Fractal & Options Analysis')
    st.caption('Institutional-grade support/resistance + directional bias from options flow & fractal structure')

    from options_fetcher import fetch_expiration_dates, fetch_live_spot
    from strategies.fractal_options import compute_composite_analysis

    # ── Session state ──────────────────────────────────────────────────────
    if 'fo_result' not in st.session_state:
        st.session_state.fo_result = None

    # ── Input row ──────────────────────────────────────────────────────────
    def _fo_mark_expiry_changed():
        # Picking a new expiry should behave like clicking Analyze. The selectbox
        # fires this only on a real change, so we flag it and let the run block
        # below re-fetch that expiry's chain on this same rerun — no more
        # silently showing the previously-analyzed expiry's data.
        st.session_state.fo_expiry_changed = True

    inp1, inp2, inp3 = st.columns([3, 2, 1])
    with inp1:
        fo_ticker = st.text_input('Ticker (stocks, ETFs, or ES=F / NQ=F / YM=F)',
                                   value='SPY', key='fo_ticker_input')
    with inp2:
        expiries = fetch_expiration_dates(fo_ticker) if fo_ticker.strip() else []
        fo_expiry = st.selectbox('Options Expiry', ['Nearest'] + expiries[:10],
                                  key='fo_expiry_select',
                                  on_change=_fo_mark_expiry_changed)
    with inp3:
        st.write('')
        st.write('')
        analyze_btn = st.button('Analyze', key='fo_analyze')

    # ── Load active weights ──────────────────────────────────────────────
    from sheets_logger import (
        is_sheets_available, get_current_weights,
        # NOTE: log_prediction / log_prediction_csv are intentionally NOT
        # imported — the website never writes the forecast ledger (that is
        # automation-only; see the "automation-only" note further below).
        read_predictions, read_predictions_csv,
        read_weight_history, log_weight_change,
    )
    _sheets_ok = is_sheets_available()
    try:
        active_weights = get_current_weights() if _sheets_ok else dict(SIGNAL_WEIGHTS)
    except Exception:
        active_weights = dict(SIGNAL_WEIGHTS)

    # Levels history, cached so the Yellow Box 30s auto-refresh fragment does
    # not hammer the Sheets API; cleared explicitly when a new run logs a row.
    @st.cache_data(ttl=120, show_spinner=False)
    def _read_levels_cached():
        from sheets_logger import read_levels_history, read_levels_history_csv
        df = read_levels_history()
        if df is None or df.empty:
            df = read_levels_history_csv()
        return df

    # ── Run analysis ───────────────────────────────────────────────────────
    # Re-run when the user clicks Analyze OR changes the expiry, so scrolling
    # through expiries refreshes the chain (GEX, walls, max-pain, dealer pin)
    # instead of silently showing the last-analyzed expiry's numbers.
    expiry_changed = st.session_state.pop('fo_expiry_changed', False)
    if (analyze_btn or expiry_changed) and fo_ticker.strip():
        exp_arg = None if fo_expiry == 'Nearest' else fo_expiry
        with st.spinner(f'Analyzing {fo_ticker.upper()} ({fo_expiry})... (options + fractals + GEX)'):
            # Anchor on the freshest available price (pre/post-market aware via
            # the active provider — Schwab, else yfinance prepost). Before the
            # 9:30 ET open the daily bar for today doesn't exist yet, so without
            # this the analysis would silently center on yesterday's settled
            # close and a wild pre-market move would be invisible. fetch_live_spot
            # returns None when no live quote exists, in which case the data
            # layer keeps the daily close and tags spot_source='daily_close'.
            _live_spot = fetch_live_spot(fo_ticker.strip().upper())
            result = compute_composite_analysis(
                fo_ticker.strip().upper(), exp_arg, weights=active_weights,
                spot_override=_live_spot,
            )
        st.session_state.fo_result = result

        # Every analysis run appends its floor/ceiling to the LevelsHistory
        # ledger, so the Yellow Box chart can draw each run's levels as its own
        # line and the migration through the day stays visible (instead of each
        # run silently replacing the previous lines). Best-effort: a logging
        # hiccup must never block the analysis itself.
        if result and 'error' not in result:
            try:
                from sheets_logger import (
                    log_levels_snapshot, log_levels_snapshot_csv,
                )
                _lh_r2 = (result.get('ranges') or {}).get('2sigma', {}) or {}
                _lh_est = result.get('estimated_close')
                _lh_kwargs = dict(
                    ticker=result.get('ticker', fo_ticker.strip().upper()),
                    spot=result.get('spot_price'),
                    floor=result.get('floor'), ceiling=result.get('ceiling'),
                    floor2=_lh_r2.get('floor'), ceiling2=_lh_r2.get('ceiling'),
                    est_close=(_lh_est.get('estimated_close')
                               if isinstance(_lh_est, dict) else None),
                    source='dashboard',
                )
                if not (_sheets_ok and log_levels_snapshot(**_lh_kwargs)):
                    log_levels_snapshot_csv(**_lh_kwargs)
                _read_levels_cached.clear()   # new line should appear right away
            except Exception:
                pass

    result = st.session_state.fo_result
    if result is None:
        st.info('Enter a ticker and click **Analyze** to see floor, ceiling, and directional bias with evidence.')
        st.stop()
    if 'error' in result:
        st.error(result['error'])
        st.stop()

    # Data-freshness banner: after the close (or whenever the live chain is
    # unusable) the pipeline serves the last-known-good snapshot. Tell the user
    # plainly so a range built on yesterday's chain is never mistaken for live.
    if result.get('stale'):
        _asof = result.get('as_of') or 'the previous session'
        st.warning(
            f"Showing the **last-known options snapshot** (as of {_asof}). "
            "The live chain is currently unavailable — the market is likely closed."
        )

    # Expiry-substitution banner: when the requested expiry's live chain is
    # unusable AND no cached snapshot exists for it, the data layer serves the
    # most recent available expiry instead. Say so loudly — otherwise the dealer
    # pin / GEX / walls / levels below silently read for a DIFFERENT date than
    # the one selected (e.g. asking for 2026-06-18 after hours and seeing the
    # nearest 2026-06-08 behind the same label).
    if result.get('expiry_substituted') and result.get('requested_expiry'):
        st.warning(
            f"**{result['requested_expiry']} isn't available right now.** Its live "
            "chain is unusable (market likely closed) and no recent "
            f"{result['requested_expiry']} snapshot is cached, so this is the most "
            f"recent available expiry: **{result['expiry']}**. The dealer pin, GEX, "
            f"walls, and levels below are all for {result['expiry']}, not "
            f"{result['requested_expiry']}. Re-run during market hours to pull the "
            f"live {result['requested_expiry']} chain."
        )

    # ══════════════════════════════════════════════════════════════════════
    # SECTION A: Bias Banner + Key Metrics
    # ══════════════════════════════════════════════════════════════════════
    proxy_note = f"  (via {result['resolved_ticker']} options)" if result['proxy_used'] else ""

    # Headline price. The analysis is now re-centered on the live (pre/post-market
    # capable) spot via spot_override, so result['spot_price'] already reflects the
    # live anchor when one was available. We re-fetch a fresh live quote here too
    # so the header keeps updating on reruns, and — crucially — we surface WHICH
    # spot the analysis used so a pre-market move is never hidden behind
    # yesterday's settled close.
    _analyzed_spot = result['spot_price']
    _spot_source = result.get('spot_source', 'daily_close')
    _live_price = _analyzed_spot
    try:
        _q = fetch_live_spot(result['ticker'])
        if _q:
            _live_price = float(_q)
    except Exception:
        pass

    st.markdown(f"### {result['ticker']}{proxy_note}  —  ${_live_price:.2f}")
    _src_label = ("live quote (pre/post-market aware)" if _spot_source == 'live_override'
                  else "last daily close — no live quote available")
    # Surface the ACTUAL option-chain backend for this run. result['source'] is
    # FallbackProvider.last_source['chain'] — 'schwab' or 'yfinance'. If the
    # sidebar says Schwab is configured but this reads 'yfinance', the Schwab
    # chain call fell back (the thing to investigate), so never hide it.
    _chain_src = str(result.get('source', '?'))
    _analysis_note = (f"Expiry: {result['expiry']}  |  Analyzed: {result['timestamp']}"
                      f"  |  Chain via: {_chain_src}"
                      f"  |  Anchored on {_src_label}")
    _price_delta = _live_price - _analyzed_spot
    if abs(_price_delta) > 0.005:
        _analysis_note += f"  |  Since analysis: {'+' if _price_delta > 0 else ''}{_price_delta:.2f}"
    st.caption(_analysis_note)

    # If the analysis fell back to yesterday's close (no live quote available),
    # say so loudly so a wild pre-market / overnight move is never silently
    # ignored. During market or extended hours fetch_live_spot returns a price,
    # the analysis is anchored on it, and this banner does not show.
    if _spot_source != 'live_override':
        st.warning(
            "Heads up: this analysis is anchored on the **last daily close**, not a "
            "live quote — no pre/post-market price was available when you ran it. "
            "Re-run during market or extended hours to capture a live move."
        )

    bias = result['bias']
    conf = result['confidence']
    floor_val = result['floor']
    ceil_val = result['ceiling']
    # Proxy→display price ratio. Defined once here, unconditionally, so the
    # Fractal Structure tab (and Iron Condor levels) never NameError if the
    # `ranges` dict happens to be empty.
    r_ratio = result.get('price_ratio', 1.0) if result['proxy_used'] else 1.0
    banner = (f"**{bias}** — {conf:.0f}% Confidence  |  "
              f"Floor: ${floor_val:.2f}  |  Ceiling: ${ceil_val:.2f}")
    if bias == 'BULLISH':
        st.success(banner)
    elif bias == 'BEARISH':
        st.error(banner)
    else:
        st.warning(banner)

    # ── Why this flag? — the signals behind the directional bias ───────────
    # A SECOND colored "confluence" banner used to render here. It is a
    # separate structural read (compute_confluence) that can point the
    # OPPOSITE way to the weighted-vote bias banner above — which is why SPY
    # showed a green AND a red flag stacked at the top. We keep exactly ONE
    # flag (the weighted bias above) and, instead of a competing banner, break
    # down precisely which indicators hold it up. `confl` is still read: its
    # factor list surfaces in the Fractal Structure tab, and its gamma-pin
    # note is folded in below as a caption (not a contradictory flag).
    confl = result.get('confluence') or {}
    _sigs = result.get('signals') or []
    if _sigs:
        _SIGNAL_LABELS = {
            'options_walls': 'Options walls (OI)',
            'gex_levels': 'Net GEX (dealer gamma)',
            'iv_range': 'IV expected move',
            'fractals': 'Fractal structure (S/R)',
            'vectors': 'Sloped vectors',
            'put_call_ratio': 'Put/Call ratio',
            'iv_skew': 'IV skew',
            'max_pain': 'Max pain',
        }
        # Re-derive the exact weighted tally _compute_bias() used so this
        # explains THIS bias call rather than a fresh recomputation.
        _bw = sum(s['weight'] for s in _sigs if s['bias'] == 'bullish')
        _rw = sum(s['weight'] for s in _sigs if s['bias'] == 'bearish')
        _nw = sum(s['weight'] for s in _sigs if s['bias'] == 'neutral')
        _tot = _bw + _rw + _nw
        _win = bias.lower()
        _win_w = {'bullish': _bw, 'bearish': _rw, 'neutral': _nw}.get(_win, 0.0)
        with st.expander(f"Why {bias}? — the signals behind this flag", expanded=True):
            st.caption(
                "Directional bias is a **weight-weighted vote** of the signals "
                f"below. Bullish weight **{_bw:.2f}** · Bearish **{_rw:.2f}** · "
                f"Neutral **{_nw:.2f}** → **{bias}** wins with **{conf:.0f}%** "
                f"confidence ({_win_w:.2f} of {_tot:.2f} total weight). The ✔ rows "
                "are the indicators currently holding this flag up."
            )
            _icon = {'bullish': '🟢 Bullish', 'bearish': '🔴 Bearish',
                     'neutral': '⚪ Neutral'}
            _rows = []
            for s in sorted(_sigs, key=lambda x: x['weight'], reverse=True):
                _rows.append({
                    'Signal': _SIGNAL_LABELS.get(s['name'], s['name']),
                    'Leans': _icon.get(s['bias'], s['bias']),
                    'Weight': round(float(s['weight']), 3),
                    'Drives flag': '✔' if s['bias'] == _win else '',
                    'Evidence': s['evidence'],
                })
            st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
            if confl:
                _aligned = max(confl.get('bull', 0), confl.get('bear', 0))
                _ctot = confl.get('bull', 0) + confl.get('bear', 0)
                _pin = ('sticky long-gamma → pin likely' if confl.get('pin')
                        else 'slippery short-gamma → no pin')
                st.caption(
                    f"Structure confluence (separate read): **{_aligned} of {_ctot}** "
                    f"signals aligned ({confl.get('label', 'low').upper()}) · {_pin}."
                )

    # Key metrics row
    iv_range = result['iv_range']
    vrp = result.get('vrp', {})
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric('Floor (1-sigma)', f"${floor_val:.2f}")
    k2.metric('Ceiling (1-sigma)', f"${ceil_val:.2f}")
    k3.metric('ATM IV', f"{iv_range.get('iv_used', 0)*100:.1f}%")
    k4.metric('Parkinson RV', f"{vrp.get('rv_parkinson', 0)*100:.1f}%" if vrp.get('rv_parkinson') else 'N/A')
    k5.metric('VRP Adj', f"IV overstates {vrp.get('vrp_pct', 0):.0f}%")
    k6.metric('Market Regime', result['market_regime'].title())

    # ── Estimated Close (dealer pin) — item 4 ──────────────────────────────
    # Where dealers are incentivized to settle price into expiry so the most
    # open interest expires out-of-the-money (the "safest outcome" for the
    # dealers short those options), anchored on max-pain + the gamma magnet,
    # shaped by fractal structure and bounded by the 1σ expected move.
    est = result.get('estimated_close')
    if est and est.get('estimated_close') is not None:
        st.subheader(f"Estimated Close — {result['expiry']} (dealer pin)")
        _dir = est.get('direction', 'flat')
        _arrow = '▲' if _dir == 'up' else ('▼' if _dir == 'down' else '◆')
        p1, p2, p3, p4 = st.columns(4)
        p1.metric(
            'Est. Close',
            f"${est['estimated_close']:.2f}",
            delta=f"{_arrow} {est['drift_from_spot']:+.2f} ({est['drift_pct']:+.2f}%)",
            delta_color='off',
        )
        p2.metric('Likely Range', f"${est['estimate_low']:.2f} — ${est['estimate_high']:.2f}")
        p3.metric(
            'Pin Target', f"${est['pin_target']:.2f}",
            delta=f"Max pain ${est['max_pain']:.2f}", delta_color='off',
        )
        p4.metric(
            'Pin Confidence', f"{est['confidence']:.0f}%",
            delta=est.get('gamma_regime', ''), delta_color='off',
        )
        _gp = est.get('gamma_pin_strike')
        st.caption(
            "Where dealers are incentivized to settle so the most open interest "
            f"expires out-of-the-money. Anchored on max-pain (${est['max_pain']:.2f})"
            + (f" + gamma magnet (${_gp:.2f})" if _gp is not None else "")
            + f", pulled {est['pull_fraction']*100:.0f}% from spot and bracketed by "
            "fractal structure and the 1σ expected move. "
            f"Horizon: {est.get('expiry_dte', '?')} DTE. "
            "Strongest in positive-gamma (sticky) regimes; in negative-gamma "
            "(slippery) regimes the pin is weak and price drifts."
        )

    # Iron Condor Range Levels
    ranges = result.get('ranges', {})
    if ranges:
        st.subheader('Iron Condor Range Levels')
        st.caption('Choose your confidence level for short strikes')
        rc1, rc2, rc3 = st.columns(3)
        for col, (key, label) in zip(
            [rc1, rc2, rc3],
            [('1sigma', '1-Sigma (~68%)'), ('1_5sigma', '1.5-Sigma (~87%)'), ('2sigma', '2-Sigma (~95%)')],
        ):
            r = ranges.get(key, {})
            with col:
                st.metric(
                    label,
                    f"${r.get('floor', 0):.2f} — ${r.get('ceiling', 0):.2f}",
                    delta=f"± ${r.get('move', 0):.2f}",
                    delta_color='off',
                )

    # Methodology transparency
    meth = result.get('range_methodology', {})
    vix_ts = result.get('vix_term_structure', {})
    with st.expander('Range Methodology (how floor/ceiling is computed)'):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Base Move (raw IV)', f"± ${meth.get('base_move', 0):.2f}")
        m2.metric('VRP Factor', f"{meth.get('vrp_factor', 0):.3f}")
        m3.metric('Regime Scale', f"{meth.get('total_regime', 0):.3f}")
        m4.metric('Final 1σ Move', f"± ${meth.get('final_move', 0):.2f}")
        st.caption(
            f"VIX Term Structure: {vix_ts.get('structure', 'unknown').title()} "
            f"(VIX={vix_ts.get('vix_spot', 'N/A')}, VIX3M={vix_ts.get('vix_3m', 'N/A')}) | "
            f"Max Pain (info only): ${result['max_pain']:.2f}"
        )

    # VIX regime row. (The old "Ensemble Consensus" metric that lived here was
    # removed: the bias flag at the top of the page already carries the
    # directional verdict, and the metric cost an extra 1-year fetch plus a
    # six-strategy run on every render just to restate it.)
    try:
        from strategies.vix_filter import fetch_vix, classify_vix_regime
        vix_s = fetch_vix(period='1mo')
        if not vix_s.empty:
            current_vix = float(vix_s.iloc[-1])
            vix_regime = classify_vix_regime(current_vix)
        else:
            current_vix = None
            vix_regime = 'unknown'
    except Exception:
        current_vix = None
        vix_regime = 'unknown'

    v1, v2 = st.columns(2)
    if current_vix is not None:
        v1.metric('VIX', f"{current_vix:.1f}", delta=vix_regime.replace('_', ' ').title())
    else:
        v1.metric('VIX', 'N/A')
    v2.metric('VIX Regime Signal',
              'Full Size' if vix_regime in ('low_vol', 'normal')
              else 'Half Size' if vix_regime == 'elevated'
              else 'Cash Only' if vix_regime == 'crisis' else 'Unknown')

    # ── Prediction logging is automation-only (intentionally NOT here) ────
    # The master forecast ledger (Predictions / Outcomes, and the pre-open
    # PinForecasts / PinOutcomes) is written *exclusively* by the scheduled
    # cloud jobs — daily_record.py (post-close) and record_preopen_pin.py
    # (pre-open) — so the scored track record reflects fixed, point-in-time
    # forecasts on a clean cadence. Clicking **Analyze** on the website is an
    # ad-hoc, interactive look and must NEVER write to that ledger: a manual
    # run (especially after the close, when the chain has rolled to the next
    # expiry and the dealer-pin collapses toward spot) would shadow the
    # automated row for that (date, ticker) — read_predictions keeps only the
    # latest write — and silently corrupt the accuracy stats. Do not re-add a
    # log_prediction() call in this code path.

    st.markdown('---')

    # ══════════════════════════════════════════════════════════════════════
    # Detailed analysis — grouped into tabs so the page is navigable instead
    # of one long scroll. Streamlit executes every tab body each rerun, so
    # side effects (prediction logging above) are unaffected by this grouping.
    # ══════════════════════════════════════════════════════════════════════
    tab_yellowbox, tab_struct, tab_flow, tab_evidence, tab_track = st.tabs([
        '🟨 Yellow Box',
        '🧬 Fractal Structure',
        '⚙️ Options Flow',
        '🧾 Evidence & Accuracy',
        '📈 Track Record',
    ])

    # ──────────────────────────────────────────────────────────────────────
    # TAB 0 — Yellow Box: Milk-RCG-style value box + laddered objectives.
    # A horizontal "where is price in the auction" read that repaints on every
    # Analyze (it reads the same recomputed `result`). Everything is drawn on
    # the proxy price axis (the axis price_df lives on); target/display-unit
    # levels are divided back to that axis via _am_ax() so SPY (ratio 1) and
    # ES=F (ratio>1) both render, while labels show the real display price.
    # ──────────────────────────────────────────────────────────────────────
    with tab_yellowbox:
        st.subheader(f"{result['ticker']} — Yellow Box")
        st.caption(
            "Milk-style value/objective map. The **yellow box** is the expected "
            "(value) zone price holds most of the session; **objectives** are the "
            "laddered targets above and below. Above the box sellers distribute "
            "(unload inventory); below it buyers accumulate (value add). Pick a "
            "range — **1D** draws a Robinhood-style price line (1-minute, incl. "
            "pre/post-market) whose tip tracks the live quote and auto-refreshes "
            "(~30s) while the market is open. Levels rebuild each time you press "
            "**Analyze**."
        )

        # The live session state is re-checked INSIDE the fragment each cycle
        # (not captured once here), so a page opened pre-open begins refreshing
        # the instant the 9:30 ET open arrives — see _yb_render below.

        # Robinhood-style presets. Intraday ranges (1D/1W) pull fresh Schwab
        # minute bars (yfinance only if Schwab is down); 1M/3M/1Y reuse the daily
        # series already fetched (also Schwab-sourced) in the result dict.
        #   win_days — calendar-day lookback the view is clipped to, so neither a
        #     provider over-fetch nor a weekend fallback can widen the x-axis.
        #   widen_period — 1D only: if the requested 1-day minute window is empty
        #     (weekend / pre-open), refetch this wider window and clip back to the
        #     last completed session, so 1D still shows a real intraday chart.
        _YB_RANGES = {
            '1D': dict(period='1d',  interval='1m',  intraday=True,  extended=True, win_days=1,   tail=2,   widen_period='5d'),
            '1W': dict(period='5d',  interval='15m', intraday=True,  win_days=7,   tail=5),
            '1M': dict(period='1mo', interval='1d',  intraday=False, win_days=31,  tail=22),
            '3M': dict(period='',    interval='',    intraday=False, win_days=93,  tail=63),
            '1Y': dict(period='',    interval='',    intraday=False, win_days=366, tail=252),
        }

        @st.fragment(run_every="30s")
        def _yb_render(result=result):
            # Always tick every 30s; re-evaluate the session each cycle so the
            # view auto-starts at the 9:30 ET open, and so the fetch/quote calls
            # only hit the feed while live (cached when closed — see use_cache
            # and the spot re-quote guard below).
            try:
                from options_fetcher import _is_market_hours as _yb_is_open
                market_open = bool(_yb_is_open())
            except Exception:
                market_open = False
            _am_ratio = result.get('price_ratio', 1.0) if result.get('proxy_used') else 1.0
            _am_sym = result.get('proxy_ticker') or result.get('ticker')

            _c0, _c1 = st.columns([3, 2])
            with _c0:
                _yb_range = st.segmented_control(
                    'Range', list(_YB_RANGES.keys()), default='1D',
                    key='yb_range', label_visibility='collapsed',
                )
            _yb_range = _yb_range or '1D'
            _spec = _YB_RANGES[_yb_range]

            # Candle source: Schwab-primary intraday for short ranges, daily
            # slice for long ranges. Always degrade to the daily series so the
            # chart never blanks (weekend / holiday / pre-open / feed outage).
            # `_is_intraday_data` stays False until a real intraday fetch lands,
            # so the overnight x-axis break is only applied to true intraday bars
            # (a daily fallback timestamped at midnight would otherwise be hidden
            # entirely by that hourly break). `_eh_active` records whether the
            # bars include pre/post-market.
            _amdf, _src_note, _is_intraday_data, _eh_active = None, '', False, False
            if _spec['intraday']:
                _want_eh = bool(_spec.get('extended')) and not result.get('proxy_used')
                try:
                    from data_fetcher import fetch_stock_data
                    _d = None
                    if _want_eh:
                        # 1D Robinhood view: pull 1-minute bars INCLUDING pre- and
                        # post-market directly from yfinance. prepost=True is a
                        # yfinance capability not exposed through the provider
                        # interface, so we call it here and fall back to the
                        # standard (RTH-only) provider path if it yields nothing.
                        try:
                            import yfinance as _yf
                            _eh = _yf.Ticker(_am_sym).history(
                                period=_spec['period'], interval=_spec['interval'],
                                prepost=True,
                            )
                            if _eh is not None and not _eh.empty:
                                if isinstance(_eh.index, pd.DatetimeIndex) and _eh.index.tz is not None:
                                    _eh.index = _eh.index.tz_localize(None)
                                _eh.columns = [str(c).strip().title() for c in _eh.columns]
                                _d, _eh_active = _eh, True
                        except Exception:
                            _d, _eh_active = None, False
                    if _d is None or _d.empty:
                        # Fresh pull while the session is live; cached when closed
                        # so the always-on 30s fragment doesn't hammer the feed.
                        _d = fetch_stock_data(
                            _am_sym, period=_spec['period'],
                            interval=_spec['interval'], use_cache=(not market_open),
                        )
                        _eh_active = False
                    # Weekend / pre-open: the 1-day minute window has no bars yet.
                    # Refetch a wider window so we can still show the most recent
                    # completed session (clipped back below) rather than blanking.
                    if (_d is None or _d.empty) and _spec.get('widen_period'):
                        _d = fetch_stock_data(
                            _am_sym, period=_spec['widen_period'],
                            interval=_spec['interval'], use_cache=(not market_open),
                        )
                        _eh_active = False
                    if _d is not None and not _d.empty:
                        _amdf = _d[['Open', 'High', 'Low', 'Close']].dropna()
                        _is_intraday_data = not _amdf.empty
                        if _eh_active and _is_intraday_data:
                            _src_note = ' · incl. pre/post'
                except Exception:
                    _amdf, _is_intraday_data, _eh_active = None, False, False
            if _amdf is None or _amdf.empty:
                _amdf = result['price_df'].tail(_spec['tail'] or 90)
                _is_intraday_data, _eh_active = False, False
                if _spec['intraday']:
                    _src_note = ' · intraday unavailable — showing daily'

            # Clip to the preset's lookback window. This is what fixes the
            # "1M shows years of history back to ~2000" bug: schwab-py's daily
            # endpoint forces period=TWENTY_YEARS (ignoring our start_datetime),
            # and any daily fallback can also carry far more than the selected
            # range — so bound the view by date here. For 1D, isolate just the
            # latest completed session (today during RTH, else the last trading
            # day captured by the widened fetch).
            if isinstance(_amdf.index, pd.DatetimeIndex) and not _amdf.empty:
                _last_ts = _amdf.index.max()
                if _yb_range == '1D':
                    _cutoff = _last_ts.normalize()
                else:
                    _cutoff = _last_ts.normalize() - pd.Timedelta(days=_spec['win_days'] - 1)
                _clipped = _amdf[_amdf.index >= _cutoff]
                if not _clipped.empty:
                    _amdf = _clipped

            with _c1:
                if market_open:
                    _live = ('🟢 live · ~30s refresh' if _spec['intraday']
                             else '🟢 market open')
                else:
                    _live = '⚪ market closed'
                st.caption(f"**{_yb_range}**{_src_note} · {_live}")

            def _am_ax(v):
                """Convert a display/target-unit price onto the proxy plotting axis."""
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    return None
                return v / _am_ratio if _am_ratio > 1 else v

            def _am_fmt(v):
                return f"${v:,.2f}" if isinstance(v, (int, float)) else "n/a"

            # ── Levels (display/target units unless noted) ─────────────────
            _am_floor1, _am_ceil1 = result.get('floor'), result.get('ceiling')
            _am_r2 = (result.get('ranges') or {}).get('2sigma', {}) or {}
            _am_floor2, _am_ceil2 = _am_r2.get('floor'), _am_r2.get('ceiling')

            _am_est = result.get('estimated_close') or {}
            _am_pivot_disp = _am_est.get('estimated_close')
            if _am_pivot_disp is not None:
                _am_pivot_ax = _am_ax(_am_pivot_disp)
            else:
                _am_pivot_disp = result.get('max_pain')   # proxy units (ax == disp at ratio 1)
                _am_pivot_ax = _am_pivot_disp

            # Live spot so the marker + Robinhood line tip track price during the
            # ~30s auto-refresh. Only re-quote while the session is live (the
            # always-on fragment must not hammer the quote feed overnight); when
            # closed we keep the analysis-time spot. Quote the proxy/axis symbol
            # so the value is already in plotting-axis units; derive the display
            # price via the ratio.
            _am_spot_ax = result.get('proxy_spot')        # already proxy axis
            _am_spot_disp = result.get('spot_price')
            if market_open:
                try:
                    from options_fetcher import fetch_live_spot as _yb_live_spot
                    _q = _yb_live_spot(_am_sym)
                    if _q and float(_q) > 0:
                        _am_spot_ax = float(_q)
                        _am_spot_disp = float(_q) * _am_ratio if _am_ratio > 1 else float(_q)
                except Exception:
                    pass

            _am_walls = result.get('options_walls') or {}
            _am_cwall = _am_walls.get('strongest_call_wall')   # proxy units
            _am_pwall = _am_walls.get('strongest_put_wall')    # proxy units

            # Axis-space copies of the target-unit levels
            _am_floor1_ax, _am_ceil1_ax = _am_ax(_am_floor1), _am_ax(_am_ceil1)
            _am_floor2_ax, _am_ceil2_ax = _am_ax(_am_floor2), _am_ax(_am_ceil2)

            fig_am = go.Figure()

            # ── Wall-in-band test + y-range, computed up front so the Robinhood
            #    gradient fill can anchor to a fixed baseline at the chart floor.
            def _am_in_band(v):
                lo = _am_floor2_ax if _am_floor2_ax is not None else _am_floor1_ax
                hi = _am_ceil2_ax if _am_ceil2_ax is not None else _am_ceil1_ax
                return (v is not None and lo is not None and hi is not None
                        and lo * 0.97 <= v <= hi * 1.03)

            _am_close = _amdf['Close'].astype(float)
            _am_extra = [w for w in (_am_cwall, _am_pwall) if _am_in_band(w)]
            _am_ys = [v for v in [_am_floor2_ax, _am_floor1_ax, _am_ceil1_ax, _am_ceil2_ax,
                                  _am_spot_ax, _am_pivot_ax, *_am_extra,
                                  float(_am_close.min()), float(_am_close.max())]
                      if v is not None]
            _am_lo, _am_hi = (min(_am_ys), max(_am_ys))
            _am_pad = (_am_hi - _am_lo) * 0.06 or 1.0
            _yb_bottom, _yb_top = _am_lo - _am_pad, _am_hi + _am_pad

            # ── Robinhood-style price line: a single Close line coloured by the
            #    session direction (green up / red down) with a soft vertical
            #    gradient fill, in place of candlesticks. On the live 1D view the
            #    line is extended one bar-width to the latest quote so the tip
            #    tracks price between 1-minute closes (the old candles only
            #    redrew once a minute and looked frozen). ──────────────────────
            _yb_x = list(_amdf.index)
            _yb_y = [float(v) for v in _am_close.values]
            if (market_open and _spec.get('extended') and _am_spot_ax is not None
                    and len(_yb_x) >= 1):
                _yb_x = _yb_x + [_yb_x[-1] + pd.Timedelta(minutes=1)]
                _yb_y = _yb_y + [float(_am_spot_ax)]
            _yb_up = (_yb_y[-1] >= _yb_y[0]) if _yb_y else True
            _yb_line = '#00c805' if _yb_up else '#ff5000'         # Robinhood green / red
            _yb_fill_top = 'rgba(0,200,5,0.18)' if _yb_up else 'rgba(255,80,0,0.18)'
            _yb_fill_bot = 'rgba(0,200,5,0.0)' if _yb_up else 'rgba(255,80,0,0.0)'
            _yb_shape = 'spline' if _spec['intraday'] else 'linear'
            # Invisible baseline at the chart floor; the price line fills down to
            # it so the vertical gradient spans the visible area (fade to clear).
            fig_am.add_trace(go.Scatter(
                x=_yb_x, y=[_yb_bottom] * len(_yb_x), mode='lines',
                line=dict(width=0), hoverinfo='skip', showlegend=False,
            ))
            fig_am.add_trace(go.Scatter(
                x=_yb_x, y=_yb_y, mode='lines', name='Price',
                line=dict(color=_yb_line, width=2.5, shape=_yb_shape,
                          smoothing=(0.3 if _spec['intraday'] else 1.0)),
                fill='tonexty',
                fillgradient=dict(type='vertical',
                                  colorscale=[[0.0, _yb_fill_bot], [1.0, _yb_fill_top]]),
                showlegend=False, hovertemplate='%{y:.2f}<extra></extra>',
            ))

            # YELLOW BOX — the value zone (1σ floor ↔ ceiling)
            if _am_floor1_ax is not None and _am_ceil1_ax is not None:
                fig_am.add_hrect(
                    y0=_am_floor1_ax, y1=_am_ceil1_ax,
                    fillcolor='rgba(255,214,0,0.10)', line_width=0, layer='below',
                    annotation_text='YELLOW BOX · value zone',
                    annotation_position='top left',
                    annotation_font_color='#ffd600', annotation_font_size=12,
                )
            # Upper distribution zone (ceiling → seller objective)
            if (_am_ceil1_ax is not None and _am_ceil2_ax is not None
                    and _am_ceil2_ax > _am_ceil1_ax):
                fig_am.add_hrect(y0=_am_ceil1_ax, y1=_am_ceil2_ax,
                                 fillcolor='rgba(239,83,80,0.06)', line_width=0, layer='below')
            # Lower accumulation zone (buyer objective → floor)
            if (_am_floor1_ax is not None and _am_floor2_ax is not None
                    and _am_floor2_ax < _am_floor1_ax):
                fig_am.add_hrect(y0=_am_floor2_ax, y1=_am_floor1_ax,
                                 fillcolor='rgba(38,166,154,0.06)', line_width=0, layer='below')

            # ── Core horizontal levels ─────────────────────────────────────
            # Lines draw at their exact prices; their right-edge labels are
            # collected and placed AFTER the walls in a collision pass
            # (_am_labels) so two levels landing within pennies of each other
            # (e.g. call wall 750.00 under seller objective 749.94) stack as
            # separate readable tags instead of printing on top of each other.
            _am_labels = []   # (y_axis_value, text, color, font_size)
            if _am_ceil1_ax is not None:
                fig_am.add_hline(y=_am_ceil1_ax, line=dict(color='#ef5350', width=2))
                _am_labels.append((_am_ceil1_ax, f"Ceiling {_am_fmt(_am_ceil1)}", '#ef5350', 11))
            if _am_floor1_ax is not None:
                fig_am.add_hline(y=_am_floor1_ax, line=dict(color='#26a69a', width=2))
                _am_labels.append((_am_floor1_ax, f"Floor {_am_fmt(_am_floor1)}", '#26a69a', 11))
            if _am_ceil2_ax is not None:
                fig_am.add_hline(y=_am_ceil2_ax, line=dict(color='#ef5350', width=1.5, dash='dash'))
                _am_labels.append((_am_ceil2_ax, f"Seller Objective {_am_fmt(_am_ceil2)}", '#ff8a80', 11))
            if _am_floor2_ax is not None:
                fig_am.add_hline(y=_am_floor2_ax, line=dict(color='#26a69a', width=1.5, dash='dash'))
                _am_labels.append((_am_floor2_ax, f"Buyer Objective {_am_fmt(_am_floor2)}", '#80cbc4', 11))
            if _am_pivot_ax is not None:
                fig_am.add_hline(y=_am_pivot_ax, line=dict(color='#b0bec5', width=1.5, dash='dashdot'))
                _am_labels.append((_am_pivot_ax, f"Pivot {_am_fmt(_am_pivot_disp)}", '#b0bec5', 11))

            # ── Floor/ceiling history: one stepped line per analysis run ──
            # Every run (6:25am recorder, 1:16pm recorder, each dashboard
            # Analyze) appends its levels to LevelsHistory; here we draw that
            # path so the floor/ceiling visibly migrate through the day instead
            # of each run replacing the previous lines. The bold hlines above
            # remain the CURRENT levels; these faded steps are where they were.
            # Fully best-effort: a malformed ledger must never blank the chart.
            try:
                _lh = _read_levels_cached() if _spec['intraday'] else None
                if _lh is not None and not _lh.empty and 'ticker' in _lh.columns:
                    _lh = _lh[
                        (_lh['ticker'].astype(str).str.upper()
                         == str(result.get('ticker', '')).upper())
                        & _lh['timestamp'].notna()
                    ].copy()
                    if not _lh.empty:
                        # Ledger stamps are Pacific; the intraday bars plot in
                        # naive Eastern — shift so the steps land on the axis.
                        try:
                            _lh['ts_et'] = (
                                _lh['timestamp']
                                .dt.tz_localize('America/Los_Angeles',
                                                ambiguous='NaT',
                                                nonexistent='NaT')
                                .dt.tz_convert('America/New_York')
                                .dt.tz_localize(None)
                            )
                        except Exception:
                            _lh['ts_et'] = _lh['timestamp']
                        _lh = _lh.dropna(subset=['ts_et']).sort_values('ts_et')
                        # Window selection — CLEAN SLATE each session: only
                        # runs from the visible window are drawn, so every
                        # morning the 1D chart starts empty and fills in as
                        # the day's runs land (6:25am print first). Yesterday's
                        # prints never carry over; the LevelsHistory ledger
                        # still keeps every row (the 1W view and research read
                        # the full history). Runs after the last bar (tonight's
                        # after-hours analyses) pin to the right edge.
                        _x_lo, _x_hi = _amdf.index.min(), _amdf.index.max()
                        _lh = _lh[(_lh['ts_et'] >= _x_lo)
                                  & (_lh['ts_et'] <= _x_hi + pd.Timedelta(hours=6))]
                        _lh['x_pos'] = _lh['ts_et'].clip(lower=_x_lo, upper=_x_hi)
                    if not _lh.empty:
                        # Milk-style: each run's level is its own horizontal
                        # segment anchored AT the run's position on the axis
                        # and extended until the next run supersedes it (the
                        # latest one runs to the chart edge), with a small
                        # price tag boxed at the anchor — so the chart reads
                        # "this floor/ceiling was called at this time," like
                        # the Yellowbox level prints. Labels are thinned by
                        # x-distance: when several runs land at (nearly) the
                        # same axis position — e.g. a burst of after-hours
                        # re-analyses all pinned to the chart edge — only the
                        # LATEST of the cluster keeps its tag, so boxes never
                        # stack. Their segments still draw and carry hovers.
                        _x_end = _amdf.index.max()
                        _min_gap = (_x_hi - _x_lo) * 0.05   # 5% of axis span
                        for _col, _color, _txt_color, _label in (
                            ('floor', 'rgba(38,166,154,0.55)', '#26a69a', 'Floor'),
                            ('ceiling', 'rgba(239,83,80,0.55)', '#ef5350', 'Ceiling'),
                        ):
                            _pts = _lh.dropna(subset=[_col])
                            if _pts.empty:
                                continue
                            _times = list(_pts['x_pos'])
                            _vals = list(_pts[_col])
                            # Pick which runs get a price tag. Floor and
                            # ceiling tag INDEPENDENTLY: a level only prints
                            # when it has moved meaningfully (≥0.05% of price,
                            # min $0.10) since ITS last printed value — so a
                            # run that only relocated the ceiling prints just
                            # a ceiling tag while the floor's line continues
                            # untagged, like Milk's prints, instead of every
                            # run stamping a synchronized floor+ceiling pair
                            # at the same timestamp. Later tags win any spot
                            # on the axis (walk newest→oldest, drop a tag
                            # landing within _min_gap of one already kept).
                            _cand = []
                            _last_tagged = None
                            for _i, _v in enumerate(_vals):
                                _eps = max(0.10, abs(_v) * 0.0005)
                                if _last_tagged is None or abs(_v - _last_tagged) >= _eps:
                                    _cand.append(_i)
                                    _last_tagged = _v
                            _tag_idx, _kept_x = [], []
                            for _i in reversed(_cand):
                                if all(abs(_times[_i] - _kx) >= _min_gap
                                       for _kx in _kept_x):
                                    _tag_idx.append(_i)
                                    _kept_x.append(_times[_i])
                            _xs, _ys = [], []
                            for _i, (_t0, _v) in enumerate(zip(_times, _vals)):
                                _t1 = _times[_i + 1] if _i + 1 < len(_times) else _x_end
                                _y = _am_ax(_v)
                                # Disconnected segments (None gap = no vertical joins)
                                _xs += [_t0, _t1, None]
                                _ys += [_y, _y, None]
                                if _i in _tag_idx:
                                    _at_right = (_x_hi - _t0) < _min_gap
                                    fig_am.add_annotation(
                                        x=_t0, y=_y, xref='x', yref='y',
                                        text=f"{_label} {_v:,.2f}",
                                        showarrow=False,
                                        xanchor='right' if _at_right else 'left',
                                        yanchor='bottom' if _col == 'ceiling' else 'top',
                                        font=dict(color=_txt_color, size=10),
                                        bgcolor='rgba(14,17,23,0.75)',
                                        bordercolor=_color, borderwidth=1,
                                        borderpad=2,
                                    )
                            fig_am.add_trace(go.Scatter(
                                x=_xs, y=_ys, mode='lines',
                                name=f'{_label} @ run', showlegend=False,
                                line=dict(color=_color, width=1.4, dash='dot'),
                                hovertemplate=(_label
                                               + ' %{y:.2f}<br>set %{x|%H:%M} ET'
                                               '<extra></extra>'),
                            ))
            except Exception:
                pass

            # ── Option walls (proxy units) as secondary objective ticks ────
            if _am_in_band(_am_cwall):
                _cw_lbl = _am_cwall * _am_ratio if _am_ratio > 1 else _am_cwall
                fig_am.add_hline(y=_am_cwall, line=dict(color='#ef9a9a', width=1, dash='dot'))
                _am_labels.append((_am_cwall, f"Call wall {_am_fmt(_cw_lbl)}", '#ef9a9a', 10))
            if _am_in_band(_am_pwall):
                _pw_lbl = _am_pwall * _am_ratio if _am_ratio > 1 else _am_pwall
                fig_am.add_hline(y=_am_pwall, line=dict(color='#a5d6a7', width=1, dash='dot'))
                _am_labels.append((_am_pwall, f"Put wall {_am_fmt(_pw_lbl)}", '#a5d6a7', 10))

            # ── Right-edge label collision pass ────────────────────────────
            # Convert each label's price to approximate pixel space; walking
            # bottom-up, any label that would overlap the one below it is
            # nudged up just enough to clear (yshift in px). A second pass
            # clamps the stack under the top of the plot — without it the
            # topmost tags (call wall / seller objective near the range high)
            # get pushed out of the chart area. Lines stay at their exact
            # prices — only the text tags fan apart.
            _plot_px = 640 - 120                       # chart height minus margins
            _ppu = _plot_px / max(_yb_top - _yb_bottom, 1e-9)
            _lbl_h = 14                                # px footprint per tag
            _ordered = sorted(_am_labels, key=lambda t: t[0])
            _pxs, _last_px = [], None
            for _yv, _txt, _col, _fs in _ordered:
                _px = (_yv - _yb_bottom) * _ppu
                if _last_px is not None and _px < _last_px + _lbl_h:
                    _px = _last_px + _lbl_h
                _pxs.append(_px)
                _last_px = _px
            _cap = _plot_px - 6
            for _i in range(len(_pxs) - 1, -1, -1):
                if _pxs[_i] > _cap:
                    _pxs[_i] = _cap
                _cap = _pxs[_i] - _lbl_h
            for (_yv, _txt, _col, _fs), _px in zip(_ordered, _pxs):
                _nat_px = (_yv - _yb_bottom) * _ppu
                fig_am.add_annotation(
                    xref='paper', x=1.0, xanchor='right',
                    y=_yv, yref='y', yanchor='middle', yshift=_px - _nat_px,
                    text=_txt, showarrow=False,
                    font=dict(color=_col, size=_fs),
                    bgcolor='rgba(14,17,23,0.6)',
                )

            # ── Spot marker ────────────────────────────────────────────────
            if _am_spot_ax is not None:
                fig_am.add_hline(y=_am_spot_ax, line=dict(color='#ff9800', width=1.5, dash='dot'))
                fig_am.add_annotation(
                    xref='paper', x=0.5, y=_am_spot_ax, yref='y',
                    text=f"◆ Spot {_am_fmt(_am_spot_disp)}", showarrow=False,
                    font=dict(color='#ff9800', size=12), bgcolor='rgba(14,17,23,0.6)',
                )

            # ── Distribution / accumulation annotations (Milk's wording) ───
            if _am_ceil1_ax is not None:
                _am_uy = (_am_ceil1_ax + _am_ceil2_ax) / 2 if _am_ceil2_ax else _am_ceil1_ax
                fig_am.add_annotation(
                    xref='paper', x=0.03, y=_am_uy, yref='y', xanchor='left',
                    text="Sellers unload inventory on test —<br>resistive first attempts",
                    showarrow=False, align='left',
                    font=dict(color='#ef5350', size=11), bgcolor='rgba(14,17,23,0.55)',
                )
            if _am_floor1_ax is not None:
                _am_ly = (_am_floor1_ax + _am_floor2_ax) / 2 if _am_floor2_ax else _am_floor1_ax
                fig_am.add_annotation(
                    xref='paper', x=0.03, y=_am_ly, yref='y', xanchor='left',
                    text="Buyers value add on test —<br>supportive first attempts",
                    showarrow=False, align='left',
                    font=dict(color='#26a69a', size=11), bgcolor='rgba(14,17,23,0.55)',
                )

            # ── Bias flag ──────────────────────────────────────────────────
            # Lives in the top MARGIN band (above the plot area, next to the
            # title) so it can never overlay the right-edge level tags — when
            # the call wall / seller objective sit near the top of the price
            # range, their tags occupy exactly the top-right corner the flag
            # used to claim.
            _am_bias = result.get('bias', 'NEUTRAL')
            _am_conf = result.get('confidence', 0)
            _am_bcolor = ('#26a69a' if _am_bias == 'BULLISH'
                          else '#ef5350' if _am_bias == 'BEARISH' else '#ff9800')
            _am_barrow = '▲' if _am_bias == 'BULLISH' else '▼' if _am_bias == 'BEARISH' else '◆'
            fig_am.add_annotation(
                xref='paper', x=1.0, y=1.0, yref='paper', xanchor='right', yanchor='bottom',
                yshift=4,
                text=f"{_am_barrow} {_am_bias} · {_am_conf:.0f}%", showarrow=False,
                font=dict(color=_am_bcolor, size=14),
                bgcolor='rgba(14,17,23,0.65)', bordercolor=_am_bcolor, borderwidth=1,
            )

            # (Y-range was computed up front — _yb_bottom/_yb_top — so the
            #  Robinhood gradient fill could anchor to the chart floor; it is
            #  applied with the layout below.)

            # ── X-axis: always hide weekend gaps. Hide the intraday overnight
            #    gap ONLY when we're actually plotting intraday bars (a daily
            #    fallback timestamped at midnight would otherwise be hidden wholly
            #    by an hourly break). With extended-hours bars present, hide only
            #    20:00→04:00 ET so pre/post-market shows; otherwise hide the full
            #    16:00→09:30 non-RTH window so the line stays contiguous.
            _yb_breaks = [dict(bounds=['sat', 'mon'])]
            if _is_intraday_data and not result.get('proxy_used'):
                if _eh_active:
                    _yb_breaks.append(dict(bounds=[20, 4], pattern='hour'))
                else:
                    _yb_breaks.append(dict(bounds=[16, 9.5], pattern='hour'))

            fig_am.update_layout(
                height=640, template='plotly_dark',
                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                font=dict(color='#fafafa'),
                xaxis_rangeslider_visible=False,
                margin=dict(l=55, r=150, t=46, b=24), showlegend=False,
                title=dict(text=f"{result['ticker']} Yellow Box — objectives bracket the value box",
                           font=dict(size=15, color='#fafafa')),
            )
            fig_am.update_xaxes(gridcolor='#1e2130', rangebreaks=_yb_breaks)
            fig_am.update_yaxes(gridcolor='#1e2130', range=[_yb_bottom, _yb_top])
            st.plotly_chart(fig_am, use_container_width=True, key='yb_chart')

            _yb_footer = (
                f"Value box {_am_fmt(_am_floor1)} – {_am_fmt(_am_ceil1)}  ·  "
                f"Seller objective {_am_fmt(_am_ceil2)}  ·  Buyer objective {_am_fmt(_am_floor2)}  ·  "
                f"Pivot {_am_fmt(_am_pivot_disp)}  ·  Spot {_am_fmt(_am_spot_disp)}.  "
                "Break above the box → rotate to the seller objective; break below → "
                "rotate to the buyer objective; rejection → revert to pivot."
            )
            # Escape '$' so Streamlit markdown doesn't treat paired '$…$' as LaTeX
            # math (which renders the amounts in an inconsistent serif/math font).
            st.caption(_yb_footer.replace("$", "\\$"))

        _yb_render()


    # ──────────────────────────────────────────────────────────────────────
    # TAB 1 — Fractal Structure: chart + vectors + neurals + confluence detail
    # ──────────────────────────────────────────────────────────────────────
    with tab_struct:
        st.subheader('Fractal Market Structure')
        st.caption('Confirmed Williams pivots joined into the swing structure '
                   '(HH/HL = uptrend legs, LH/LL = downtrend legs) with '
                   'confirmation-lag-honest break-of-structure flags, sloped '
                   'vectors (flip role on a cross), neural zones drawn as '
                   'bands (scored by repeated bounces), and the 1σ/2σ '
                   'envelope + max-pain.')
        price_df = result['price_df'].tail(120)

        fig_frac = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.03,
            subplot_titles=(
                f"{result['resolved_ticker']} — Fractal Pivots",
                "Fractal Dimension (1.0=trending, 2.0=choppy)",
            ),
        )

        # Candlestick
        fig_frac.add_trace(go.Candlestick(
            x=price_df.index, open=price_df['Open'], high=price_df['High'],
            low=price_df['Low'], close=price_df['Close'], name='Price',
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        ), row=1, col=1)

        # Fractal high markers
        fh = price_df.dropna(subset=['fractal_high'])
        if not fh.empty:
            fig_frac.add_trace(go.Scatter(
                x=fh.index, y=fh['fractal_high'] * 1.003,
                mode='markers', name='Fractal High (Resistance)',
                marker=dict(symbol='triangle-down', color='#ef5350', size=10),
            ), row=1, col=1)

        # Fractal low markers
        fl = price_df.dropna(subset=['fractal_low'])
        if not fl.empty:
            fig_frac.add_trace(go.Scatter(
                x=fl.index, y=fl['fractal_low'] * 0.997,
                mode='markers', name='Fractal Low (Support)',
                marker=dict(symbol='triangle-up', color='#26a69a', size=10),
            ), row=1, col=1)

        # ── Swing structure: the actual way to READ fractals ──────────────
        # A raw pivot scatter hides the story; joining alternating confirmed
        # pivots into a zigzag exposes it: HH+HL sequences = uptrend legs,
        # LH+LL = downtrend legs, mixed = range. Consecutive same-side pivots
        # collapse to the more extreme one (the true swing point). On top of
        # the zigzag: HH/HL/LH/LL tags per swing, a structure-bias badge from
        # the last four tags, and break-of-structure flags where a CLOSE takes
        # out the most recent swing high/low. BOS detection honors the
        # confirmation lag — a Williams pivot is only knowable FRACTAL_PERIOD
        # bars after it prints, so no break is flagged with hindsight data.
        # Best-effort: structure overlays must never blank the chart.
        try:
            from config import FRACTAL_PERIOD as _frac_lag
            _piv = []
            for _ts, _prow in price_df.iterrows():
                if pd.notna(_prow.get('fractal_high')):
                    _piv.append([_ts, float(_prow['fractal_high']), 'H'])
                if pd.notna(_prow.get('fractal_low')):
                    _piv.append([_ts, float(_prow['fractal_low']), 'L'])
            _swings = []
            for _p in _piv:
                if _swings and _swings[-1][2] == _p[2]:
                    if (_p[2] == 'H' and _p[1] >= _swings[-1][1]) or \
                       (_p[2] == 'L' and _p[1] <= _swings[-1][1]):
                        _swings[-1] = _p
                else:
                    _swings.append(_p)

            if len(_swings) >= 2:
                fig_frac.add_trace(go.Scatter(
                    x=[_s[0] for _s in _swings], y=[_s[1] for _s in _swings],
                    mode='lines', name='Swing structure',
                    line=dict(color='rgba(176,190,197,0.45)', width=1.3),
                    hoverinfo='skip',
                ), row=1, col=1)

            # HH/HL/LH/LL vs the previous same-side swing (last 8 to stay legible)
            _last_side = {'H': None, 'L': None}
            _tags = []
            for _ts, _v, _side in _swings:
                _prev = _last_side[_side]
                if _prev is not None:
                    _tag = (('HH' if _v > _prev else 'LH') if _side == 'H'
                            else ('LL' if _v < _prev else 'HL'))
                    _tags.append((_ts, _v, _side, _tag))
                _last_side[_side] = _v
            for _ts, _v, _side, _tag in _tags[-8:]:
                _bull = _tag in ('HH', 'HL')
                fig_frac.add_annotation(
                    x=_ts, y=_v, text=_tag, showarrow=False,
                    yanchor='bottom' if _side == 'H' else 'top',
                    yshift=10 if _side == 'H' else -10,
                    font=dict(size=10,
                              color='#26a69a' if _bull else '#ef5350'),
                    row=1, col=1,
                )

            # Structure-bias badge from the last four swing tags
            _recent = [_t[3] for _t in _tags[-4:]]
            if _recent:
                _nb = sum(1 for _t in _recent if _t in ('HH', 'HL'))
                _ns = sum(1 for _t in _recent if _t in ('LH', 'LL'))
                if _nb >= 3:
                    _stxt, _scol = '▲ Uptrend structure (HH·HL)', '#26a69a'
                elif _ns >= 3:
                    _stxt, _scol = '▼ Downtrend structure (LH·LL)', '#ef5350'
                else:
                    _stxt, _scol = '◆ Mixed / range structure', '#ff9800'
                fig_frac.add_annotation(
                    xref='paper', yref='paper', x=0.01, y=0.99,
                    xanchor='left', yanchor='top', text=_stxt, showarrow=False,
                    font=dict(size=12, color=_scol),
                    bgcolor='rgba(14,17,23,0.7)', bordercolor=_scol,
                    borderwidth=1, borderpad=3,
                )

            # Break of structure: close crossing the most recent ACTIVE swing
            # (a pivot activates FRACTAL_PERIOD bars after its bar prints).
            _pos_swings = []
            _pos_of = {ts: i for i, ts in enumerate(price_df.index)}
            for _ts, _v, _side in _swings:
                if _ts in _pos_of:
                    _pos_swings.append((_pos_of[_ts], _v, _side))
            _closes = price_df['Close'].values
            _events, _act_h, _act_l, _pi = [], None, None, 0
            for _bar in range(len(price_df)):
                while (_pi < len(_pos_swings)
                       and _pos_swings[_pi][0] + _frac_lag <= _bar):
                    if _pos_swings[_pi][2] == 'H':
                        _act_h = _pos_swings[_pi][1]
                    else:
                        _act_l = _pos_swings[_pi][1]
                    _pi += 1
                _c = float(_closes[_bar])
                if _act_h is not None and _c > _act_h:
                    _events.append((price_df.index[_bar], _act_h, True))
                    _act_h = None
                if _act_l is not None and _c < _act_l:
                    _events.append((price_df.index[_bar], _act_l, False))
                    _act_l = None
            for _ts, _lvl, _up in _events[-3:]:
                fig_frac.add_annotation(
                    x=_ts, y=_lvl, text='BOS ▲' if _up else 'BOS ▼',
                    showarrow=True, arrowhead=2, arrowsize=0.8,
                    arrowcolor='#26a69a' if _up else '#ef5350',
                    ax=0, ay=22 if _up else -22,
                    font=dict(size=10,
                              color='#26a69a' if _up else '#ef5350'),
                    row=1, col=1,
                )
        except Exception:
            pass

        # Floor / Ceiling lines at multiple sigma levels
        proxy_floor = floor_val / r_ratio if r_ratio > 1 else floor_val
        proxy_ceil = ceil_val / r_ratio if r_ratio > 1 else ceil_val
        fig_frac.add_hline(y=proxy_floor, line=dict(color='#26a69a', width=2, dash='dash'),
                           row=1, col=1, annotation_text=f"1σ Floor ${floor_val:.2f}",
                           annotation_position='right')
        fig_frac.add_hline(y=proxy_ceil, line=dict(color='#ef5350', width=2, dash='dash'),
                           row=1, col=1, annotation_text=f"1σ Ceiling ${ceil_val:.2f}",
                           annotation_position='right')
        # 2-sigma lines (lighter)
        r2s = ranges.get('2sigma', {})
        if r2s:
            f2 = r2s['floor'] / r_ratio if r_ratio > 1 else r2s['floor']
            c2 = r2s['ceiling'] / r_ratio if r_ratio > 1 else r2s['ceiling']
            fig_frac.add_hline(y=f2, line=dict(color='#26a69a', width=1, dash='dot'),
                               row=1, col=1, annotation_text=f"2σ ${r2s['floor']:.2f}",
                               annotation_position='right')
            fig_frac.add_hline(y=c2, line=dict(color='#ef5350', width=1, dash='dot'),
                               row=1, col=1, annotation_text=f"2σ ${r2s['ceiling']:.2f}",
                               annotation_position='right')

        # Max pain line
        fig_frac.add_hline(y=result['max_pain'], line=dict(color='#ff9800', width=1, dash='dot'),
                           row=1, col=1, annotation_text=f"Max Pain ${result['max_pain']:.0f} (info)",
                           annotation_position='right')

        # Vectors — sloped dynamic support/resistance (Fractal-Exchange style),
        # projected across the visible window on the proxy price axis. A crossed
        # (flipped) vector is drawn dotted.
        _vecs = result.get('vectors') or {}
        _nbars = len(price_df)
        for _vkey, _vcolor in (('support_vector', '#26a69a'), ('resistance_vector', '#ef5350')):
            _v = _vecs.get(_vkey)
            if not _v or _v.get('current_value') is None:
                continue
            _ry = _v['current_value'] / r_ratio if r_ratio > 1 else _v['current_value']
            _slope = _v.get('slope_per_bar', 0.0)        # proxy units per bar
            _ly = _ry - _slope * (_nbars - 1)
            fig_frac.add_trace(go.Scatter(
                x=[price_df.index[0], price_df.index[-1]], y=[_ly, _ry],
                mode='lines', name=f"{_vkey.split('_')[0].title()} Vector ({_v.get('role')})",
                line=dict(color=_vcolor, width=2,
                          dash='dot' if _v.get('crossed') else 'solid'),
            ), row=1, col=1)

        # Neurals — strongest horizontal zones (multiple-bounce levels), drawn
        # as BANDS rather than 1px lines: a level that has bounced price
        # repeatedly is a zone with width (the clustering tolerance), and
        # opacity scales with its bounce strength. Only zones inside the
        # visible price window are drawn so a strong-but-distant level doesn't
        # compress the y-axis.
        from config import NEURAL_TOLERANCE_PCT as _neur_tol
        _neur = result.get('neural_zones') or {}
        _vis_lo, _vis_hi = float(price_df['Low'].min()), float(price_df['High'].max())
        for _zones, _rgb, _zlabel in (
            (_neur.get('support_zones') or [], '38,166,154', 'Neural S'),
            (_neur.get('resistance_zones') or [], '239,83,80', 'Neural R'),
        ):
            for _z in _zones[:3]:
                _yc = _z['center'] / r_ratio if r_ratio > 1 else _z['center']
                if _z.get('strength', 0) < 2 or not (_vis_lo <= _yc <= _vis_hi):
                    continue
                _half = _yc * _neur_tol / 100.0
                _alpha = min(0.08 + 0.04 * float(_z.get('strength', 2)), 0.28)
                fig_frac.add_hrect(
                    y0=_yc - _half, y1=_yc + _half,
                    fillcolor=f'rgba({_rgb},{_alpha:.2f})', line_width=0,
                    layer='below', row=1, col=1,
                    annotation_text=f"{_zlabel} ${_z['center']:.0f} (×{_z['bounces']})",
                    annotation_position='top right',
                    annotation_font_color=f'rgb({_rgb})',
                    annotation_font_size=10,
                )

        # Fractal dimension subplot — the regime dial that tells you HOW MUCH
        # to trust the structure: in trending tape (FD < 1.35) breaks tend to
        # follow through, in choppy tape (FD > 1.65) fractal levels act as
        # fade zones rather than breakout triggers. Shaded regime bands +
        # a live badge with the current reading make that instant to read.
        if 'fractal_dimension' in price_df.columns:
            fig_frac.add_hrect(y0=1.0, y1=1.35, fillcolor='rgba(38,166,154,0.07)',
                               line_width=0, layer='below', row=2, col=1)
            fig_frac.add_hrect(y0=1.65, y1=2.0, fillcolor='rgba(239,83,80,0.07)',
                               line_width=0, layer='below', row=2, col=1)
            fig_frac.add_trace(go.Scatter(
                x=price_df.index, y=price_df['fractal_dimension'],
                name='Fractal Dimension', line=dict(color='#ff9800', width=1.5),
            ), row=2, col=1)
            fig_frac.add_hline(y=1.5, line=dict(color='#666', dash='dot'),
                               row=2, col=1, annotation_text='Random Walk')
            fig_frac.add_hline(y=1.35, line=dict(color='#26a69a', dash='dot'),
                               row=2, col=1, annotation_text='Trending')
            fig_frac.add_hline(y=1.65, line=dict(color='#ef5350', dash='dot'),
                               row=2, col=1, annotation_text='Choppy')
            _fd_series = price_df['fractal_dimension'].dropna()
            if not _fd_series.empty:
                _fd_now = float(_fd_series.iloc[-1])
                _fd_col = ('#26a69a' if _fd_now < 1.35
                           else '#ef5350' if _fd_now > 1.65 else '#ff9800')
                _fd_word = ('trending' if _fd_now < 1.35
                            else 'choppy' if _fd_now > 1.65 else 'transitional')
                fig_frac.add_annotation(
                    x=_fd_series.index[-1], y=_fd_now,
                    text=f'{_fd_now:.2f} · {_fd_word}', showarrow=False,
                    xanchor='right', yanchor='bottom',
                    font=dict(size=11, color=_fd_col),
                    bgcolor='rgba(14,17,23,0.7)',
                    row=2, col=1,
                )

        fig_frac.update_layout(
            height=650, xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            # l gutter keeps y-axis price ticks readable; wider r holds the
            # right-anchored Max Pain / Neural labels (flipped off the y-axis).
            margin=dict(t=50, b=20, l=55, r=140),
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            font=dict(color='#fafafa'),
        )
        fig_frac.update_xaxes(gridcolor='#1e2130')
        fig_frac.update_yaxes(gridcolor='#1e2130')
        st.plotly_chart(fig_frac)

        # ── Vectors / Neurals tables + confluence factor list ──────────────
        st.markdown('**Vectors, Neurals & Confluence**')
        st.caption('Sloped vectors (flip role on a cross) + horizontal neural '
                   'zones (scored by repeated bounces) + options flow. The '
                   'structure-confluence tally these feed is noted under the '
                   'bias flag at the top; the per-factor votes are listed below.')

        _vcol, _ncol = st.columns(2)
        with _vcol:
            st.markdown('**Vectors** (dynamic S/R)')
            _vrows = []
            for _vkey, _vlabel in (('support_vector', 'Support-anchored'),
                                   ('resistance_vector', 'Resistance-anchored')):
                _v = _vecs.get(_vkey)
                if not _v:
                    continue
                _vrows.append({
                    'Vector': _vlabel,
                    'Now @': f"${_v['current_value']:.2f}",
                    'Role': _v['role'] + (' (flipped)' if _v.get('crossed') else ''),
                    'Slope': _v.get('direction', ''),
                })
            if _vrows:
                st.dataframe(pd.DataFrame(_vrows), hide_index=True)
            else:
                st.caption('No active vectors (need two recent same-type pivots).')

        with _ncol:
            st.markdown('**Neurals** (strongest zones)')
            _nrows = []
            for _z in (_neur.get('resistance_zones') or [])[:3]:
                _nrows.append({'Zone': 'Resistance', 'Level': f"${_z['center']:.2f}",
                               'Bounces': _z['bounces'], 'Strength': _z['strength']})
            for _z in (_neur.get('support_zones') or [])[:3]:
                _nrows.append({'Zone': 'Support', 'Level': f"${_z['center']:.2f}",
                               'Bounces': _z['bounces'], 'Strength': _z['strength']})
            if _nrows:
                st.dataframe(pd.DataFrame(_nrows), hide_index=True)
            else:
                st.caption('No scored neural zones yet.')

        if confl.get('factors'):
            with st.expander('Confluence factors (what voted, and which way)'):
                for _f in confl['factors']:
                    _fi = {'bullish': '🟢', 'bearish': '🔴'}.get(_f['direction'], '⚪')
                    st.markdown(f"- {_fi} **{_f['name']}** — {_f['detail']}")

        with st.expander('Fractal Levels (Recent)'):
            fl_data = result['fractal_levels']
            fl_c1, fl_c2 = st.columns(2)
            with fl_c1:
                st.markdown('**Resistance (Fractal Highs)**')
                for dt, price in fl_data.get('resistance_levels', []):
                    d = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)[:10]
                    st.text(f"  {d}  —  ${price:.2f}")
            with fl_c2:
                st.markdown('**Support (Fractal Lows)**')
                for dt, price in fl_data.get('support_levels', []):
                    d = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)[:10]
                    st.text(f"  {d}  —  ${price:.2f}")

    # ──────────────────────────────────────────────────────────────────────
    # TAB 2 — Options Flow: GEX, OI walls, sentiment/skew, raw chain
    # ──────────────────────────────────────────────────────────────────────
    with tab_flow:
        st.subheader('Gamma Exposure (GEX) Profile')
        st.caption('Positive GEX = sticky/mean-reverting (dealers dampen moves)  |  '
                   'Negative GEX = slippery (dealers amplify moves)')
        st.caption(
            'Units: \\$ of dealer hedge notional per \\$1 move. Black-Scholes '
            'gamma × overnight-settled OI × 100 × spot, calls + / puts − '
            '(standard naive dealer convention). For a same-day 0DTE expiry, '
            'gamma uses the actual time remaining to the 4:00pm ET close. '
            'Note: OI settles once per night, so positions opened TODAY are '
            'not in today\'s GEX — true of every OI-based GEX feed.'
        )
        gex_df = result['gex_df']
        if gex_df is not None and not gex_df.empty:
            fig_gex = go.Figure()
            colors = ['#26a69a' if v >= 0 else '#ef5350' for v in gex_df['net_gex']]
            fig_gex.add_trace(go.Bar(
                x=gex_df['strike'], y=gex_df['net_gex'],
                marker_color=colors, name='Net GEX',
            ))
            fig_gex.add_vline(x=result['proxy_spot'],
                              line=dict(color='white', width=2, dash='dash'),
                              annotation_text=f"Spot ${result['proxy_spot']:.2f}")
            fig_gex.update_layout(
                height=350, xaxis_title='Strike Price', yaxis_title='Net Gamma Exposure ($)',
                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                font=dict(color='#fafafa'),
                margin=dict(t=30, b=20, l=0, r=20),
            )
            fig_gex.update_xaxes(gridcolor='#1e2130')
            fig_gex.update_yaxes(gridcolor='#1e2130')
            st.plotly_chart(fig_gex)

            # GEX summary
            total_gex = gex_df['net_gex'].sum()
            max_gex_strike = gex_df.loc[gex_df['net_gex'].abs().idxmax(), 'strike'] if not gex_df.empty else 0
            g1, g2, g3 = st.columns(3)
            g1.metric('Total Net GEX', f"${total_gex:,.0f}")
            g2.metric('GEX Regime', 'Sticky' if total_gex > 0 else 'Slippery')
            g3.metric('Highest GEX Strike', f"${max_gex_strike:.0f}")
        else:
            st.warning('GEX data not available for this ticker.')

        st.markdown('---')

        st.subheader('Open Interest Walls — Support & Resistance')
        st.caption('High OI concentrations at strikes act as magnets and barriers due to dealer hedging')
        walls = result['options_walls']

        oi_col1, oi_col2 = st.columns(2)
        with oi_col1:
            st.markdown('**Call Walls (Resistance)**')
            if walls.get('call_walls'):
                cw_strikes = [w[0] for w in walls['call_walls']]
                cw_oi = [w[1] for w in walls['call_walls']]
                fig_cw = go.Figure(go.Bar(
                    x=cw_strikes, y=cw_oi, marker_color='#ef5350', name='Call OI',
                ))
                fig_cw.add_vline(x=result['proxy_spot'],
                                 line=dict(color='white', width=1, dash='dash'))
                fig_cw.update_layout(
                    height=280, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                    font=dict(color='#fafafa'), margin=dict(t=10, b=20, l=0, r=0),
                    xaxis_title='Strike', yaxis_title='Open Interest',
                )
                fig_cw.update_xaxes(gridcolor='#1e2130')
                fig_cw.update_yaxes(gridcolor='#1e2130')
                st.plotly_chart(fig_cw)
            else:
                st.info('No significant call walls found.')

        with oi_col2:
            st.markdown('**Put Walls (Support)**')
            if walls.get('put_walls'):
                pw_strikes = [w[0] for w in walls['put_walls']]
                pw_oi = [w[1] for w in walls['put_walls']]
                fig_pw = go.Figure(go.Bar(
                    x=pw_strikes, y=pw_oi, marker_color='#26a69a', name='Put OI',
                ))
                fig_pw.add_vline(x=result['proxy_spot'],
                                 line=dict(color='white', width=1, dash='dash'))
                fig_pw.update_layout(
                    height=280, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
                    font=dict(color='#fafafa'), margin=dict(t=10, b=20, l=0, r=0),
                    xaxis_title='Strike', yaxis_title='Open Interest',
                )
                fig_pw.update_xaxes(gridcolor='#1e2130')
                fig_pw.update_yaxes(gridcolor='#1e2130')
                st.plotly_chart(fig_pw)
            else:
                st.info('No significant put walls found.')

        st.markdown('---')

        st.subheader('Sentiment & IV Skew')
        pc = result['put_call_ratios']
        skew_data = result['iv_skew']

        s1, s2, s3, s4 = st.columns(4)
        s1.metric('P/C Ratio (OI)', f"{pc['pc_ratio_oi']:.2f}",
                  delta=pc['oi_bias'].title())
        s2.metric('P/C Ratio (Volume)', f"{pc['pc_ratio_volume']:.2f}",
                  delta=pc['volume_bias'].title())
        s3.metric('IV Skew Ratio', f"{skew_data['skew_ratio']:.2f}",
                  delta=skew_data['skew_bias'].title())
        s4.metric('OTM Put IV / Call IV',
                  f"{skew_data['otm_put_iv']*100:.1f}% / {skew_data['otm_call_iv']*100:.1f}%")

        # Sentiment interpretation
        pc_note = ''
        if pc['oi_bias'] == 'bearish':
            pc_note = 'Elevated put OI suggests institutional hedging or bearish positioning.'
        elif pc['oi_bias'] == 'bullish':
            pc_note = 'Low put/call ratio suggests bullish sentiment — less hedging activity.'
        if skew_data['skew_bias'] == 'bearish':
            pc_note += ' High IV skew indicates institutions are paying up for downside protection.'
        elif skew_data['skew_bias'] == 'bullish':
            pc_note += ' Low IV skew suggests complacency — less demand for downside hedging.'
        if pc_note:
            st.caption(pc_note.strip())

        s5, s6, s7, s8 = st.columns(4)
        s5.metric('Total Call OI', f"{pc['call_oi_total']:,}")
        s6.metric('Total Put OI', f"{pc['put_oi_total']:,}")
        s7.metric('Total Call Volume', f"{pc['call_volume_total']:,}")
        s8.metric('Total Put Volume', f"{pc['put_volume_total']:,}")

        st.markdown('---')

        with st.expander('Raw Options Chain Data'):
            # Gate the network fetch behind a checkbox: the expander body runs on
            # every rerun (incl. the 30s Yellow Box refresh), so an unconditional
            # fetch would re-hit the provider each time. Pull the SAME expiry the
            # analysis used (not the provider's nearest default) so the raw chain
            # matches the levels shown above.
            _raw_expiry = result.get('expiry') or None
            st.caption(f"Expiry: {_raw_expiry or 'nearest available'}")
            if st.checkbox('Load raw chain', key='fo_raw_chain'):
                from options_fetcher import fetch_options_chain as _fetch_chain
                raw_c, raw_p, _ = _fetch_chain(result['ticker'], expiry=_raw_expiry)
                display_cols = ['strike', 'lastPrice', 'bid', 'ask',
                                'volume', 'openInterest', 'impliedVolatility']
                available_cols_c = [c for c in display_cols if c in raw_c.columns]
                available_cols_p = [c for c in display_cols if c in raw_p.columns]

                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown('**Calls**')
                    if available_cols_c:
                        st.dataframe(raw_c[available_cols_c].head(30))
                with rc2:
                    st.markdown('**Puts**')
                    if available_cols_p:
                        st.dataframe(raw_p[available_cols_p].head(30))

    # ──────────────────────────────────────────────────────────────────────
    # TAB 3 — Evidence & Accuracy: VRP, signal evidence, range validation
    # ──────────────────────────────────────────────────────────────────────
    with tab_evidence:
        st.subheader('Variance Risk Premium Analysis')
        st.caption('IV systematically overstates realized vol — this is the #1 edge for iron condor sellers')
        vp1, vp2, vp3, vp4 = st.columns(4)
        vp1.metric('Raw ATM IV', f"{vrp.get('iv', 0)*100:.1f}%")
        vp2.metric('Parkinson RV (20d)', f"{vrp.get('rv_parkinson', 0)*100:.1f}%" if vrp.get('rv_parkinson') else 'N/A')
        vp3.metric('VRP Ratio (RV/IV)', f"{vrp.get('scaling_factor', 0):.3f}")
        vp4.metric('IV Overstatement', f"{vrp.get('vrp_pct', 0):.1f}%")

        st.markdown('---')

        st.subheader('Evidence Breakdown')
        st.caption('Signal weights control directional bias voting only. Floor/ceiling uses the evidence-based IV + VRP pipeline.')

        evidence_rows = []
        for sig in result['signals']:
            if sig['bias'] == 'bullish':
                bias_display = 'Bullish'
            elif sig['bias'] == 'bearish':
                bias_display = 'Bearish'
            else:
                bias_display = 'Neutral'
            evidence_rows.append({
                'Signal': sig['name'].replace('_', ' ').title(),
                'Bias': bias_display,
                'Weight': f"{sig['weight']*100:.0f}%",
                'Evidence': sig['evidence'],
            })
        st.dataframe(pd.DataFrame(evidence_rows))

        st.markdown('---')

        # ── Range Engine Calibration — engine-true, out-of-sample ─────────────
        # Replays the ACTUAL evidence-based floor/ceiling engine over SPY history
        # (VIX as the point-in-time IV proxy), grades realized next-session
        # coverage per sigma against its Gaussian target, and runs an anchored
        # out-of-sample width sweep. This is the faithful backbone calibration:
        # the same IV → VRP → regime → VIX-term pipeline production uses, with the
        # non-replayable option-chain dealer overlay held neutral. SPY is the
        # reference instrument (VIX is its vol index, and the forward-test ticker).
        st.subheader('Range Engine Calibration — SPY (out-of-sample)')
        st.caption('Replays the real floor/ceiling engine over SPY history (VIX as the IV proxy) and grades '
                   'how often the next-session close actually landed inside each confidence band.')
        calib = load_range_calibration('SPY', period='2y')
        if calib and calib.get('summary', {}).get('n_days', 0) > 0:
            summ = calib['summary']
            sweep = calib.get('sweep', {})
            cov = summ['coverage']
            tgt = summ['targets']
            cal_err = summ.get('calibration_error')
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric('Days Graded', summ['n_days'])
            c1 = cov.get('1sigma')
            t1 = tgt.get('1sigma', 68.27)
            if c1 is not None:
                cc2.metric('1σ Coverage', f"{c1:.1f}%", delta=f"{c1 - t1:+.1f}% vs {t1:.1f}%")
            c2 = cov.get('2sigma')
            t2 = tgt.get('2sigma', 95.45)
            if c2 is not None:
                cc3.metric('2σ Coverage', f"{c2:.1f}%", delta=f"{c2 - t2:+.1f}% vs {t2:.1f}%")
            if cal_err is not None:
                cc4.metric('Calibration Error', f"{cal_err:.2f}",
                           help='Mean |coverage − target| across sigmas. Lower = better calibrated.')

            label_disp = {'1sigma': '1σ (target 68.3%)', '1_5sigma': '1.5σ (target 86.6%)',
                          '2sigma': '2σ (target 95.4%)'}
            cov_rows = []
            for label in cov:
                target = tgt.get(label)
                cov_rows.append({
                    'Confidence': label_disp.get(label, label),
                    'Realized Coverage': f"{cov[label]:.1f}%",
                    'Target': f"{target:.1f}%" if target is not None else '—',
                    'Gap': f"{cov[label] - target:+.1f}%" if target is not None else '—',
                })
            st.dataframe(pd.DataFrame(cov_rows))
            if summ.get('mean_width_pct') is not None:
                st.caption(f"Mean 1σ band width: {summ['mean_width_pct']:.2f}% of spot — "
                           'how tight the honest range is.')

            # Out-of-sample width verdict (anchored train/test, like auto-retune).
            if sweep and 'error' not in sweep:
                if sweep.get('improved'):
                    st.success(
                        f"Out-of-sample tuner: widen bands ×{sweep['best_width']:.2f} — "
                        f"holdout calibration error {sweep['baseline_test_error']:.2f} → "
                        f"{sweep['proposed_test_error']:.2f}.")
                else:
                    st.info(
                        f"Out-of-sample tuner: keep ×1.00 — the in-sample-best ×{sweep['best_width']:.2f} "
                        f"did not beat baseline on the holdout "
                        f"(error {sweep['baseline_test_error']:.2f}). No overfit applied.")
            st.caption('Engine-true backbone: IV → VRP shrink → regime / VIX-term scaling, with the '
                       'option-chain dealer overlay held neutral (not replayable without point-in-time chains).')
        else:
            st.info('Not enough SPY/VIX history to calibrate the range engine yet.')

        st.markdown('---')

        st.subheader('Range Accuracy — Realized-Vol Cross-Check')
        st.caption('Secondary check on the analyzed ticker: how often the predicted daily range contained the '
                   'actual next-day close, using realized volatility (not the engine) as the IV proxy.')
        from strategies.fractal_indicators import compute_range_containment
        price_ticker = result['resolved_ticker'] if result['proxy_used'] else result['ticker']
        range_df = fetch_stock_data(price_ticker, period='1y')
        if not range_df.empty:
            containment = compute_range_containment(range_df, window=60)
            ra1, ra2, ra3, ra4 = st.columns(4)
            ra1.metric('Days Tested', containment['days_tested'])
            c1s = containment['containment_1sigma_pct']
            ra2.metric('1-Sigma Containment', f"{c1s:.1f}%",
                       delta=f"{c1s - 68.3:+.1f}% vs expected 68.3%")
            c2s = containment['containment_2sigma_pct']
            ra3.metric('2-Sigma Containment', f"{c2s:.1f}%",
                       delta=f"{c2s - 95.4:+.1f}% vs expected 95.4%")
            # Accuracy grade
            deviation = abs(c1s - 68.3) + abs(c2s - 95.4)
            if deviation < 5:
                grade = 'A — Excellent'
            elif deviation < 10:
                grade = 'B — Good'
            elif deviation < 20:
                grade = 'C — Fair'
            else:
                grade = 'D — Needs refinement'
            ra4.metric('Calibration Grade', grade)

            with st.expander('Daily Range Results (last 20 days)'):
                if containment['daily_results']:
                    range_rows = []
                    for dr in containment['daily_results'][-20:]:
                        range_rows.append({
                            'Date': dr['date'].strftime('%Y-%m-%d') if hasattr(dr['date'], 'strftime') else str(dr['date'])[:10],
                            'Close': f"${dr['close']:.2f}",
                            'Next Close': f"${dr['next_close']:.2f}",
                            '1-Sigma Range': f"${dr['range_low_1s']:.2f} — ${dr['range_high_1s']:.2f}",
                            'In 1-Sigma': 'Yes' if dr['in_1sigma'] else 'No',
                            'In 2-Sigma': 'Yes' if dr['in_2sigma'] else 'No',
                        })
                    st.dataframe(pd.DataFrame(range_rows))

    # ──────────────────────────────────────────────────────────────────────
    # TAB 4 — Track Record: live predictions, dealer-pin skill, auto-retune
    # ──────────────────────────────────────────────────────────────────────
    with tab_track:
        # NOTE: the old "Live Prediction Tracking" table was removed here. It
        # scored each forecast against the *next* trading day's daily close
        # (hist_df.index > pred_date), so same-day 0DTE forecasts never got a
        # close ("requires at least one subsequent trading day") and it merely
        # duplicated — less rigorously — the graded Outcomes views below. The
        # scored ledger (Dealer-Pin Close, from the pre-open pin pipeline) is
        # the single source of truth for accuracy, fed only by the scheduled jobs.

        # ── Dealer-Pin Close — Track Record (the 6:25am pre-open call) ───────
        # ONE dealer-pin close prediction per day: recorded pre-market at
        # 6:25am PT — anchored on the live pre-market spot and the overnight-
        # settled 0DTE chain (OI/max-pain/gamma are fixed by then) — and graded
        # against that day's actual closing print at 1:16pm PT. Sourced from
        # the PinForecasts/PinOutcomes ledgers. The after-close daily recorder
        # still logs a reference row into Predictions/Outcomes for calibration
        # research, but it is NOT displayed here — the pre-open call is the
        # scored dealer-pin track record.
        st.subheader('Dealer-Pin Close — Track Record')
        st.caption(
            'Forecast of the closing print, logged pre-market at 6:25am PT '
            '(live pre-market spot + overnight-settled 0DTE chain) and graded '
            'against the realized close at 1:16pm PT the same day. Skill = '
            'naive error − model error, so positive means the pin added value '
            'over "price just stays at spot."'
        )
        try:
            from track_record import (
                summarize_track_record as _pin_summarize,
                join_predictions_outcomes as _pin_join,
            )
            from sheets_logger import (
                read_pin_forecasts, read_pin_forecasts_csv,
                read_pin_outcomes, read_pin_outcomes_csv,
            )

            # Read BOTH ledgers: the recorded forecasts (PinForecasts) and the
            # grades (PinOutcomes). The section used to read only the graded
            # store and hide everything until a forecast matured — so a pin
            # logged this morning was invisible until the next evening. Now we
            # surface the recorded forecast immediately (status "Pending") and
            # layer the grade on once the session closes.
            pin_fc = read_pin_forecasts() if _sheets_ok else read_pin_forecasts_csv()
            pin_df = read_pin_outcomes() if _sheets_ok else read_pin_outcomes_csv()
            _tkr = str(result['ticker']).upper()

            def _pin_by_ticker(df):
                if df is not None and not df.empty and 'ticker' in df.columns:
                    return df[df['ticker'].astype(str).str.upper().str.strip() == _tkr].copy()
                return df.copy() if df is not None and not df.empty else pd.DataFrame()

            fc = _pin_by_ticker(pin_fc)
            pin_out = _pin_by_ticker(pin_df)

            if fc.empty:
                st.info(
                    f'No dealer-pin close forecasts recorded for {_tkr} yet. The '
                    'automation logs one pre-market at 6:25am PT; it appears '
                    'here as soon as it is recorded, then earns a grade at '
                    '1:16pm PT after that day\'s close.'
                )
            else:
                # Left-join each recorded forecast to its grade (blank if still
                # pending). A forecast counts as graded once a realized close is
                # attached.
                joined = _pin_join(fc, pin_out)
                if 'actual_close' in joined.columns:
                    _ac = pd.to_numeric(joined['actual_close'], errors='coerce')
                else:
                    _ac = pd.Series([float('nan')] * len(joined), index=joined.index)
                joined['status'] = ['Graded' if pd.notna(v) else 'Pending' for v in _ac]

                n_total = len(joined)
                n_graded = int(_ac.notna().sum())
                n_pending = n_total - n_graded
                st.caption(f'{n_total} recorded · {n_graded} graded · {n_pending} pending')

                # Headline skill scorecard — only meaningful once a forecast has
                # actually matured against a realized close.
                psumm = _pin_summarize(pin_out)
                if psumm['n_graded'] > 0:
                    if psumm['beats_naive']:
                        st.success(
                            f"✅ Dealer pin beats the naive baseline — mean error "
                            f"${psumm['mean_abs_err']:.2f} vs ${psumm['naive_mean_abs_err']:.2f} "
                            f"for \"stays at spot\" across {psumm['n_graded']} graded forecast(s)."
                        )
                    else:
                        st.warning(
                            f"⚠️ Dealer pin not yet beating the naive baseline — mean error "
                            f"${psumm['mean_abs_err']:.2f} vs ${psumm['naive_mean_abs_err']:.2f} "
                            f"for \"stays at spot\" across {psumm['n_graded']} graded forecast(s)."
                        )

                    pp1, pp2, pp3, pp4 = st.columns(4)
                    pp1.metric('Graded Forecasts', psumm['n_graded'])
                    _pskill = psumm['mean_skill']
                    pp2.metric(
                        'Mean Skill ($)',
                        f"{_pskill:+.2f}" if _pskill is not None else '—',
                        help='Avg ($) the pre-open pin beat the naive spot baseline by.',
                    )
                    pp3.metric(
                        'Skill Rate',
                        f"{psumm['skill_rate'] * 100:.0f}%" if psumm['skill_rate'] is not None else '—',
                        help='Share of pre-open forecasts that beat the naive baseline.',
                    )
                    pp4.metric(
                        'Direction Accuracy',
                        f"{psumm['dir_accuracy'] * 100:.0f}%" if psumm['dir_accuracy'] is not None else '—',
                        help='Share of pre-open forecasts that called up/down vs spot correctly.',
                    )

                    pp5, pp6, pp7 = st.columns(3)
                    pp5.metric('Mean Abs Error', f"${psumm['mean_abs_err']:.2f}")
                    pp6.metric('Naive Baseline Error', f"${psumm['naive_mean_abs_err']:.2f}")
                    pp7.metric(
                        'In-Range Rate',
                        f"{psumm['in_range_rate'] * 100:.0f}%" if psumm['in_range_rate'] is not None else '—',
                        help='Share of realized closes inside the pre-open [floor, ceiling] band.',
                    )

                    # Progress over time: cumulative mean skill as the record grows.
                    prog = pin_out.copy()
                    prog['pred_date'] = pd.to_datetime(prog['pred_date'], errors='coerce')
                    prog = prog.dropna(subset=['pred_date']).sort_values('pred_date')
                    prog['skill'] = pd.to_numeric(prog['skill'], errors='coerce')
                    prog = prog.dropna(subset=['skill'])
                    if len(prog) >= 2:
                        prog['Cumulative Mean Skill ($)'] = prog['skill'].expanding().mean()
                        chart = prog.set_index('pred_date')[['Cumulative Mean Skill ($)']]
                        st.markdown('**Progress over time** — cumulative mean skill vs the naive baseline')
                        st.line_chart(chart)
                        st.caption('Above zero and trending up = the pin is learning to beat "do nothing."')
                elif n_pending:
                    st.info(
                        f'{n_pending} dealer-pin close forecast(s) recorded and awaiting a '
                        'realized close — the skill scorecard fills in automatically '
                        'after each session closes.'
                    )

                # Always show the recorded forecasts themselves (newest first),
                # with grade columns where a session has matured.
                show = joined.copy()
                if 'date' in show.columns:
                    show = show.sort_values(
                        'date', ascending=False,
                        key=lambda s: pd.to_datetime(s, errors='coerce'),
                    )
                    # Plain date — casting to datetime renders a bogus 00:00:00.
                    show['date'] = show['date'].astype(str).str[:10]
                # The ledger's `timestamp` is the actual recording moment in
                # Pacific time (legacy UTC rows were migrated in place). Surface
                # it as the pred time so a 6:25am run reads as 06:25, not midnight.
                if 'timestamp' in show.columns:
                    _ts = pd.to_datetime(show['timestamp'], errors='coerce')
                    show['pred_time'] = _ts.dt.strftime('%H:%M').fillna('—')
                disp_cols = [c for c in [
                    'date', 'pred_time', 'expiry', 'status', 'spot_price',
                    'estimated_close', 'pin_target', 'max_pain', 'floor', 'ceiling',
                    'actual_close', 'close_abs_err', 'naive_abs_err', 'skill',
                    'in_range', 'dir_correct',
                ] if c in show.columns]
                rename = {
                    'date': 'Pred Date', 'pred_time': 'Pred Time (PT)',
                    'expiry': 'Expiry', 'status': 'Status',
                    'spot_price': 'Spot', 'estimated_close': 'Est Close',
                    'pin_target': 'Pin Target', 'max_pain': 'Max Pain',
                    'floor': 'Floor', 'ceiling': 'Ceiling', 'actual_close': 'Actual',
                    'close_abs_err': 'Abs Err', 'naive_abs_err': 'Naive Err',
                    'skill': 'Skill', 'in_range': 'In Range', 'dir_correct': 'Dir OK',
                }
                st.dataframe(
                    show[disp_cols].rename(columns=rename).head(60),
                    width='stretch', hide_index=True,
                )
        except Exception as e:
            st.warning(f'Could not load dealer-pin track record: {e}')

        # ── Signal Weights & Auto-Retune ──────────────────────────────────
        st.markdown('---')
        st.subheader('Signal Weights & Auto-Retune')
        st.caption('Current signal weights used for floor/ceiling/bias calculation. '
                   'Auto-retune analyzes prediction accuracy and proposes safer weight adjustments.')

        # Display current weights with delta from baseline
        wt_names = list(active_weights.keys())
        wt_cols = st.columns(len(wt_names))
        for i, name in enumerate(wt_names):
            baseline_val = SIGNAL_WEIGHTS.get(name, active_weights[name])
            delta = active_weights[name] - baseline_val
            delta_str = f"{delta:+.3f}" if abs(delta) > 0.001 else None
            wt_cols[i].metric(name.replace('_', ' ').title(), f"{active_weights[name]:.3f}", delta=delta_str)

        # Weight history
        if _sheets_ok:
            try:
                wh_df = read_weight_history()
                if not wh_df.empty:
                    with st.expander(f'Weight Change History ({len(wh_df)} changes)'):
                        st.dataframe(wh_df)
            except Exception:
                pass

        # Retune button
        st.markdown('---')
        retune_col1, retune_col2 = st.columns([1, 3])
        with retune_col1:
            retune_btn = st.button('Run Auto-Retune', key='retune_btn')
        with retune_col2:
            st.caption(
                'Analyzes prediction accuracy over the last 90 days. '
                'Requires 60+ days of data. Changes are capped at 15% per cycle. '
                'All changes are validated on a holdout set before applying.'
            )

        if retune_btn:
            from auto_retune import run_retune
            with st.spinner('Running auto-retune analysis...'):
                retune_pred = read_predictions() if _sheets_ok else read_predictions_csv()
                retune_result = run_retune(retune_pred, active_weights)

            if not retune_result["eligible"]:
                st.warning(f'Not eligible for retuning: {retune_result["reason"]}')
            else:
                # Signal scores
                if retune_result["signal_scores"]:
                    st.markdown('**Signal Accuracy Scores** (positive = signal improves accuracy)')
                    sc_cols = st.columns(len(retune_result["signal_scores"]))
                    for i, (name, score) in enumerate(retune_result["signal_scores"].items()):
                        sc_cols[i].metric(name.replace('_', ' ').title(), f"{score:+.1f}%")

                # Proposed weights
                if retune_result["proposed_weights"]:
                    st.markdown('**Proposed Weight Changes**')
                    change_data = []
                    for name in active_weights:
                        proposed_val = retune_result['proposed_weights'].get(name, active_weights[name])
                        change_data.append({
                            'Signal': name.replace('_', ' ').title(),
                            'Current': f"{active_weights[name]:.4f}",
                            'Proposed': f"{proposed_val:.4f}",
                            'Change': f"{proposed_val - active_weights[name]:+.4f}",
                        })
                    st.dataframe(pd.DataFrame(change_data))

                # Holdout result
                if retune_result["holdout_passed"]:
                    st.success(
                        f'Holdout validation PASSED. '
                        f'Baseline accuracy: {retune_result["baseline_accuracy"]:.1f}% | '
                        f'Proposed accuracy: {retune_result["proposed_accuracy"]:.1f}%'
                    )
                    if _sheets_ok and st.button('Apply New Weights', key='apply_retune'):
                        for name, old_val, new_val in retune_result["weight_changes"]:
                            log_weight_change(
                                name, old_val, new_val,
                                reason=f"Auto-retune: holdout {retune_result['proposed_accuracy']:.1f}% "
                                       f"vs baseline {retune_result['baseline_accuracy']:.1f}%"
                            )
                        st.success('Weights updated and logged to Google Sheets. Refresh to use new weights.')
                    elif not _sheets_ok:
                        st.warning('Connect Google Sheets to persist weight changes across sessions.')
                else:
                    st.error(
                        f'Holdout validation FAILED. Proposed weights did not improve accuracy. '
                        f'Baseline: {retune_result["baseline_accuracy"]:.1f}% | '
                        f'Proposed: {retune_result["proposed_accuracy"]:.1f}%'
                    )

                # Regime analysis
                if retune_result.get("regime_adjustments"):
                    st.markdown('**Regime Analysis** (range tightening opportunities)')
                    regime_data = []
                    for regime, info in retune_result["regime_adjustments"].items():
                        regime_data.append({
                            'Regime': regime.replace('_', ' ').title(),
                            'Accuracy': f"{info['accuracy']:.0f}%",
                            'Range Scale': f"{info['range_scale']:.2f}x",
                            'Avg Width %': f"{info['avg_range_width_pct']:.2f}%",
                            'Samples': info['count'],
                        })
                    st.dataframe(pd.DataFrame(regime_data))
