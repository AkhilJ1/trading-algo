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
from datetime import date, timedelta

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))

from config import WATCHLIST, MA_SHORT, MA_LONG, RSI_OVERSOLD, BB_WICK_LOOKBACK
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
st.sidebar.markdown("---")
st.sidebar.caption(f"Watchlist: {len(WATCHLIST)} tickers")
st.sidebar.caption(f"Strategy: RSI<{RSI_OVERSOLD} + BB wick | MA({MA_SHORT}/{MA_LONG})")


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
                html = (
                    f'<div style="background:#1a1f2e;padding:10px 14px;border-radius:6px;'
                    f'border-left:3px solid {dir_color};margin:4px 0;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<div>'
                    f'<span style="font-weight:bold;font-size:14px;color:#fafafa;">{opp["ticker"]}</span>'
                    f'&nbsp;&nbsp;{_tier_badge(tier_num)}'
                    f'&nbsp;&nbsp;<span style="color:{dir_color};font-weight:bold;">{dir_icon} {opp["direction"].upper()}</span>'
                    f'&nbsp;&nbsp;<span style="color:#90caf9;font-size:13px;">{opp["setup"]}</span>'
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
                st.dataframe(df_cons, use_container_width=True)

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
        st.dataframe(pd.DataFrame(watch_data), use_container_width=True)

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
            st.dataframe(pd.DataFrame(factor_rows), use_container_width=True)

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
    st.title('📡 Daily Scanner')
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

    st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(dd_fig, use_container_width=True)

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
                st.plotly_chart(hist_fig, use_container_width=True)
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
                st.plotly_chart(hm_fig, use_container_width=True)
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
            st.plotly_chart(rs_fig, use_container_width=True)

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
                    st.plotly_chart(mc_fig, use_container_width=True)

        # ── Walk-Forward Validation ───────────────────────────────────────
        with st.expander('Walk-Forward Validation'):
            st.caption('Tests if strategy parameters are overfit by validating on unseen data.')
            wf_ticker = st.text_input('Ticker for walk-forward', value=tickers[0], key='wf_ticker')
            wf_strat = st.selectbox('Strategy', STRATEGIES, key='wf_strat',
                                     index=STRATEGIES.index(selected_strategy))
            if st.button('Run Walk-Forward', key='wf_btn'):
                with st.spinner('Running walk-forward analysis (8 folds)...'):
                    from walk_forward import walk_forward_test
                    wf = walk_forward_test(wf_ticker, wf_strat)
                if 'error' in wf:
                    st.error(wf['error'])
                else:
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
                    st.dataframe(folds_df[['fold','train_sharpe','test_sharpe','test_win_rate','test_return','test_trades']])

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

    from options_fetcher import fetch_expiration_dates
    from strategies.fractal_options import compute_composite_analysis

    # ── Session state ──────────────────────────────────────────────────────
    if 'fo_result' not in st.session_state:
        st.session_state.fo_result = None

    # ── Input row ──────────────────────────────────────────────────────────
    inp1, inp2, inp3 = st.columns([3, 2, 1])
    with inp1:
        fo_ticker = st.text_input('Ticker (stocks, ETFs, or ES=F / NQ=F / YM=F)',
                                   value='SPY', key='fo_ticker_input')
    with inp2:
        expiries = fetch_expiration_dates(fo_ticker) if fo_ticker.strip() else []
        fo_expiry = st.selectbox('Options Expiry', ['Nearest'] + expiries[:10],
                                  key='fo_expiry_select')
    with inp3:
        st.write('')
        st.write('')
        analyze_btn = st.button('Analyze', key='fo_analyze')

    # ── Load active weights ──────────────────────────────────────────────
    from sheets_logger import (
        is_sheets_available, get_current_weights,
        log_prediction, log_prediction_csv,
        read_predictions, read_predictions_csv,
        read_weight_history, log_weight_change,
    )
    _sheets_ok = is_sheets_available()
    try:
        active_weights = get_current_weights() if _sheets_ok else dict(SIGNAL_WEIGHTS)
    except Exception:
        active_weights = dict(SIGNAL_WEIGHTS)
    from config import SIGNAL_WEIGHTS

    # ── Run analysis ───────────────────────────────────────────────────────
    if analyze_btn and fo_ticker.strip():
        exp_arg = None if fo_expiry == 'Nearest' else fo_expiry
        with st.spinner(f'Analyzing {fo_ticker.upper()}... (options + fractals + GEX)'):
            result = compute_composite_analysis(
                fo_ticker.strip().upper(), exp_arg, weights=active_weights,
            )
        st.session_state.fo_result = result

    result = st.session_state.fo_result
    if result is None:
        st.info('Enter a ticker and click **Analyze** to see floor, ceiling, and directional bias with evidence.')
        st.stop()
    if 'error' in result:
        st.error(result['error'])
        st.stop()

    # ══════════════════════════════════════════════════════════════════════
    # SECTION A: Bias Banner + Key Metrics
    # ══════════════════════════════════════════════════════════════════════
    proxy_note = f"  (via {result['resolved_ticker']} options)" if result['proxy_used'] else ""

    # Fetch live/latest price on every refresh (falls back to last analysis price)
    _live_price = result['spot_price']
    try:
        import yfinance as yf
        _hist = yf.Ticker(result['ticker']).history(period='2d')
        if not _hist.empty:
            _live_price = float(_hist['Close'].iloc[-1])
    except Exception:
        pass

    st.markdown(f"### {result['ticker']}{proxy_note}  —  ${_live_price:.2f}")
    _price_delta = _live_price - result['spot_price']
    _analysis_note = f"Expiry: {result['expiry']}  |  Analyzed: {result['timestamp']}"
    if abs(_price_delta) > 0.005:
        _analysis_note += f"  |  Since analysis: {'+' if _price_delta > 0 else ''}{_price_delta:.2f}"
    st.caption(_analysis_note)

    bias = result['bias']
    conf = result['confidence']
    floor_val = result['floor']
    ceil_val = result['ceiling']
    banner = (f"**{bias}** — {conf:.0f}% Confidence  |  "
              f"Floor: ${floor_val:.2f}  |  Ceiling: ${ceil_val:.2f}")
    if bias == 'BULLISH':
        st.success(banner)
    elif bias == 'BEARISH':
        st.error(banner)
    else:
        st.warning(banner)

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

    # Iron Condor Range Levels
    ranges = result.get('ranges', {})
    if ranges:
        st.subheader('Iron Condor Range Levels')
        st.caption('Choose your confidence level for short strikes')
        rc1, rc2, rc3 = st.columns(3)
        r_ratio = result.get('price_ratio', 1.0) if result['proxy_used'] else 1.0
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

    # VIX regime + Ensemble consensus row
    try:
        from strategies.vix_filter import fetch_vix, classify_vix_regime
        vix_s = fetch_vix(period='1mo')
        if not vix_s.empty:
            current_vix = float(vix_s.iloc[-1])
            vix_regime = classify_vix_regime(current_vix)
            regime_colors = {'low_vol': 'green', 'normal': 'blue', 'elevated': 'orange', 'crisis': 'red'}
        else:
            current_vix = None
            vix_regime = 'unknown'
    except Exception:
        current_vix = None
        vix_regime = 'unknown'

    try:
        from strategies.ensemble import generate_signals as _ens_signals
        price_ticker = result['resolved_ticker'] if result['proxy_used'] else result['ticker']
        ens_df = fetch_stock_data(price_ticker, period='1y')
        if not ens_df.empty:
            if isinstance(ens_df.index, pd.DatetimeIndex) and ens_df.index.tz is not None:
                ens_df.index = ens_df.index.tz_localize(None)
            ens_df = _ens_signals(ens_df, threshold=3)
            ens_votes = int(ens_df['ensemble_votes_buy'].iloc[-1])
            ens_total = 6
            ens_conf = ens_df['ensemble_confidence'].iloc[-1]
        else:
            ens_votes = 0
            ens_total = 6
            ens_conf = 0
    except Exception:
        ens_votes = 0
        ens_total = 6
        ens_conf = 0

    v1, v2, v3 = st.columns(3)
    if current_vix is not None:
        v1.metric('VIX', f"{current_vix:.1f}", delta=vix_regime.replace('_', ' ').title())
    else:
        v1.metric('VIX', 'N/A')
    v2.metric('Ensemble Consensus', f"{ens_votes}/{ens_total} bullish",
              delta=f"{ens_conf*100:.0f}% agreement")
    v3.metric('VIX Regime Signal',
              'Full Size' if vix_regime in ('low_vol', 'normal')
              else 'Half Size' if vix_regime == 'elevated'
              else 'Cash Only' if vix_regime == 'crisis' else 'Unknown')

    # ── Log prediction (after VIX is available) ───────────────────────────
    if analyze_btn and 'error' not in result:
        _gex_df = result.get('gex_df', pd.DataFrame())
        _gex_net = float(_gex_df['net_gex'].sum()) if _gex_df is not None and not _gex_df.empty else None
        _vix_val = current_vix if current_vix is not None else None
        _regime_val = vix_regime if vix_regime != 'unknown' else None

        logged = False
        if _sheets_ok:
            try:
                logged = log_prediction(
                    date_str=result.get('timestamp', '')[:10],
                    ticker=result.get('ticker', ''),
                    spot_price=result.get('spot_price', 0),
                    floor=result.get('floor', 0),
                    ceiling=result.get('ceiling', 0),
                    bias=result.get('bias', ''),
                    confidence=result.get('confidence', 0),
                    expiry=result.get('expiry', ''),
                    vix=_vix_val, gex_net=_gex_net, regime=_regime_val,
                )
            except Exception:
                pass
        if not logged:
            log_prediction_csv(
                date_str=result.get('timestamp', '')[:10],
                ticker=result.get('ticker', ''),
                spot_price=result.get('spot_price', 0),
                floor=result.get('floor', 0),
                ceiling=result.get('ceiling', 0),
                bias=result.get('bias', ''),
                confidence=result.get('confidence', 0),
                expiry=result.get('expiry', ''),
                vix=_vix_val, gex_net=_gex_net, regime=_regime_val,
            )

    st.markdown('---')

    # ══════════════════════════════════════════════════════════════════════
    # SECTION B: Variance Risk Premium Analysis
    # ══════════════════════════════════════════════════════════════════════
    st.subheader('Variance Risk Premium Analysis')
    st.caption('IV systematically overstates realized vol — this is the #1 edge for iron condor sellers')
    vp1, vp2, vp3, vp4 = st.columns(4)
    vp1.metric('Raw ATM IV', f"{vrp.get('iv', 0)*100:.1f}%")
    vp2.metric('Parkinson RV (20d)', f"{vrp.get('rv_parkinson', 0)*100:.1f}%" if vrp.get('rv_parkinson') else 'N/A')
    vp3.metric('VRP Ratio (RV/IV)', f"{vrp.get('scaling_factor', 0):.3f}")
    vp4.metric('IV Overstatement', f"{vrp.get('vrp_pct', 0):.1f}%")

    st.markdown('---')

    # ══════════════════════════════════════════════════════════════════════
    # SECTION C: Fractal Structure Chart
    # ══════════════════════════════════════════════════════════════════════
    st.subheader('Fractal Market Structure')
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
                       annotation_position='left')

    # Fractal dimension subplot
    if 'fractal_dimension' in price_df.columns:
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

    fig_frac.update_layout(
        height=650, xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=50, b=20, l=0, r=80),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
    )
    fig_frac.update_xaxes(gridcolor='#1e2130')
    fig_frac.update_yaxes(gridcolor='#1e2130')
    st.plotly_chart(fig_frac)

    st.markdown('---')

    # ══════════════════════════════════════════════════════════════════════
    # SECTION D: GEX Profile
    # ══════════════════════════════════════════════════════════════════════
    st.subheader('Gamma Exposure (GEX) Profile')
    st.caption('Positive GEX = sticky/mean-reverting (dealers dampen moves)  |  '
               'Negative GEX = slippery (dealers amplify moves)')
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

    # ══════════════════════════════════════════════════════════════════════
    # SECTION E: Options OI Walls
    # ══════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════
    # SECTION F: Sentiment & IV Skew
    # ══════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════
    # SECTION G: Evidence Breakdown Table
    # ══════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════
    # SECTION H: Range Accuracy (Historical Validation)
    # ══════════════════════════════════════════════════════════════════════
    st.subheader('Range Accuracy — Historical Validation')
    st.caption('Tests how often the predicted daily range contained the actual next-day close (using realized volatility as IV proxy)')
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

    st.markdown('---')

    # ══════════════════════════════════════════════════════════════════════
    # SECTION I: Raw Options Data
    # ══════════════════════════════════════════════════════════════════════
    with st.expander('Raw Options Chain Data'):
        from options_fetcher import fetch_options_chain as _fetch_chain
        raw_c, raw_p, _ = _fetch_chain(result['ticker'])
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

    st.markdown('---')

    # ══════════════════════════════════════════════════════════════════════
    # SECTION J: Live Prediction Tracking
    # ══════════════════════════════════════════════════════════════════════
    st.subheader('Live Prediction Tracking')
    st.caption('Historical log of predictions — compare predicted floor/ceiling vs actual closes')
    try:
        pred_df = read_predictions() if _sheets_ok else read_predictions_csv()
        if not pred_df.empty:
            ticker_preds = pred_df[pred_df['ticker'] == result['ticker']].tail(30)
            if not ticker_preds.empty:
                scored_rows = []
                price_ticker = result['resolved_ticker'] if result['proxy_used'] else result['ticker']
                hist_df = fetch_stock_data(price_ticker, period='3mo')
                if not hist_df.empty and isinstance(hist_df.index, pd.DatetimeIndex):
                    if hist_df.index.tz is not None:
                        hist_df.index = hist_df.index.tz_localize(None)
                    for _, pred in ticker_preds.iterrows():
                        pred_date = pd.Timestamp(pred['date'])
                        future = hist_df[hist_df.index > pred_date]
                        if len(future) > 0:
                            actual_close = float(future['Close'].iloc[0])
                            in_range = pred['floor'] <= actual_close <= pred['ceiling']
                            bias_correct = (
                                (pred['bias'] == 'BULLISH' and actual_close > pred['spot_price']) or
                                (pred['bias'] == 'BEARISH' and actual_close < pred['spot_price']) or
                                (pred['bias'] == 'NEUTRAL')
                            )
                            scored_rows.append({
                                'Date': str(pred['date'])[:10] if not pd.isna(pred['date']) else '',
                                'Spot': f"${pred['spot_price']:.2f}",
                                'Floor': f"${pred['floor']:.2f}",
                                'Ceiling': f"${pred['ceiling']:.2f}",
                                'Bias': pred['bias'],
                                'Actual Close': f"${actual_close:.2f}",
                                'In Range': 'Yes' if in_range else 'No',
                                'Bias Correct': 'Yes' if bias_correct else 'No',
                            })

                if scored_rows:
                    scored_disp = pd.DataFrame(scored_rows)
                    range_acc = scored_disp['In Range'].value_counts().get('Yes', 0) / len(scored_disp) * 100
                    bias_acc = scored_disp['Bias Correct'].value_counts().get('Yes', 0) / len(scored_disp) * 100
                    lt1, lt2, lt3 = st.columns(3)
                    lt1.metric('Predictions Tracked', len(scored_disp))
                    lt2.metric('Range Accuracy', f"{range_acc:.0f}%")
                    lt3.metric('Bias Accuracy', f"{bias_acc:.0f}%")
                    st.dataframe(scored_disp)
                else:
                    st.info(f'{len(ticker_preds)} prediction(s) logged for {result["ticker"]}. '
                            'Accuracy scoring requires at least one subsequent trading day.')
            else:
                st.info(f'No predictions logged yet for {result["ticker"]}. Click Analyze to start tracking.')
        else:
            st.info('No predictions recorded yet. Click Analyze to start tracking.')
            st.caption('Data source: ' + ('Google Sheets' if _sheets_ok else 'Local CSV (connect Google Sheets for persistence)'))
    except Exception as e:
        st.warning(f'Could not load predictions: {e}')

    # ══════════════════════════════════════════════════════════════════════
    # SECTION K: Signal Weights & Auto-Retune
    # ══════════════════════════════════════════════════════════════════════
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
