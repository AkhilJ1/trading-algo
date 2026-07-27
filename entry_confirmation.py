"""
Entry confirmation — what to watch for AFTER a ticker shows up as unusual.

WHY THIS EXISTS
---------------
Page 1 answers "which tickers are behaving uncharacteristically for
themselves". It deliberately stops there, because the underlying signal has no
measurable forward edge (p ≈ 0.5–0.8 out-of-sample), so ranking by predicted
profit would be sorting noise.

But "TSLA is at the 4th percentile of its own RSI range" is not something you
can act on. An oversold reading can stay oversold for six weeks, and the whole
practical question is *what happens next that tells you the fall has stopped*.
This module turns that into a concrete watchlist: a handful of conditions you
can see on a 1-year daily chart in any broker app, each with

  * whether it is showing RIGHT NOW (and if not, what number it still needs),
  * how often it historically showed up at all after this ticker looked
    like this — the "you would have sat out N% of these" number, and
  * what the forward return was when you waited for it, against the return
    from simply buying the moment the setup appeared.

TWO DESIGN DECISIONS, BOTH LOAD-BEARING
---------------------------------------
* **Confirmations are measured as a WAIT, not as a filter.** The naive version
  intersects "RSI at today's percentile" with "trigger firing today" and scores
  that. It is wrong, and wrong in a way that quietly guarantees an empty or
  nonsense sample: by the time RSI has climbed back above 30 it is no longer at
  its 4th percentile, so the two conditions are close to mutually exclusive.
  What a human actually does is spot the setup, then WAIT for the confirmation
  and buy at that later bar. So that is what gets measured: from each historical
  setup, find the first bar within `MAX_WAIT` sessions where the trigger fired,
  and score the forward return from THERE.

* **Nothing here is ranked by measured edge.** Eight confirmations across ~31
  tickers is ~250 comparisons; the best-looking one is expected to look good on
  noise alone, and sorting by it would rebuild the exact "random number
  generator with a confidence badge" this page was rewritten to avoid.
  Triggers are ordered by what is showing now, then by how reliably they showed
  up at all. The return columns are reported as description, never as a ranking
  key and never with a confidence badge attached.

No trigger here uses future data. The one that could — swing-low structure,
which needs bars on both sides of the pivot — is shifted forward so it only
becomes visible on the bar a chart could actually have shown it.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import BB_PERIOD, BB_STD, BB_WICK_LOOKBACK, RSI_OVERSOLD
from trade_ideas import MIN_ANALOGS, _dedupe_events, _rsi, macd_histogram, score_analogs

# The mirror of the oversold line. config.py only names the buy-side threshold
# because the original scanner was buy-only; 100-x is the symmetric partner and
# keeps a single knob in config rather than a second one that could drift.
RSI_OVERBOUGHT = 100.0 - RSI_OVERSOLD

# How long after a setup appears we keep watching for a confirmation before
# giving up on that occurrence. Two trading weeks: long enough that a genuine
# turn has room to print, short enough that the entry still belongs to the
# setup rather than to whatever happened next month.
MAX_WAIT = 10

# A trigger counts as "showing now" if it fired within this many sessions.
# Same span as BB_WICK_LOOKBACK — one trading week is what "right now" means
# on a daily chart, and demanding it fire on today's bar exactly would blink
# the whole checklist off every time the tape pauses for a day.
RECENT_BARS = 5

# Half-width for swing pivot detection: the lowest low of 3 bars either side.
PIVOT = 3

# Lookback for divergence: today's leg against the previous leg of equal length.
DIVERGENCE_WINDOW = 20

# How close price must still be to its recent low for a divergence to count as
# current rather than a historical curiosity.
DIVERGENCE_PROXIMITY = 0.02

# Volume multiple over the 20-day average that counts as a genuine expansion.
VOLUME_SURGE = 1.5


@dataclass
class Trigger:
    """One chart-visible condition, with its full history as a boolean mask."""
    key: str
    label: str          # what to look for, in the words you'd use out loud
    where: str          # which pane of a 1-year daily chart shows it
    mask: pd.Series     # True on every bar the condition held
    detail: str         # the live reading, or what it still needs

    def bars_since(self) -> Optional[int]:
        """Sessions since it last fired; None if it never has."""
        hits = np.flatnonzero(self.mask.to_numpy(dtype=bool))
        if not len(hits):
            return None
        return int(len(self.mask) - 1 - hits[-1])

    def showing(self) -> bool:
        b = self.bars_since()
        return b is not None and b <= RECENT_BARS


def _bool(s: pd.Series, index: pd.Index) -> pd.Series:
    """
    Align to `index` and force real booleans.

    Both `shift` and `reindex` turn a bool series into object dtype with NaNs
    in the gaps, and a missing bar must read as "did not fire" rather than
    propagating NaN into a mask that later gets summed. `.where` rather than
    `.fillna` because filling an object column is a deprecated downcast.
    """
    aligned = s.reindex(index)
    return aligned.where(aligned.notna(), False).astype(bool)


def _pivot_flags(series: pd.Series, half: int, low_side: bool) -> pd.Series:
    """
    Bars that were the extreme of `half` bars either side — a swing point.

    Centred windows read `half` bars into the future, so a naive version would
    let the scan "see" a swing low three sessions before any chart could plot
    one. The result is therefore shifted forward by `half`: the flag lands on
    the bar where the pivot became confirmable, which is the first moment a
    human watching the chart could have acted on it.
    """
    win = 2 * half + 1
    roll = series.rolling(win, center=True)
    extreme = roll.min() if low_side else roll.max()
    return _bool(series == extreme, series.index)


def _structure_break(series: pd.Series, half: int, low_side: bool) -> tuple:
    """
    Higher low (or lower high) against the previous swing point.

    Returns the confirmation mask plus the last two swing values, so the page
    can print the actual levels rather than just a yes/no.
    """
    piv = _pivot_flags(series, half, low_side)
    vals = series.where(piv).dropna()
    if len(vals) < 2:
        empty = pd.Series(False, index=series.index)
        return empty, None, None
    better = vals > vals.shift(1) if low_side else vals < vals.shift(1)
    flag = _bool(better, series.index)
    # Only visible `half` bars after the pivot printed — see _pivot_flags.
    return _bool(flag.shift(half), series.index), float(vals.iloc[-1]), float(vals.iloc[-2])


def _pct(a: float, b: float) -> str:
    """Signed percentage distance from b to a, or a dash when undefined."""
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return "—"
    return f"{(a / b - 1.0) * 100.0:+.1f}%"


def build_triggers(df: pd.DataFrame, direction: str = "oversold") -> list:
    """
    The confirmation checklist for one ticker, in the direction that matters.

    `direction` is the idea's own label: an oversold name gets the conditions
    that would say the fall has stopped, an overbought one the conditions that
    would say the run is rolling over. Triggers whose inputs are missing (no
    volume in the frame, say) are dropped rather than faked.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return []

    close = df["Close"].dropna()
    if len(close) < max(2 * DIVERGENCE_WINDOW, BB_PERIOD) + 5:
        return []

    idx = close.index
    bull = direction == "oversold"
    low = df["Low"].reindex(idx) if "Low" in df.columns else close
    high = df["High"].reindex(idx) if "High" in df.columns else close
    vol = df["Volume"].reindex(idx) if "Volume" in df.columns else None

    rsi = _rsi(close)
    sma = close.rolling(BB_PERIOD).mean()
    sd = close.rolling(BB_PERIOD).std()
    bb_lower, bb_upper = sma - BB_STD * sd, sma + BB_STD * sd
    hist = macd_histogram(close)

    c_now = float(close.iloc[-1])
    rsi_now = float(rsi.iloc[-1])
    sma_now = float(sma.iloc[-1]) if pd.notna(sma.iloc[-1]) else float("nan")
    band_now = float((bb_lower if bull else bb_upper).iloc[-1])
    level = RSI_OVERSOLD if bull else RSI_OVERBOUGHT

    out = []

    # ── 1. RSI turning with price ──────────────────────────────────────────
    # "RSI slowly increasing as price increases", made precise: the RSI line
    # ticks up three sessions in a ROW, and price is net higher across them.
    #
    # The monotone requirement is what makes this a trigger rather than a coin
    # flip. Asking only for "RSI higher than three bars ago" fires on ~44% of
    # all bars, so a historical test of it measures nothing — the confirmation
    # would already be present almost every time you looked.
    slope = rsi.diff()
    price_leg = close.shift(3)
    detail = (f"RSI {rsi_now:.1f} vs {float(rsi.iloc[-4]):.1f} three sessions ago; "
              f"price {_pct(c_now, float(close.iloc[-4]))} over the same span")
    if bull:
        mask = (slope > 0) & (slope.shift(1) > 0) & (slope.shift(2) > 0) & (close > price_leg)
        out.append(Trigger(
            "rsi_up_with_price",
            "RSI ticking up three sessions running while price also closes higher",
            "RSI(14) pane — the line grinding up bar after bar, with the candles rising too",
            _bool(mask, idx), detail))
    else:
        mask = (slope < 0) & (slope.shift(1) < 0) & (slope.shift(2) < 0) & (close < price_leg)
        out.append(Trigger(
            "rsi_down_with_price",
            "RSI ticking down three sessions running while price also closes lower",
            "RSI(14) pane — the line grinding down bar after bar, with the candles falling too",
            _bool(mask, idx), detail))

    # ── 2. RSI reclaims / loses the threshold ──────────────────────────────
    if bull:
        mask = (rsi > level) & (rsi.shift(1) <= level)
        detail = (f"RSI {rsi_now:.1f} — still under {level:.0f}"
                  if rsi_now <= level else
                  f"RSI {rsi_now:.1f} — back above {level:.0f}")
        out.append(Trigger(
            "rsi_reclaim", f"RSI closing back above {level:.0f}",
            f"RSI(14) pane — the line crossing back up through the {level:.0f} line",
            _bool(mask, idx), detail))
    else:
        mask = (rsi < level) & (rsi.shift(1) >= level)
        detail = (f"RSI {rsi_now:.1f} — still above {level:.0f}"
                  if rsi_now >= level else
                  f"RSI {rsi_now:.1f} — back below {level:.0f}")
        out.append(Trigger(
            "rsi_reclaim", f"RSI dropping back below {level:.0f}",
            f"RSI(14) pane — the line crossing back down through the {level:.0f} line",
            _bool(mask, idx), detail))

    # ── 3. Divergence ──────────────────────────────────────────────────────
    # Price makes the lower low, RSI refuses to. Two adjacent windows of equal
    # length, so it is the same comparison your eye makes between two legs.
    w = DIVERGENCE_WINDOW
    leg_now, leg_prev = close.rolling(w), close.shift(w).rolling(w)
    rsi_now_leg, rsi_prev_leg = rsi.rolling(w), rsi.shift(w).rolling(w)
    if bull:
        mask = (
            (leg_now.min() < leg_prev.min())
            & (rsi_now_leg.min() > rsi_prev_leg.min())
            & (close <= leg_now.min() * (1.0 + DIVERGENCE_PROXIMITY))
        )
        out.append(Trigger(
            "bullish_divergence",
            "Price making a lower low while RSI makes a higher low",
            f"Compare the last two ~{w}-session lows: lower on price, higher on RSI",
            _bool(mask, idx),
            f"price leg low ${float(leg_now.min().iloc[-1]):.2f} vs "
            f"${float(leg_prev.min().iloc[-1]):.2f}; RSI low "
            f"{float(rsi_now_leg.min().iloc[-1]):.1f} vs {float(rsi_prev_leg.min().iloc[-1]):.1f}"))
    else:
        mask = (
            (leg_now.max() > leg_prev.max())
            & (rsi_now_leg.max() < rsi_prev_leg.max())
            & (close >= leg_now.max() * (1.0 - DIVERGENCE_PROXIMITY))
        )
        out.append(Trigger(
            "bearish_divergence",
            "Price making a higher high while RSI makes a lower high",
            f"Compare the last two ~{w}-session highs: higher on price, lower on RSI",
            _bool(mask, idx),
            f"price leg high ${float(leg_now.max().iloc[-1]):.2f} vs "
            f"${float(leg_prev.max().iloc[-1]):.2f}; RSI high "
            f"{float(rsi_now_leg.max().iloc[-1]):.1f} vs {float(rsi_prev_leg.max().iloc[-1]):.1f}"))

    # ── 4. Close back inside the Bollinger Band ────────────────────────────
    # The wick-and-reclaim candle: the bar pierced the band intraday but closed
    # back inside it. This is the original scanner's setup used as a TRIGGER
    # rather than as a screen.
    if bull:
        pierced = _bool((low <= bb_lower).rolling(BB_WICK_LOOKBACK).max() > 0, idx)
        mask = _bool(close > bb_lower, idx) & pierced
        label = "A candle wicking below the lower Bollinger Band but closing back inside"
        where = "Price pane with Bollinger Bands (20, 2) — tail pokes below, body closes above"
    else:
        pierced = _bool((high >= bb_upper).rolling(BB_WICK_LOOKBACK).max() > 0, idx)
        mask = _bool(close < bb_upper, idx) & pierced
        label = "A candle poking above the upper Bollinger Band but closing back inside"
        where = "Price pane with Bollinger Bands (20, 2) — tail pokes above, body closes below"
    out.append(Trigger(
        "bb_reclaim", label, where, mask,
        f"close ${c_now:.2f} vs band ${band_now:.2f} ({_pct(c_now, band_now)})"))

    # ── 5. The 20-day moving average ───────────────────────────────────────
    if bull:
        mask = (close > sma) & (close.shift(1) <= sma.shift(1))
        label = f"Price closing back above its {BB_PERIOD}-day moving average"
    else:
        mask = (close < sma) & (close.shift(1) >= sma.shift(1))
        label = f"Price closing back below its {BB_PERIOD}-day moving average"
    out.append(Trigger(
        "ma_reclaim", label,
        f"Price pane — add a simple moving average of {BB_PERIOD} and watch the cross",
        _bool(mask, idx),
        f"close ${c_now:.2f} vs SMA({BB_PERIOD}) ${sma_now:.2f} ({_pct(c_now, sma_now)})"))

    # ── 6. Swing structure ─────────────────────────────────────────────────
    struct_mask, last_piv, prev_piv = _structure_break(low if bull else high, PIVOT, bull)
    if last_piv is not None:
        out.append(Trigger(
            "structure",
            "A higher low — the next dip stops above the last one" if bull else
            "A lower high — the next bounce stalls below the last one",
            "Price pane — compare the last two swing points by eye",
            struct_mask,
            f"last swing {'low' if bull else 'high'} ${last_piv:.2f} vs prior ${prev_piv:.2f} "
            f"({_pct(last_piv, prev_piv)})"))

    # ── 7. Volume expansion ────────────────────────────────────────────────
    # A column of zeros is MISSING data, not a run of very quiet days — Schwab
    # reports 0 volume for index symbols ($VIX, $SPX) and yfinance does the
    # same. Kept, the row would render "never showed up, 0% of the time",
    # which reads as a fact about the ticker rather than an absent feed. Coerced
    # to numeric first because a JSON-derived column can arrive as strings.
    vol = pd.to_numeric(vol, errors="coerce") if vol is not None else None
    if vol is not None and bool((vol.fillna(0) > 0).any()):
        avg = vol.rolling(BB_PERIOD).mean()
        moved = close > close.shift(1) if bull else close < close.shift(1)
        mask = _bool(moved, idx) & _bool(vol > VOLUME_SURGE * avg, idx)
        v_now = float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else float("nan")
        a_now = float(avg.iloc[-1]) if pd.notna(avg.iloc[-1]) else float("nan")
        out.append(Trigger(
            "volume",
            f"An up day on {VOLUME_SURGE:g}x normal volume" if bull else
            f"A down day on {VOLUME_SURGE:g}x normal volume",
            "Volume pane — a bar clearly taller than the recent run of bars",
            mask,
            f"{v_now / a_now:.1f}x the {BB_PERIOD}-day average"
            if np.isfinite(v_now) and np.isfinite(a_now) and a_now else "—"))

    # ── 8. MACD histogram ──────────────────────────────────────────────────
    if bull:
        mask = (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2)) & (hist.shift(2) > hist.shift(3))
        label = "MACD histogram bars shrinking then rising, three sessions running"
    else:
        mask = (hist < hist.shift(1)) & (hist.shift(1) < hist.shift(2)) & (hist.shift(2) < hist.shift(3))
        label = "MACD histogram bars topping out then falling, three sessions running"
    out.append(Trigger(
        "macd", label,
        "MACD(12,26,9) pane — the bars, not the lines",
        _bool(mask, idx),
        f"histogram {'rising' if float(hist.iloc[-1]) > float(hist.iloc[-2]) else 'falling'} "
        f"on the latest bar"))

    return out


def confirmation_stats(
    close: pd.Series,
    bench: pd.Series,
    mask: pd.Series,
    setups: list,
    horizon: int,
    max_wait: int = MAX_WAIT,
) -> Optional[dict]:
    """
    Score "wait for this confirmation, then buy" across a ticker's own history.

    For each historical setup, walk forward up to `max_wait` sessions looking
    for the first bar where the trigger fired, and take THAT bar as the entry.
    The setup bar itself counts as wait 0 — if the confirmation is already
    showing when the setup appears, you act immediately.

    Setups where the confirmation never arrived are not failures to be scored;
    they are trades you would not have taken. They are reported separately as
    `arrival_rate`, because a trigger that only shows up a third of the time is
    a very different proposition from one that always does, however good the
    returns look on the third that fired.
    """
    if not setups or mask is None or mask.empty:
        return None

    m = mask.to_numpy(dtype=bool)
    n_bars = min(len(m), len(close))
    entries, waits = [], []
    for p in setups:
        if p >= n_bars:
            continue
        stop = min(n_bars, p + max_wait + 1)
        hit = next((q for q in range(p, stop) if m[q]), None)
        if hit is not None:
            entries.append(hit)
            waits.append(hit - p)

    considered = sum(1 for p in setups if p < n_bars)
    if not considered:
        return None

    arrival_rate = len(entries) / considered
    # Entries from neighbouring setups can land on the same bar or overlap, so
    # they get the same independence treatment the setups themselves get.
    independent = _dedupe_events(sorted(set(entries)), min_gap=horizon)
    stats = score_analogs(close, bench, independent, horizon)

    out = {
        "horizon": horizon,
        "arrival_rate": arrival_rate,
        "setups_considered": considered,
        "median_wait": float(np.median(waits)) if waits else None,
        "n": stats.n if stats else 0,
        "sufficient": bool(stats and stats.n >= MIN_ANALOGS),
    }
    if stats:
        out.update({
            "expectancy": stats.expectancy,
            "hit_rate": stats.hit_rate,
            "median": stats.median,
            "worst": stats.worst,
            "best": stats.best,
        })
    return out


def attach_confirmations(
    idea: dict,
    df: pd.DataFrame,
    bench: pd.Series,
    setups: dict,
    max_wait: int = MAX_WAIT,
) -> dict:
    """
    Add the confirmation checklist to an idea dict, in place.

    `setups` maps horizon -> analog bar positions, reused from the idea's own
    scoring so the confirmations are measured against exactly the moments that
    put this ticker on the page.

    Ordering is deliberate and is NOT by measured return: what is showing now
    comes first, then whatever showed up most reliably. With eight triggers per
    ticker across a whole watchlist, sorting on the best-looking forward number
    would surface noise every single day.
    """
    triggers = build_triggers(df, direction=idea.get("direction", "oversold"))
    if not triggers:
        idea["confirmations"] = []
        return idea

    close = df["Close"].dropna()
    b = bench.reindex(close.index).ffill()

    rows = []
    for trig in triggers:
        stats = {}
        for h, positions in (setups or {}).items():
            s = confirmation_stats(close, b, trig.mask, positions, h, max_wait=max_wait)
            if s is not None:
                stats[h] = s
        bars = trig.bars_since()
        rows.append({
            "key": trig.key,
            "label": trig.label,
            "where": trig.where,
            "detail": trig.detail,
            "showing": trig.showing(),
            "bars_since": bars,
            "stats": stats,
            # Averaged only for ordering ties; never shown as a headline number.
            "arrival_rate": float(np.mean([s["arrival_rate"] for s in stats.values()]))
                            if stats else 0.0,
        })

    rows.sort(key=lambda r: (not r["showing"], -r["arrival_rate"]))
    idea["confirmations"] = rows
    idea["confirmations_showing"] = sum(1 for r in rows if r["showing"])
    idea["max_wait"] = max_wait
    return idea


def status_text(row: dict) -> str:
    """Whether this confirmation is visible on the chart right now."""
    bars = row.get("bars_since")
    if bars is None:
        return "— never"
    if bars == 0:
        return "✅ today"
    return f"{'✅' if row.get('showing') else '—'} {bars}d ago"


def confirmation_table(idea: dict, horizon: int, compact: bool = False) -> list:
    """
    Render-ready rows for the checklist, formatted but not yet drawn.

    Kept out of the dashboard so the arithmetic is testable. The trap it exists
    to guard is unit mixing: `expectancy` is a FRACTION (0.012 == 1.2%), so a
    difference of two of them has to be scaled by 100 before it can carry a
    "pp" label. The same slip once made every lift on this page render as
    "+0pp".
    """
    base = (idea.get("stats") or {}).get(horizon) or {}
    base_exp = base.get("expectancy")

    table = []
    for r in idea.get("confirmations") or []:
        s = r.get("stats", {}).get(horizon) or {}
        if s.get("sufficient") and s.get("expectancy") is not None:
            waited = f"{s['expectancy']:+.2%}"
            delta = (f"{(s['expectancy'] - base_exp) * 100.0:+.1f}pp"
                     if base_exp is not None else "—")
        else:
            # Below MIN_ANALOGS there is no number worth printing. Saying so is
            # the point: a 2-occurrence "+8%" would read as the best row here.
            waited = delta = "too few"
        row = {
            "Look for": r["label"],
            "On your chart now": status_text(r),
            "Latest reading": r["detail"],
            "Showed up": f"{s['arrival_rate']:.0%}" if s else "—",
            "Typical wait": (f"{s['median_wait']:.0f}d"
                             if s.get("median_wait") is not None else "—"),
            f"{horizon}d vs SPY": waited,
            "vs not waiting": delta,
            "n": s.get("n", 0),
        }
        if compact:
            for drop in ("Latest reading", "Typical wait"):
                row.pop(drop)
        table.append(row)
    return table
