"""
Validated trade ideas — setups scored against their own history, not asserted.

WHY THIS EXISTS
---------------
The old scanner applied one global RSI<30 threshold to every ticker and called
whatever crossed it a buy signal. Two problems, both fatal to "high conviction":

  1. A fixed threshold is not "uncharacteristic". A low-vol staple whose RSI
     rarely dips under 40 is behaving far more unusually at 32 than a high-beta
     name that touches 25 monthly. Unusualness is only meaningful relative to
     the ticker's OWN distribution.

  2. Nothing was ever measured. No scanner signal in this repo has ever been
     graded, so "conviction" was a label with nothing behind it.

This module makes conviction earned: for each candidate it finds analogous
moments in that ticker's own history and reports what actually happened next —
including the losses.

THREE MEASUREMENT DECISIONS, EACH LOAD-BEARING
----------------------------------------------
* **Benchmark-relative returns.** Measured over this watchlist across 5 years,
  P(10-day return > 0) is 53% unconditionally while P(beating SPY) is 49.3%.
  The 53% is market drift, so a raw win rate near it means no edge at all.
  Abnormal (vs-SPY) return is the honest unit — standard event-study practice.

* **Lift, not level.** P(+5% within 10 days) is 13% for QQQ and 39% for MSTR
  purely from volatility. Ranking on hit rate would surface the same handful of
  high-beta names every day regardless of setup quality. Every rate is therefore
  reported against that ticker's OWN unconditional base rate, and `lift` — the
  difference — is what ranking uses.

* **Events, not bars.** RSI sitting under 20 for six sessions is ONE event, not
  six analogs. Counting bars would inflate n sixfold and make the base rate look
  far more certain than it is, so overlapping occurrences are collapsed to their
  first bar and analogs are spaced by at least the forward horizon.

Nothing here is a prediction. A base rate is a description of what happened when
this ticker last looked like this, with the sample size attached so a 100%-on-2
setup can never masquerade as a 68%-on-25 one.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Forward windows scored for every candidate. 5d matches the classic
# mean-reversion holding period and 10d the outer edge of the intended holding
# period; 15/30/120 are longer-run context.
#
# The long windows are deliberately NOT part of the validation decision. Two
# reasons: independent (non-overlapping) events get scarce fast — a 120-day
# window over 15 years admits at most ~31 — and at that range the forward
# return is dominated by the ticker's own drift rather than by whatever the
# setup was. They answer "what did this name do over the following months",
# which is useful background, not "did this setup work".
HORIZONS = (5, 10, 15, 30, 120)

# Only these decide `confident`. Requiring all five to clear significance would
# be a bar nothing could pass, which communicates nothing (the same calibration
# mistake as the original hit-rate gate).
VALIDATION_HORIZONS = (5, 10)

# A setup needs at least this many independent historical analogs before any
# rate computed from it is worth showing. Below this the page says
# "insufficient history" rather than inventing a conviction number.
MIN_ANALOGS = 10

# Purely informational: how often analogs cleared a round-number move. Never
# ranked on, because P(+5% in 10d) is 13% for QQQ and 39% for MSTR from
# volatility alone, so ranking on it would just sort by beta.
TARGET_RETURN = 0.05

# Confidence level for intervals. The goal is confidence that the RETURN is
# positive, so the gate is a one-sided test on mean abnormal return, not on the
# hit rate: a point estimate cannot express confidence, and 8 wins from 12 is a
# 67% hit rate whose interval still contains a coin flip.
#
# Calibration note — an earlier version required the hit-rate interval's lower
# bound to clear the unconditional base rate. That gate failed even on a
# synthetic series with a deliberately enormous injected edge (hit 65% vs 52%
# base, expectancy +0.75%/bar, n=48), which makes it useless rather than
# conservative: a threshold nothing real can pass communicates nothing. Testing
# the mean directly is both better powered and a closer match to the question.
CONFIDENCE_Z = 1.96   # ~95% two-sided, for displayed intervals
ALPHA_ONE_SIDED = 0.05

# How close a historical bar must be to today's reading to count as analogous,
# in percentile points. Wider finds more analogs but blurs what "similar" means.
PERCENTILE_TOLERANCE = 10.0


def wilson_interval(wins: int, n: int, z: float = CONFIDENCE_Z) -> tuple:
    """
    Wilson score interval for a proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and
    behaves sanely at small n and at rates near 0 or 1 — exactly the regime a
    thin analog sample lives in, and exactly where the naive interval would
    claim false precision.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


@dataclass
class AnalogStats:
    """What happened after this ticker last looked like this."""
    horizon: int
    n: int
    hit_rate: float          # P(abnormal return > 0) across analogs
    hit_ci_low: float        # conservative floor on that rate
    hit_ci_high: float
    base_rate: float         # same, unconditionally — the bar to beat
    lift: float              # hit_rate - base_rate; the point estimate
    expectancy: float        # mean abnormal return (the number that pays)
    expectancy_ci_low: float # one-sided floor on it; what ranking trusts
    p_value: float           # one-sided P(mean abnormal return <= 0)
    confident: bool          # expectancy floor above zero AND enough analogs
    median: float
    worst: float
    best: float
    target_hits: int         # how many reached TARGET_RETURN on raw return
    target_base_rate: float  # ...vs unconditionally
    examples: list = field(default_factory=list)  # winners AND losers

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["examples"] = list(self.examples)
        return d


def macd_histogram(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    """MACD histogram normalised by price, so it is comparable across tickers."""
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return (line - line.ewm(span=signal, adjust=False).mean()) / close


def scan_universe(price_frame: pd.DataFrame, benchmark: str = "SPY",
                  lookback: int = 252, ohlcv: Optional[dict] = None) -> list:
    """
    Evaluate every column of a wide close-price frame and rank by how UNUSUAL
    each ticker currently is — not by predicted profit.

    That ranking choice is the honest one. Out-of-sample, date-clustered testing
    of RSI and RSI+MACD extremes on this universe produced p-values of 0.49-0.79:
    no measurable edge. Ranking by "expected return" would therefore be sorting
    noise and presenting it as a forecast. Ranking by unusualness makes a claim
    the data does support — this ticker is behaving atypically for itself — and
    leaves the forward-looking judgement to the human, with the historical
    distribution shown alongside so that judgement is informed.

    `ohlcv` optionally supplies the full daily bars per ticker. Closes alone are
    enough to rank unusualness, but the entry-confirmation checklist needs highs,
    lows and volume — pass them and every idea gains a `confirmations` list.
    """
    if benchmark not in price_frame.columns:
        return []
    bench = price_frame[benchmark].dropna()
    out = []
    for ticker in price_frame.columns:
        if ticker == benchmark:
            continue
        close = price_frame[ticker].dropna()
        if close.empty:
            continue
        bars = (ohlcv or {}).get(ticker)
        usable_bars = (
            bars is not None and not bars.empty and "Close" in bars.columns
        )
        idea = evaluate_ticker(bars if usable_bars else pd.DataFrame({"Close": close}),
                               bench, ticker=ticker, lookback=lookback,
                               with_confirmations=usable_bars)
        if idea is None:
            continue
        macd_pct = percentile_of_last(macd_histogram(close), lookback)
        idea["macd_percentile"] = round(macd_pct, 1) if macd_pct is not None else None
        # Distance from the middle of the ticker's own distribution, in
        # percentile points. 50 = most extreme reading possible either way.
        rsi_extremity = abs(idea["rsi_percentile"] - 50.0)
        macd_extremity = abs(macd_pct - 50.0) if macd_pct is not None else 0.0
        idea["unusualness"] = round((rsi_extremity + macd_extremity) / 2.0, 1)
        # Both indicators at the same extreme. Reported because it is what the
        # user asked to see — NOT because it was found predictive. Tested
        # out-of-sample with date-clustered errors it was p=0.67.
        idea["confluence"] = bool(
            macd_pct is not None
            and (idea["rsi_percentile"] < 20 and macd_pct < 20
                 or idea["rsi_percentile"] > 80 and macd_pct > 80)
        )
        out.append(idea)

    apply_multiple_testing_correction(out)
    out.sort(key=lambda i: i["unusualness"], reverse=True)
    return out


def apply_multiple_testing_correction(ideas: list, alpha: float = ALPHA_ONE_SIDED) -> list:
    """
    Benjamini-Hochberg FDR correction across a whole scan.

    THIS IS NOT OPTIONAL. Scanning 31 tickers at a 5% one-sided threshold
    produces ~1.5 "significant" results from pure noise, so an uncorrected scan
    will essentially always surface a winner and present luck as validation —
    the precise way a screener becomes a random-number generator with a
    confidence badge.

    Each idea's per-horizon p-values are combined (worst case: a setup must hold
    at BOTH horizons), then BH-adjusted across the scan. `confident` survives
    only if it clears the adjusted threshold, and `p_combined` / `n_tested` are
    attached so the page can state how many names were searched.
    """
    scored = [i for i in ideas if i.get("stats")]
    if not scored:
        return ideas

    for idea in scored:
        ps = [st["p_value"] for h, st in idea["stats"].items()
              if h in VALIDATION_HORIZONS
              and st.get("p_value") is not None and np.isfinite(st["p_value"])]
        idea["p_combined"] = max(ps) if ps else float("nan")

    testable = [i for i in scored if np.isfinite(i.get("p_combined", np.nan))]
    m = len(testable)
    for idea in scored:
        idea["n_tested"] = m

    if m == 0:
        return ideas

    # BH: sort ascending, largest k with p_(k) <= k/m * alpha sets the cutoff.
    order = sorted(testable, key=lambda i: i["p_combined"])
    cutoff = 0.0
    for k, idea in enumerate(order, start=1):
        if idea["p_combined"] <= (k / m) * alpha:
            cutoff = idea["p_combined"]

    for idea in scored:
        p = idea.get("p_combined", float("nan"))
        survived = bool(np.isfinite(p) and p <= cutoff)
        idea["confident_uncorrected"] = idea.get("confident", False)
        idea["confident"] = bool(idea.get("confident", False) and survived)
        idea["fdr_threshold"] = cutoff

    return ideas


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Local so this module stays usable without the strategy stack."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def percentile_of_last(series: pd.Series, lookback: int = 252) -> Optional[float]:
    """
    Where the latest reading sits in this ticker's own trailing distribution.

    This is what makes "uncharacteristic" meaningful per-ticker: an RSI of 32 is
    the 3rd percentile for one name and the 30th for another, and only the
    percentile says which one is actually behaving strangely.
    """
    s = series.dropna()
    if len(s) < max(30, lookback // 5):
        return None
    window = s.tail(lookback)
    return float((window < window.iloc[-1]).mean() * 100.0)


def _dedupe_events(idx: pd.DatetimeIndex, min_gap: int) -> list:
    """
    Collapse a run of consecutive qualifying bars into single events.

    Without this an oversold stretch contributes one analog per bar — heavily
    overlapping windows that share most of their forward return. The sample size
    inflates, the confidence interval shrinks, and a setup looks far better
    established than it is.
    """
    kept, last = [], None
    for pos in idx:
        if last is None or (pos - last) >= min_gap:
            kept.append(pos)
            last = pos
    return kept


def rolling_percentile(feat: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Each bar's percentile within its OWN trailing window.

    Rolling rather than full-sample: judging a 2019 bar against the whole
    15-year distribution would leak the future into the comparison. Exposed
    separately because it is the expensive step — one `rolling.apply` over
    15 years per ticker — and every horizon reuses the same answer.
    """
    return feat.dropna().rolling(lookback).apply(
        lambda w: (w < w[-1]).mean() * 100.0, raw=True
    )


def find_analogs(
    feat: pd.Series,
    target_pct: float,
    horizon: int,
    tolerance: float = PERCENTILE_TOLERANCE,
    lookback: int = 252,
    roll_pct: Optional[pd.Series] = None,
) -> list:
    """
    Positions in history where `feat` sat at a similar percentile to today.

    Pass `roll_pct` when the caller already has the rolling percentiles for
    this exact series, to avoid recomputing them once per horizon.
    """
    s = feat.dropna()
    if len(s) < lookback + horizon + 5:
        return []

    if roll_pct is None:
        roll_pct = rolling_percentile(s, lookback)
    match = roll_pct.sub(target_pct).abs() <= tolerance

    # Drop the tail that has no room for a forward return yet.
    usable = match.iloc[:-horizon] if horizon < len(match) else match.iloc[0:0]
    positions = [s.index.get_loc(ts) for ts in usable[usable].index]
    return _dedupe_events(positions, min_gap=horizon)


def _forward(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def score_analogs(
    close: pd.Series,
    bench: pd.Series,
    analog_positions: list,
    horizon: int,
    max_examples: int = 6,
) -> Optional[AnalogStats]:
    """
    Turn a set of analog dates into an honest scorecard for one horizon.

    `hit_rate` is measured on abnormal (vs-benchmark) return; `base_rate` is the
    same statistic over every bar, so `lift` isolates what the SETUP contributed
    from what the ticker does anyway.
    """
    if not analog_positions:
        return None

    fwd = _forward(close, horizon)
    b = bench.reindex(close.index).ffill()
    bfwd = _forward(b, horizon)
    abn = (fwd - bfwd)

    take = [p for p in analog_positions if p < len(close) and pd.notna(abn.iloc[p])]
    if not take:
        return None

    a = abn.iloc[take]
    r = fwd.iloc[take]

    abn_all = abn.dropna()
    raw_all = fwd.dropna()

    # Examples span the full outcome range — showing only winners would turn a
    # coin flip into a sales pitch.
    order = a.sort_values(ascending=False)
    head = order.head(max(1, max_examples // 2))
    tail = order.tail(max(1, max_examples // 2))
    examples = []
    for ts, val in list(head.items()) + list(tail.items()):
        examples.append({
            "date": pd.Timestamp(ts).strftime("%Y-%m-%d"),
            "abnormal_return": round(float(val), 4),
            "raw_return": round(float(fwd.loc[ts]), 4) if pd.notna(fwd.loc[ts]) else None,
        })
    seen, uniq = set(), []
    for e in examples:
        if e["date"] not in seen:
            seen.add(e["date"])
            uniq.append(e)

    wins = int((a > 0).sum())
    n = int(len(a))
    hit = wins / n
    base = float((abn_all > 0).mean()) if len(abn_all) else float("nan")
    ci_low, ci_high = wilson_interval(wins, n)

    # One-sided test that mean abnormal return exceeds zero. Analog windows
    # overlap only minimally (events are spaced by the horizon), so a t-test is
    # a fair approximation; the standard error is what makes a thin sample
    # unable to claim confidence no matter how flattering its mean.
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    if n > 1 and np.isfinite(sd) and sd > 0:
        se = sd / np.sqrt(n)
        t_stat = mean / se
        try:
            from scipy import stats as _st
            p_val = float(_st.t.sf(t_stat, df=n - 1))
            t_crit = float(_st.t.ppf(1.0 - ALPHA_ONE_SIDED, df=n - 1))
        except Exception:  # scipy absent — fall back to the normal approximation
            p_val = float(0.5 * np.math.erfc(t_stat / np.sqrt(2)))
            t_crit = 1.645
        exp_low = mean - t_crit * se
    else:
        p_val, exp_low = float("nan"), float("-inf")

    return AnalogStats(
        horizon=horizon,
        n=n,
        hit_rate=hit,
        hit_ci_low=ci_low,
        hit_ci_high=ci_high,
        base_rate=base,
        lift=hit - base,
        expectancy=mean,
        expectancy_ci_low=exp_low,
        p_value=p_val,
        # The claim worth making: even at the pessimistic end of the interval,
        # the expected abnormal return is still positive.
        confident=bool(exp_low > 0.0),
        median=float(a.median()),
        worst=float(a.min()),
        best=float(a.max()),
        target_hits=int((r >= TARGET_RETURN).sum()),
        target_base_rate=float((raw_all >= TARGET_RETURN).mean()) if len(raw_all) else float("nan"),
        examples=uniq,
    )


def evaluate_ticker(
    df: pd.DataFrame,
    bench: pd.Series,
    ticker: str = "",
    lookback: int = 252,
    with_confirmations: bool = False,
) -> Optional[dict]:
    """
    Full evaluation for one ticker: how unusual is it right now, and what
    historically followed moments that looked like this?

    Returns None when there is not enough price history to say anything. A
    returned idea is NOT a recommendation — it is a description of a base rate
    with its sample size attached.

    With `with_confirmations`, the idea also carries the entry-confirmation
    checklist — what to watch for before acting — measured against these same
    analog moments. Requires `df` to hold real OHLCV, not just closes.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = df["Close"].dropna()
    if len(close) < lookback + max(HORIZONS) + 20:
        return None

    rsi = _rsi(close)
    rsi_pct = percentile_of_last(rsi, lookback)
    if rsi_pct is None:
        return None

    roll_pct = rolling_percentile(rsi, lookback)
    stats, setups = {}, {}
    for h in HORIZONS:
        positions = find_analogs(rsi, rsi_pct, h, lookback=lookback, roll_pct=roll_pct)
        s = score_analogs(close, bench, positions, h)
        if s is not None:
            stats[h] = s
            setups[h] = positions

    if not stats:
        return None

    primary = stats.get(HORIZONS[0]) or next(iter(stats.values()))
    validation = [s for h, s in stats.items() if h in VALIDATION_HORIZONS]
    rsi_pct_shown = round(rsi_pct, 1)
    idea = {
        "ticker": ticker,
        "rsi": round(float(rsi.iloc[-1]), 1),
        "rsi_percentile": rsi_pct_shown,
        "direction": "oversold" if rsi_pct_shown < 50 else "overbought",
        "last_close": round(float(close.iloc[-1]), 2),
        "stats": {h: s.as_dict() for h, s in stats.items()},
        "n_analogs": primary.n,
        "sufficient_history": primary.n >= MIN_ANALOGS,
        # Confidence rests on the trading horizons only, and every one of them
        # must clear its baseline: a setup that works at 5d but not 10d is not
        # something to be confident about. The long windows are context.
        "confident": (
            bool(validation)
            and all(s.confident for s in validation)
            and primary.n >= MIN_ANALOGS
        ),
        "expectancy": round(float(np.mean([s.expectancy for s in validation])), 4)
            if validation else 0.0,
        # Ranking key: mean CONSERVATIVE expectancy across horizons — the floor
        # of a one-sided interval, not the point estimate. A thin sample sinks
        # under its own standard error, so 3-for-3 ranks below 17-of-25, which
        # is the whole point when the ask is "confident the return is positive".
        "score": (
            float(np.mean([s.expectancy_ci_low for s in validation]))
            if validation and primary.n >= MIN_ANALOGS else float("-inf")
        ),
    }

    if with_confirmations:
        # Imported here, not at module scope: entry_confirmation builds on this
        # module's analog machinery, so a top-level import either way would be
        # circular. By the time this line runs, trade_ideas is fully defined.
        from entry_confirmation import attach_confirmations
        attach_confirmations(idea, df, bench, setups)

    return idea
