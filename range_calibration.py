"""
Range Calibration Loop — is the predicted [floor, ceiling] band honest?

The evidence-based range engine (``strategies.fractal_options._compute_floor_ceiling``)
turns IV → VRP shrink → regime / VIX-term scaling into a multi-sigma
floor/ceiling. A 1-sigma band that *claims* ~68% coverage is only trustworthy
if, replayed over history, the next session's close actually landed inside it
~68% of the time. This module is the honest, out-of-sample calibration check —
and tuner — for that engine.

Scope & honesty note
--------------------
Only the historically *replayable* core is calibrated here:

    IV   — VIX as the point-in-time implied-vol proxy for SPY
    VRP  — realized / Parkinson vol vs IV (compute_vrp_ratio)
    regime — fractal dimension (calculate_fractal_dimension → classify_regime)
    term — VIX vs VIX3M term structure

The option-chain dealer overlay (GEX clusters, OI walls) cannot be back-replayed
without point-in-time chains, so it is held *neutral* (non-binding) in the
replay — exactly as ``_compute_floor_ceiling`` behaves when no dealer level sits
inside the IV band. This measures the engine's evidence-based backbone, not the
intraday dealer tightening. The dealer-pin *close* forecast (a separate,
chain-dependent estimate) is calibrated by the live track record
(track_record.py), not here.

Why a single width knob? With the dealer overlay neutral, every confidence band
is ``spot ± final_move · sigma``. The only honest, overfit-resistant calibration
control is a single variance multiplier on ``final_move`` (a "coverage
temperature"). We sweep that out-of-sample and *report* per-sigma coverage so
any tail-shape mismatch is visible rather than curve-fit away.

Everything except the __main__ driver is pure/offline: price+VIX frames in,
dicts/DataFrames out, mirroring walk_forward.py and track_record.py.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import (
    CONFIDENCE_SIGMAS,
    REGIME_SCALE,
    VIX_CONTANGO_SHRINK,
    VIX_BACKWARDATION_EXPAND,
    VRP_LOOKBACK_DAYS,
    VRP_MIN_RATIO,
    VRP_MAX_RATIO,
    FRACTAL_DIM_WINDOW,
)
from strategies.fractal_indicators import calculate_fractal_dimension, classify_regime
from strategies.fractal_options import compute_vrp_ratio

# Trading-day fraction used for the daily expected move, matching
# compute_iv_expected_move's daily_move = spot * iv * sqrt(1/365).
_DAILY_T = 1.0 / 365.0

# Theoretical two-sided coverage of a Gaussian at each confidence sigma, e.g.
# 1.0σ → 68.27%, 1.5σ → 86.64%, 2.0σ → 95.45%. The whole point of calibration is
# to check the realized coverage against these.
COVERAGE_TARGETS = {
    label: round((2 * norm.cdf(sigma) - 1) * 100, 2)
    for label, sigma in CONFIDENCE_SIGMAS.items()
}


def baseline_params() -> dict:
    """
    The production engine's settings as a tunable params dict. With these,
    ``_bands_for_row`` reproduces ``_compute_floor_ceiling`` (dealer overlay
    neutral) — guarded by test_range_calibration.
    """
    return {
        "regime_scale": dict(REGIME_SCALE),
        "contango": VIX_CONTANGO_SHRINK,
        "backwardation": VIX_BACKWARDATION_EXPAND,
        "width_mult": 1.0,
        "sigmas": dict(CONFIDENCE_SIGMAS),
    }


# ── point-in-time inputs ─────────────────────────────────────────────────────

def _term_structure(vix_spot, vix_3m) -> str:
    """
    Point-in-time VIX term-structure label, identical logic to
    compute_vix_term_structure() but fed historical values instead of a live
    fetch. 'contango' / 'backwardation' alter the band; 'flat' / 'unknown' don't.
    """
    if vix_spot and vix_3m and vix_3m > 0:
        ratio = vix_spot / vix_3m
        if ratio < 0.95:
            return "contango"
        if ratio > 1.05:
            return "backwardation"
        return "flat"
    if vix_spot:
        if vix_spot < 18:
            return "contango"
        if vix_spot > 28:
            return "backwardation"
        return "flat"
    return "unknown"


def _bands_for_row(spot, base_move, vrp_factor, regime, structure, params) -> dict:
    """
    The evidence-based band for one day, parameterized for sweeping.

    final_move = base_move · vrp_factor · regime_factor · term_factor · width_mult
    band(sigma) = spot ± final_move · sigma

    At baseline_params() this is exactly ``_compute_floor_ceiling`` with a
    non-binding dealer overlay (rounding to 2dp included).
    """
    regime_factor = params["regime_scale"].get(regime, 1.0)
    if structure == "contango":
        term_factor = params["contango"]
    elif structure == "backwardation":
        term_factor = params["backwardation"]
    else:
        term_factor = 1.0

    final_move = base_move * vrp_factor * regime_factor * term_factor * params["width_mult"]

    out = {}
    for label, sigma in params["sigmas"].items():
        move = final_move * sigma
        out[label] = {
            "floor": round(spot - move, 2),
            "ceiling": round(spot + move, 2),
            "move": round(move, 2),
        }
    return out


# ── historical replay ────────────────────────────────────────────────────────

def replay_components(
    price_df: pd.DataFrame,
    vix: pd.Series,
    vix3m: pd.Series | None = None,
    *,
    horizon: int = 1,
    warmup: int | None = None,
) -> pd.DataFrame:
    """
    Heavy, params-independent half of the replay: for every eligible day compute
    the inputs available *at that day's close* and the realized close `horizon`
    sessions later. Bands are NOT applied here — that is cheap and params-
    dependent, so it is split out (sweep_parameters reuses this frame).

    Parameters
    ----------
    price_df : OHLC frame with a DatetimeIndex (SPY).
    vix      : ^VIX close series (the point-in-time IV proxy), indexed by date.
    vix3m    : optional ^VIX3M close series for term structure.
    horizon  : sessions ahead to grade (1 = next-session close).
    warmup   : bars to skip before the first forecast (default covers the VRP
               lookback and the fractal-dimension window so every input is real).

    Returns one row per forecast day with: spot, vix, base_move, vrp_factor,
    regime, structure, next_close. No lookahead — fractal dimension is causal
    (uses closes strictly before the day) and VRP uses data through the day.
    """
    if price_df is None or price_df.empty or "Close" not in price_df.columns:
        return pd.DataFrame()

    df = price_df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    closes = df["Close"].astype(float)
    n = len(df)

    if warmup is None:
        warmup = max(VRP_LOOKBACK_DAYS, FRACTAL_DIM_WINDOW) + 1
    if n <= warmup + horizon:
        return pd.DataFrame()

    # Align VIX (+ VIX3M) onto the price calendar; forward-fill the rare gap.
    vix_a = pd.to_numeric(vix, errors="coerce")
    vix_a.index = pd.to_datetime(vix_a.index).tz_localize(None) if getattr(vix_a.index, "tz", None) is not None else pd.to_datetime(vix_a.index)
    vix_a = vix_a.reindex(df.index).ffill()
    if vix3m is not None and len(vix3m):
        v3 = pd.to_numeric(vix3m, errors="coerce")
        v3.index = pd.to_datetime(v3.index).tz_localize(None) if getattr(v3.index, "tz", None) is not None else pd.to_datetime(v3.index)
        v3 = v3.reindex(df.index).ffill()
    else:
        v3 = pd.Series(np.nan, index=df.index)

    # Fractal dimension once on the full series — causal, so no lookahead.
    fd_full = calculate_fractal_dimension(df).reindex(df.index)

    rows = []
    for i in range(warmup, n - horizon):
        spot = float(closes.iloc[i])
        vix_i = vix_a.iloc[i]
        if pd.isna(vix_i) or vix_i <= 0 or spot <= 0:
            continue

        iv_used = float(vix_i) / 100.0
        base_move = round(spot * iv_used * math.sqrt(_DAILY_T), 2)

        vrp = compute_vrp_ratio(
            df.iloc[: i + 1], iv_used,
            lookback=VRP_LOOKBACK_DAYS, min_ratio=VRP_MIN_RATIO, max_ratio=VRP_MAX_RATIO,
        )
        vrp_factor = float(vrp["scaling_factor"])

        fd_i = fd_full.iloc[i]
        regime = classify_regime(fd_i) if not pd.isna(fd_i) else "unknown"
        structure = _term_structure(float(vix_i), None if pd.isna(v3.iloc[i]) else float(v3.iloc[i]))

        rows.append({
            "date": df.index[i],
            "spot": round(spot, 2),
            "vix": round(float(vix_i), 2),
            "base_move": base_move,
            "vrp_factor": round(vrp_factor, 4),
            "regime": regime,
            "structure": structure,
            "next_close": round(float(closes.iloc[i + horizon]), 2),
        })

    return pd.DataFrame(rows)


def grade(components: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Cheap, params-dependent half: apply the bands and grade each day's realized
    close. Adds floor_/ceiling_/in_ columns per confidence sigma. Vectorized over
    the precomputed `components` frame.
    """
    if components is None or components.empty:
        return pd.DataFrame()
    params = params or baseline_params()
    out = components.copy()

    band_cols = {label: {"floor": [], "ceiling": [], "in": []} for label in params["sigmas"]}
    for _, r in out.iterrows():
        bands = _bands_for_row(
            r["spot"], r["base_move"], r["vrp_factor"], r["regime"], r["structure"], params)
        nc = r["next_close"]
        for label, b in bands.items():
            band_cols[label]["floor"].append(b["floor"])
            band_cols[label]["ceiling"].append(b["ceiling"])
            band_cols[label]["in"].append(bool(b["floor"] <= nc <= b["ceiling"]))

    for label, cols in band_cols.items():
        out[f"floor_{label}"] = cols["floor"]
        out[f"ceiling_{label}"] = cols["ceiling"]
        out[f"in_{label}"] = cols["in"]
    return out


def summarize_calibration(graded: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Reduce a graded replay to headline calibration stats: realized coverage per
    sigma vs its Gaussian target, the mean |coverage − target| gap (lower =
    better calibrated), and the mean 1-sigma band width as % of spot (how *tight*
    the honest range is). Empty-safe.
    """
    params = params or baseline_params()
    labels = list(params["sigmas"].keys())
    empty = {
        "n_days": 0, "coverage": {}, "targets": dict(COVERAGE_TARGETS),
        "calibration_error": None, "mean_width_pct": None, "primary_label": None,
    }
    if graded is None or graded.empty:
        return empty

    coverage, gaps = {}, []
    for label in labels:
        col = f"in_{label}"
        if col not in graded.columns:
            continue
        cov = round(float(graded[col].mean()) * 100, 2)
        coverage[label] = cov
        tgt = COVERAGE_TARGETS.get(label)
        if tgt is not None:
            gaps.append(abs(cov - tgt))

    primary = "1sigma" if "1sigma" in labels else (labels[0] if labels else None)
    mean_width_pct = None
    if primary and f"floor_{primary}" in graded.columns:
        width = (graded[f"ceiling_{primary}"] - graded[f"floor_{primary}"])
        mean_width_pct = round(float((width / graded["spot"] * 100).mean()), 3)

    return {
        "n_days": int(len(graded)),
        "coverage": coverage,
        "targets": {l: COVERAGE_TARGETS[l] for l in coverage if l in COVERAGE_TARGETS},
        "calibration_error": round(float(np.mean(gaps)), 3) if gaps else None,
        "mean_width_pct": mean_width_pct,
        "primary_label": primary,
    }


def replay_and_summarize(
    price_df, vix, vix3m=None, *, params=None, horizon=1,
) -> dict:
    """Convenience: replay + grade + summarize in one call (baseline by default)."""
    params = params or baseline_params()
    comp = replay_components(price_df, vix, vix3m, horizon=horizon)
    graded = grade(comp, params)
    summary = summarize_calibration(graded, params)
    summary["params"] = params
    return {"components": comp, "graded": graded, "summary": summary}


# ── out-of-sample width sweep ────────────────────────────────────────────────

def default_width_grid() -> list:
    """Variance multipliers to search (a 'coverage temperature')."""
    return [round(x, 2) for x in np.arange(0.80, 1.41, 0.05)]


def sweep_parameters(
    price_df, vix, vix3m=None, *,
    grid: list | None = None,
    train_frac: float = 0.70,
    horizon: int = 1,
) -> dict:
    """
    Anchored out-of-sample calibration of the single width multiplier.

    Computes the per-day components once, splits them chronologically into a
    train head (first `train_frac`) and a test tail, picks the width that
    minimizes calibration error *in-sample*, then reports how that choice does
    *out-of-sample*. Mirrors walk_forward.py's train/test discipline so we never
    grade a tuning choice on the data that chose it.

    Returns the baseline vs proposed coverage on the holdout, the selected
    multiplier, and whether it actually improved out-of-sample calibration
    (so callers can refuse to apply a non-improvement — like auto_retune).
    """
    grid = grid or default_width_grid()
    comp = replay_components(price_df, vix, vix3m, horizon=horizon)
    if comp.empty or len(comp) < 30:
        return {"error": "insufficient data for calibration sweep", "n_days": len(comp)}

    split = int(len(comp) * train_frac)
    split = max(15, min(split, len(comp) - 10))
    train, test = comp.iloc[:split], comp.iloc[split:]

    base = baseline_params()

    def _calib_err(components, width):
        p = dict(base); p["width_mult"] = width
        return summarize_calibration(grade(components, p), p)["calibration_error"]

    # Pick the in-sample-best width.
    train_scores = {w: _calib_err(train, w) for w in grid}
    train_scores = {w: e for w, e in train_scores.items() if e is not None}
    if not train_scores:
        return {"error": "could not score any width on the training window"}
    best_width = min(train_scores, key=train_scores.get)

    # Evaluate baseline vs best-width out-of-sample (the honest number).
    base_test = summarize_calibration(grade(test, base), base)
    prop_params = dict(base); prop_params["width_mult"] = best_width
    prop_test = summarize_calibration(grade(test, prop_params), prop_params)

    base_err = base_test["calibration_error"]
    prop_err = prop_test["calibration_error"]
    improved = (
        base_err is not None and prop_err is not None
        and prop_err < base_err - 1e-9 and abs(best_width - 1.0) > 1e-9
    )

    return {
        "n_days": int(len(comp)),
        "n_train": int(len(train)), "n_test": int(len(test)),
        "grid": grid,
        "best_width": best_width,
        "train_calibration_error": round(train_scores[best_width], 3),
        "baseline_test": base_test,
        "proposed_test": prop_test,
        "baseline_test_error": base_err,
        "proposed_test_error": prop_err,
        "improved": bool(improved),
    }


# ── reporting ────────────────────────────────────────────────────────────────

def print_calibration(summary: dict, title: str = "RANGE CALIBRATION") -> None:
    w = 60
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")
    if not summary or summary.get("n_days", 0) == 0:
        print("  No calibratable days.")
        print(f"{'='*w}")
        return
    print(f"  Days graded      : {summary['n_days']}")
    print(f"  Calibration err  : {summary.get('calibration_error')}  (mean |cov-target|, lower=better)")
    print(f"  Mean 1σ width    : {summary.get('mean_width_pct')}% of spot")
    print(f"{'-'*w}")
    print(f"  {'Sigma':<10}{'Realized':>12}{'Target':>12}{'Gap':>10}")
    for label, cov in summary.get("coverage", {}).items():
        tgt = summary.get("targets", {}).get(label)
        gap = f"{cov - tgt:+.1f}" if tgt is not None else "—"
        print(f"  {label:<10}{cov:>11.1f}%{(str(tgt)+'%'):>12}{gap:>10}")
    print(f"{'='*w}")


def print_sweep(sweep: dict) -> None:
    w = 60
    print(f"\n{'='*w}")
    print("  OUT-OF-SAMPLE WIDTH CALIBRATION")
    print(f"{'='*w}")
    if "error" in sweep:
        print(f"  {sweep['error']}")
        print(f"{'='*w}")
        return
    print(f"  Train / Test days : {sweep['n_train']} / {sweep['n_test']}")
    print(f"  Best width (in-sample) : {sweep['best_width']}x")
    print(f"{'-'*w}")
    print(f"  Baseline OOS calib err : {sweep['baseline_test_error']}")
    print(f"  Proposed OOS calib err : {sweep['proposed_test_error']}")
    verdict = "IMPROVES" if sweep["improved"] else "no improvement — keep 1.0x"
    print(f"  Verdict                : {verdict}")
    print(f"{'='*w}")


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    ticker = argv[0] if argv else "SPY"
    from data_fetcher import fetch_stock_data

    price = fetch_stock_data(ticker, period="2y")
    vix = fetch_stock_data("^VIX", period="2y")
    vix3m = fetch_stock_data("^VIX3M", period="2y")
    if price is None or price.empty:
        print(f"No price data for {ticker}.")
        return 1
    vix_s = vix["Close"] if vix is not None and not vix.empty else pd.Series(dtype=float)
    v3_s = vix3m["Close"] if vix3m is not None and not vix3m.empty else None

    res = replay_and_summarize(price, vix_s, v3_s)
    print_calibration(res["summary"], title=f"RANGE CALIBRATION — {ticker} (2y, baseline)")
    print_sweep(sweep_parameters(price, vix_s, v3_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
