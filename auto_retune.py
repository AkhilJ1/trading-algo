"""
Auto-Retune Engine — Safe optimization of bias weights and regime parameters.
------------------------------------------------------------------------------
Floor/ceiling is now computed via the evidence-based IV+VRP pipeline
(not from signal weights). This engine tunes:
  - Signal weights for BIAS VOTING only
  - Regime scaling factors (via learn_regime_adjustments)

Safety rails:
  - Minimum 60 days of data before retuning
  - Rolling 90-day window for learning
  - Max 15% change per weight per cycle
  - Holdout validation (last 20%) must pass
  - Accuracy must stay >= 65%
  - All changes logged to Google Sheets
"""

from datetime import date, timedelta
from typing import Tuple

import numpy as np
import pandas as pd

from config import (
    SIGNAL_WEIGHTS,
    RETUNE_MIN_DAYS,
    RETUNE_ROLLING_WINDOW,
    RETUNE_MAX_WEIGHT_CHANGE,
    RETUNE_MIN_ACCURACY,
    RETUNE_HOLDOUT_FRACTION,
    RETUNE_MIN_WEIGHT,
    RETUNE_MAX_WEIGHT,
)
from data_fetcher import fetch_stock_data


def check_retune_eligibility(pred_df: pd.DataFrame) -> Tuple[bool, str]:
    """Check if there is enough data to retune."""
    if pred_df.empty:
        return False, "No predictions recorded yet."

    unique_dates = pred_df['date'].dropna().nunique()
    if unique_dates < RETUNE_MIN_DAYS:
        return False, f"Need {RETUNE_MIN_DAYS} days of predictions, have {unique_dates}."

    return True, f"{unique_dates} days of predictions available."


def score_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each historical prediction against actual next-day close.

    Adds columns: actual_close, in_range, bias_correct, range_error
    """
    pred_df = pred_df.copy()
    pred_df['actual_close'] = np.nan
    pred_df['in_range'] = False
    pred_df['bias_correct'] = False
    pred_df['range_error'] = 0.0

    # Group by ticker to minimize data fetches
    for ticker in pred_df['ticker'].unique():
        mask = pred_df['ticker'] == ticker
        ticker_preds = pred_df[mask]

        df = fetch_stock_data(ticker, period='1y')
        if df.empty:
            continue
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        for idx in ticker_preds.index:
            pred_date = pred_df.loc[idx, 'date']
            if pd.isna(pred_date):
                continue

            # Find next trading day's close
            future = df[df.index > pred_date]
            if len(future) == 0:
                continue

            actual = float(future['Close'].iloc[0])
            floor_val = pred_df.loc[idx, 'floor']
            ceil_val = pred_df.loc[idx, 'ceiling']
            spot = pred_df.loc[idx, 'spot_price']
            bias = pred_df.loc[idx, 'bias']

            in_range = floor_val <= actual <= ceil_val

            # Range error: how far outside the range (0 if inside)
            if actual < floor_val:
                range_error = (floor_val - actual) / spot * 100
            elif actual > ceil_val:
                range_error = (actual - ceil_val) / spot * 100
            else:
                range_error = 0.0

            bias_correct = (
                (bias == 'BULLISH' and actual > spot) or
                (bias == 'BEARISH' and actual < spot) or
                (bias == 'NEUTRAL')
            )

            pred_df.loc[idx, 'actual_close'] = actual
            pred_df.loc[idx, 'in_range'] = in_range
            pred_df.loc[idx, 'bias_correct'] = bias_correct
            pred_df.loc[idx, 'range_error'] = range_error

    # Drop rows where we couldn't score
    scored = pred_df.dropna(subset=['actual_close'])
    return scored


def compute_signal_accuracy(
    scored_df: pd.DataFrame,
    current_weights: dict,
) -> dict:
    """
    Score each signal's contribution via leave-one-out analysis.

    For each signal: zero out its weight, redistribute, recompute
    floor/ceiling, measure accuracy change.

    Positive score = signal improves accuracy (keep/increase).
    Negative score = signal hurts accuracy (decrease).
    """
    if scored_df.empty:
        return {name: 0.0 for name in current_weights}

    baseline_acc = _range_accuracy(scored_df)
    signal_scores = {}

    for signal_name in current_weights:
        # Create weights with this signal zeroed out
        modified = dict(current_weights)
        removed_weight = modified[signal_name]
        modified[signal_name] = 0.0

        # Redistribute removed weight proportionally to remaining signals
        remaining_total = sum(v for k, v in modified.items() if v > 0)
        if remaining_total > 0:
            for k in modified:
                if modified[k] > 0:
                    modified[k] += removed_weight * (modified[k] / remaining_total)

        # Recompute floor/ceiling with modified weights
        acc_without = _recompute_accuracy(scored_df, modified)

        # If accuracy drops when we remove the signal, it's valuable
        signal_scores[signal_name] = round(baseline_acc - acc_without, 2)

    return signal_scores


def _range_accuracy(scored_df: pd.DataFrame) -> float:
    """Fraction of predictions where actual close was in [floor, ceiling]."""
    if scored_df.empty:
        return 0.0
    return scored_df['in_range'].sum() / len(scored_df) * 100


def _recompute_accuracy(scored_df: pd.DataFrame, weights: dict) -> float:
    """
    Recompute floor/ceiling using different weights and measure accuracy.

    Uses a simplified model: scales the original floor/ceiling proportionally
    based on the weight changes, rather than re-running the full analysis pipeline.
    """
    if scored_df.empty:
        return 0.0

    baseline_w = SIGNAL_WEIGHTS
    correct = 0

    for _, row in scored_df.iterrows():
        spot = row['spot_price']
        floor_orig = row['floor']
        ceil_orig = row['ceiling']
        actual = row['actual_close']

        if pd.isna(actual) or pd.isna(spot) or spot == 0:
            continue

        # Estimate how floor/ceiling would shift with new weights
        # The range center stays at spot, but width scales with weight emphasis
        range_width = ceil_orig - floor_orig
        center = (floor_orig + ceil_orig) / 2

        # Compute effective range scaling based on weight redistribution
        # Signals that contribute to tighter ranges (iv_range, fractals) vs wider
        # This is a linearized approximation
        iv_scale = weights.get('iv_range', 0.2) / max(baseline_w.get('iv_range', 0.2), 0.01)
        frac_scale = weights.get('fractals', 0.15) / max(baseline_w.get('fractals', 0.15), 0.01)
        wall_scale = weights.get('options_walls', 0.25) / max(baseline_w.get('options_walls', 0.25), 0.01)

        # Weighted average of scale factors
        avg_scale = (iv_scale * 0.4 + frac_scale * 0.3 + wall_scale * 0.3)
        new_width = range_width * avg_scale

        new_floor = center - new_width / 2
        new_ceil = center + new_width / 2

        if new_floor <= actual <= new_ceil:
            correct += 1

    return correct / len(scored_df) * 100 if len(scored_df) > 0 else 0.0


def propose_new_weights(
    current_weights: dict,
    signal_scores: dict,
) -> dict:
    """
    Propose adjusted weights based on signal accuracy scores.

    1. Rank signals by accuracy contribution
    2. Increase/decrease proportionally, clamped to max 15% change
    3. Normalize to sum to 1.0
    """
    proposed = dict(current_weights)

    scores = np.array([signal_scores.get(k, 0) for k in proposed])
    if scores.std() == 0:
        return proposed  # No differentiation

    # Normalize scores to [-1, 1]
    max_abs = max(abs(scores.max()), abs(scores.min()), 0.01)
    normed = scores / max_abs

    names = list(proposed.keys())
    for i, name in enumerate(names):
        # Scale adjustment by max allowed change
        adjustment = normed[i] * RETUNE_MAX_WEIGHT_CHANGE * proposed[name]
        new_val = proposed[name] + adjustment

        # Clamp to absolute bounds
        new_val = max(RETUNE_MIN_WEIGHT, min(RETUNE_MAX_WEIGHT, new_val))
        proposed[name] = new_val

    # Normalize to sum to 1.0
    total = sum(proposed.values())
    if total > 0:
        proposed = {k: v / total for k, v in proposed.items()}

    return proposed


def validate_on_holdout(
    scored_df: pd.DataFrame,
    baseline_weights: dict,
    proposed_weights: dict,
) -> Tuple[bool, float, float]:
    """
    Validate proposed weights on holdout set (last 20%).
    Only apply if proposed >= baseline AND >= min accuracy.
    """
    if len(scored_df) < 10:
        return False, 0.0, 0.0

    holdout_start = int(len(scored_df) * (1 - RETUNE_HOLDOUT_FRACTION))
    holdout = scored_df.iloc[holdout_start:]

    if len(holdout) < 5:
        return False, 0.0, 0.0

    baseline_acc = _recompute_accuracy(holdout, baseline_weights)
    proposed_acc = _recompute_accuracy(holdout, proposed_weights)

    should_apply = (
        proposed_acc >= baseline_acc and
        proposed_acc >= RETUNE_MIN_ACCURACY * 100
    )

    return should_apply, round(baseline_acc, 1), round(proposed_acc, 1)


def learn_regime_adjustments(scored_df: pd.DataFrame) -> dict:
    """
    Learn which VIX regimes allow tighter ranges.

    Returns per-regime accuracy, average range width, and optimal scaling.
    """
    if scored_df.empty or 'regime' not in scored_df.columns:
        return {}

    results = {}
    for regime in scored_df['regime'].dropna().unique():
        if not regime or regime == '':
            continue
        subset = scored_df[scored_df['regime'] == regime]
        if len(subset) < 5:
            continue

        acc = _range_accuracy(subset)
        avg_width = (subset['ceiling'] - subset['floor']).mean()
        avg_spot = subset['spot_price'].mean()
        width_pct = avg_width / avg_spot * 100 if avg_spot > 0 else 0

        # If accuracy is high, we can tighten; if low, widen
        if acc > 75:
            scale = 0.85  # Tighten by 15%
        elif acc > 65:
            scale = 1.0   # Keep as is
        elif acc > 55:
            scale = 1.15  # Widen by 15%
        else:
            scale = 1.30  # Widen by 30%

        results[regime] = {
            'accuracy': round(acc, 1),
            'avg_range_width_pct': round(width_pct, 2),
            'range_scale': scale,
            'count': len(subset),
        }

    return results


def run_retune(pred_df: pd.DataFrame, current_weights: dict) -> dict:
    """
    Master retune function. Orchestrates the full pipeline.
    Returns results dict for dashboard display. Does NOT auto-apply.
    """
    result = {
        "eligible": False,
        "reason": "",
        "signal_scores": None,
        "proposed_weights": None,
        "holdout_passed": False,
        "baseline_accuracy": 0.0,
        "proposed_accuracy": 0.0,
        "regime_adjustments": None,
        "applied": False,
        "weight_changes": [],
    }

    # Step 1: Eligibility
    eligible, reason = check_retune_eligibility(pred_df)
    result["reason"] = reason
    if not eligible:
        return result
    result["eligible"] = True

    # Step 2: Rolling window
    cutoff = pd.Timestamp(date.today() - timedelta(days=RETUNE_ROLLING_WINDOW))
    window_df = pred_df[pred_df["date"] >= cutoff].copy()
    if len(window_df) < 20:
        result["reason"] = f"Only {len(window_df)} predictions in the last {RETUNE_ROLLING_WINDOW} days."
        result["eligible"] = False
        return result

    # Step 3: Score predictions
    scored = score_predictions(window_df)
    if scored.empty:
        result["reason"] = "Could not score any predictions (no subsequent price data)."
        result["eligible"] = False
        return result

    current_acc = _range_accuracy(scored)
    result["baseline_accuracy"] = round(current_acc, 1)

    # Step 4: Per-signal accuracy
    signal_scores = compute_signal_accuracy(scored, current_weights)
    result["signal_scores"] = signal_scores

    # Step 5: Propose new weights
    proposed = propose_new_weights(current_weights, signal_scores)
    result["proposed_weights"] = proposed

    # Step 6: Holdout validation
    passed, base_acc, prop_acc = validate_on_holdout(
        scored, dict(SIGNAL_WEIGHTS), proposed
    )
    result["holdout_passed"] = passed
    result["baseline_accuracy"] = base_acc
    result["proposed_accuracy"] = prop_acc

    # Step 7: Regime adjustments
    result["regime_adjustments"] = learn_regime_adjustments(scored)

    # Step 8: Build change list
    if passed:
        changes = []
        for name in current_weights:
            if name in proposed and abs(proposed[name] - current_weights[name]) > 0.001:
                changes.append((name, current_weights[name], proposed[name]))
        result["weight_changes"] = changes

    return result
