"""
Tests for the range calibration loop (range_calibration.py).

The calibration loop replays the evidence-based floor/ceiling engine over price
history (VIX as the point-in-time IV proxy), grades realized next-session
coverage against each sigma's Gaussian target, and tunes a single variance
multiplier out-of-sample. Everything here is pure/offline: hand-built OHLC + VIX
frames and small synthetic component frames in, dicts out — mirroring
test_track_record.py and test_fractal_structure.py.

The cornerstone is test_replay_band_matches_production_engine: it pins the
replay's band math to the real strategies.fractal_options._compute_floor_ceiling
so the two can never silently drift.
"""
import math

import numpy as np
import pandas as pd

from config import CONFIDENCE_SIGMAS
from strategies.fractal_options import _compute_floor_ceiling
import range_calibration as rc


# ── faithfulness guard ───────────────────────────────────────────────────────

def _neutral_dealer(spot):
    """Dealer inputs that never bind, so _compute_floor_ceiling = pure IV+VRP."""
    gex_bounds = {"gex_floor": None, "gex_ceiling": None}
    walls = {"strongest_put_wall": spot * 0.5, "strongest_call_wall": spot * 1.5}
    return gex_bounds, walls


def test_replay_band_matches_production_engine():
    spot, base_move, vrp_factor = 600.0, 6.0, 0.82
    for regime in ("trending", "transitional", "choppy"):
        for structure in ("contango", "backwardation", "flat", "unknown"):
            iv_range = {"daily_expected_move": base_move}
            vrp = {"scaling_factor": vrp_factor}
            vix_term = {"structure": structure}
            gex_bounds, walls = _neutral_dealer(spot)
            prod = _compute_floor_ceiling(
                spot, iv_range, vrp, vix_term, gex_bounds, walls, regime, 1.5)

            mine = rc._bands_for_row(
                spot, base_move, vrp_factor, regime, structure, rc.baseline_params())

            for label in CONFIDENCE_SIGMAS:
                assert mine[label]["floor"] == prod["ranges"][label]["floor"], (regime, structure, label)
                assert mine[label]["ceiling"] == prod["ranges"][label]["ceiling"], (regime, structure, label)


def test_width_multiplier_widens_symmetrically():
    spot, base_move = 100.0, 1.0
    p = rc.baseline_params()
    p["width_mult"] = 2.0
    bands = rc._bands_for_row(spot, base_move, 1.0, "transitional", "flat", p)
    # transitional=1.0, flat term=1.0, vrp=1.0 → final_move = base_move*2 = 2.0
    assert bands["1sigma"]["ceiling"] == 102.0 and bands["1sigma"]["floor"] == 98.0
    assert bands["2sigma"]["ceiling"] == 104.0 and bands["2sigma"]["floor"] == 96.0


# ── term structure ───────────────────────────────────────────────────────────

def test_term_structure_labels():
    assert rc._term_structure(14.0, 16.0) == "contango"       # ratio 0.875 < 0.95
    assert rc._term_structure(30.0, 20.0) == "backwardation"  # ratio 1.5 > 1.05
    assert rc._term_structure(20.0, 20.0) == "flat"           # ratio 1.0
    # No VIX3M → fall back to absolute VIX thresholds.
    assert rc._term_structure(12.0, None) == "contango"
    assert rc._term_structure(35.0, None) == "backwardation"
    assert rc._term_structure(22.0, None) == "flat"
    assert rc._term_structure(None, None) == "unknown"


# ── replay (point-in-time, no lookahead) ─────────────────────────────────────

def _synthetic_prices(n=140, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    steps = rng.normal(0, 1.0, n).cumsum()
    close = 600.0 + steps
    high = close + np.abs(rng.normal(0, 0.5, n))
    low = close - np.abs(rng.normal(0, 0.5, n))
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close}, index=idx)


def test_replay_components_is_causal_and_well_formed():
    price = _synthetic_prices(140)
    vix = pd.Series(20.0, index=price.index)
    comp = rc.replay_components(price, vix, horizon=1)

    warmup = max(63, 30) + 1
    assert len(comp) == len(price) - warmup - 1            # one row per eligible day
    for col in ("spot", "vix", "base_move", "vrp_factor", "regime", "structure", "next_close"):
        assert col in comp.columns
    assert set(comp["regime"]).issubset({"trending", "transitional", "choppy", "unknown"})

    # next_close is exactly the realized close one session later (no lookahead in
    # the predictor: bands depend only on spot/vix/vrp/regime as of the day).
    closes = price["Close"]
    first = comp.iloc[0]
    pos = closes.index.get_loc(first["date"])
    assert first["next_close"] == round(float(closes.iloc[pos + 1]), 2)

    # base_move equals the VIX-implied daily expected move (spot * iv * sqrt(1/365)).
    expected_bm = round(first["spot"] * (first["vix"] / 100.0) * math.sqrt(1 / 365.0), 2)
    assert abs(first["base_move"] - expected_bm) < 1e-9


def test_replay_insufficient_data_is_safe():
    price = _synthetic_prices(40)        # < warmup
    vix = pd.Series(20.0, index=price.index)
    assert rc.replay_components(price, vix).empty
    assert rc.grade(pd.DataFrame()).empty
    s = rc.summarize_calibration(pd.DataFrame())
    assert s["n_days"] == 0 and s["calibration_error"] is None


# ── grading + coverage summary ───────────────────────────────────────────────

def _components_from_moves(moves, *, spot=100.0, base_move=1.0):
    """Hand-built component frame: final_move == base_move (neutral factors)."""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=len(moves), freq="B"),
        "spot": spot,
        "vix": 20.0,
        "base_move": base_move,
        "vrp_factor": 1.0,
        "regime": "transitional",   # scale 1.0
        "structure": "flat",        # term 1.0
        "next_close": spot + np.asarray(moves, dtype=float),
    })


def test_grade_and_coverage_counts_in_band_days():
    # 1σ band half-width = 1.0 (base_move). Moves: 0 and ±0.5 land inside 1σ;
    # ±1.5 land outside 1σ but inside 2σ (half-width 2.0).
    comp = _components_from_moves([0.0, 0.5, -0.5, 1.5, -1.5])
    graded = rc.grade(comp)
    assert list(graded["in_1sigma"]) == [True, True, True, False, False]
    assert list(graded["in_2sigma"]) == [True, True, True, True, True]

    summ = rc.summarize_calibration(graded)
    assert summ["n_days"] == 5
    assert summ["coverage"]["1sigma"] == 60.0     # 3/5
    assert summ["coverage"]["2sigma"] == 100.0
    assert summ["mean_width_pct"] == 2.0          # (102-98)/100*100, 1σ width


# ── out-of-sample width sweep ────────────────────────────────────────────────

def test_sweep_widens_when_bands_undercover():
    # Realized moves ~ N(0, 1.2). With base_move=1.0 the band half-width at
    # width w, sigma s is w*s, so coverage matches every Gaussian target exactly
    # when w == 1.2. Baseline w=1.0 therefore systematically UNDER-covers, and an
    # out-of-sample sweep should push the multiplier up toward ~1.2.
    rng = np.random.default_rng(7)
    moves = rng.normal(0.0, 1.2, 400)
    comp = _components_from_moves(moves)
    # Drive the sweep off the precomputed components by faking the heavy step.
    out = _sweep_on_components(comp)

    assert "error" not in out
    assert 1.10 <= out["best_width"] <= 1.30
    assert out["improved"] is True
    # Baseline 1σ coverage should sit below its 68.3% target (under-covering).
    assert out["baseline_test"]["coverage"]["1sigma"] < 68.0


def test_sweep_keeps_unity_when_already_calibrated():
    # Moves ~ N(0, 1.0): the engine is already well calibrated, so the chosen
    # multiplier should hug 1.0.
    rng = np.random.default_rng(11)
    moves = rng.normal(0.0, 1.0, 400)
    comp = _components_from_moves(moves)
    out = _sweep_on_components(comp)
    assert 0.90 <= out["best_width"] <= 1.10


def test_sweep_insufficient_data_errors_cleanly():
    comp = _components_from_moves([0.1, -0.1, 0.2])
    out = _sweep_on_components(comp)
    assert "error" in out


def _sweep_on_components(comp, train_frac=0.7):
    """
    Exercise the sweep logic directly on a precomputed component frame by
    monkeypatching replay_components (the network/heavy half) to return it.
    Keeps the test fully offline.
    """
    orig = rc.replay_components
    rc.replay_components = lambda *a, **k: comp
    try:
        return rc.sweep_parameters(None, None, None, train_frac=train_frac)
    finally:
        rc.replay_components = orig
