"""
Tests for the level-snap floor/ceiling stage (strategies.fractal_options).

The primary floor/ceiling should snap toward the strongest *confluent*
evidence level (per-strike GEX, OI walls, neural zones, fractal pivots,
vectors) inside the snap window, while the multi-sigma `ranges` envelope
stays pure band math (guarded separately by test_range_calibration's
parity test). Everything here is pure/offline: hand-built frames in,
dicts out.
"""
import pandas as pd

from config import (
    LEVEL_SNAP_BLEND_BASE, LEVEL_SNAP_BLEND_STEP, LEVEL_SNAP_BLEND_MAX,
    LEVEL_SNAP_MIN_SIGMA, LEVEL_SNAP_MAX_SIGMA,
    CONFIDENCE_SIGMAS, PRIMARY_BAND_LABEL,
)
from strategies.fractal_options import (
    _compute_floor_ceiling, _collect_level_candidates, _snap_level,
    nearest_reaction_levels,
)


SPOT = 100.0
BASE_MOVE = 2.0   # with neutral factors below, final_move == 2.0
# The primary (Yellow Box) floor/ceiling starts from the PRIMARY_BAND_LABEL
# band edge (1.5σ → 97/103 here), not the 1σ band.
PRIM_SIGMA = CONFIDENCE_SIGMAS[PRIMARY_BAND_LABEL]
FLOOR_EDGE = SPOT - PRIM_SIGMA * BASE_MOVE
CEIL_EDGE = SPOT + PRIM_SIGMA * BASE_MOVE


def _neutral_inputs():
    """Factors that make final_move == base_move and a non-binding dealer overlay."""
    iv_range = {'daily_expected_move': BASE_MOVE, 'iv_used': 0.20}
    vrp = {'scaling_factor': 1.0}
    vix_term = {'structure': 'flat'}
    gex_bounds = {'gex_floor': None, 'gex_ceiling': None}
    walls = {'strongest_put_wall': SPOT * 0.5, 'strongest_call_wall': SPOT * 1.5}
    return iv_range, vrp, vix_term, gex_bounds, walls


def _gex_df(rows):
    """rows = [(strike, net_gex), ...]"""
    return pd.DataFrame([
        {'strike': s, 'call_gex': max(g, 0.0), 'put_gex': min(g, 0.0), 'net_gex': g}
        for s, g in rows
    ])


def _fc(**snap_kwargs):
    iv_range, vrp, vix_term, gex_bounds, walls = _neutral_inputs()
    walls = {**walls, **snap_kwargs.pop('walls_extra', {})}
    return _compute_floor_ceiling(
        SPOT, iv_range, vrp, vix_term, gex_bounds, walls,
        'transitional', 1.5, **snap_kwargs)


# ── no evidence → unchanged band ─────────────────────────────────────────────

def test_no_candidates_keeps_iv_band():
    out = _fc()
    assert out['floor'] == FLOOR_EDGE
    assert out['ceiling'] == CEIL_EDGE
    assert out['methodology']['floor_basis'] is None
    assert out['methodology']['ceiling_basis'] is None


def test_primary_band_is_the_yellow_box_band_not_1sigma():
    # Milk-RCG yellow box = ~80–90% containment → the primary floor/ceiling
    # starts from the 1.5σ (~86.6%) edge, not the 68% 1σ edge.
    out = _fc()
    assert out['floor'] == out['ranges'][PRIMARY_BAND_LABEL]['floor']
    assert out['ceiling'] == out['ranges'][PRIMARY_BAND_LABEL]['ceiling']
    assert out['floor'] != out['ranges']['1sigma']['floor']


def test_ranges_envelope_is_never_snapped():
    out = _fc(gex_df=_gex_df([(98.5, -5e9), (101.6, 4e9)]))
    # Primary floor/ceiling moved, the sigma envelope did not.
    assert out['ranges']['1sigma']['floor'] == 98.0
    assert out['ranges']['1sigma']['ceiling'] == 102.0
    assert out['ranges'][PRIMARY_BAND_LABEL]['floor'] == FLOOR_EDGE
    assert out['ranges'][PRIMARY_BAND_LABEL]['ceiling'] == CEIL_EDGE
    assert out['floor'] != FLOOR_EDGE
    assert out['ceiling'] != CEIL_EDGE


# ── single-source snap ───────────────────────────────────────────────────────

def test_single_gex_strike_pulls_floor_halfway():
    out = _fc(gex_df=_gex_df([(98.5, -5e9)]))
    expected = round(FLOOR_EDGE * (1 - LEVEL_SNAP_BLEND_BASE) + 98.5 * LEVEL_SNAP_BLEND_BASE, 2)
    assert out['floor'] == expected
    basis = out['methodology']['floor_basis']
    assert basis['sources'] == ['gex']
    assert basis['level'] == 98.5
    assert out['ceiling'] == CEIL_EDGE      # no candidates above spot


def test_ceiling_side_is_symmetric():
    out = _fc(gex_df=_gex_df([(101.6, 4e9)]))
    expected = round(CEIL_EDGE * (1 - LEVEL_SNAP_BLEND_BASE) + 101.6 * LEVEL_SNAP_BLEND_BASE, 2)
    assert out['ceiling'] == expected
    assert out['methodology']['ceiling_basis']['sources'] == ['gex']
    assert out['floor'] == FLOOR_EDGE


# ── confluence raises the snap weight ────────────────────────────────────────

def test_confluent_sources_snap_harder_than_one():
    one = _fc(gex_df=_gex_df([(98.5, -5e9)]))
    three = _fc(
        gex_df=_gex_df([(98.5, -5e9)]),
        walls_extra={'put_walls': [(98.5, 5000)], 'call_walls': []},
        neurals={'support_zones': [{'center': 98.45, 'strength': 3}],
                 'resistance_zones': []},
    )
    basis = three['methodology']['floor_basis']
    assert basis['sources'] == ['gex', 'neural', 'oi_wall']
    assert basis['blend'] == min(LEVEL_SNAP_BLEND_MAX,
                                 LEVEL_SNAP_BLEND_BASE + 2 * LEVEL_SNAP_BLEND_STEP)
    # Cluster center = mean of the three agreeing levels.
    center = round((98.5 + 98.5 + 98.45) / 3, 2)
    assert basis['level'] == center
    # More agreeing sources → the floor sits closer to the level.
    assert abs(three['floor'] - center) < abs(one['floor'] - 98.5)


# ── window discipline ────────────────────────────────────────────────────────

def test_levels_outside_snap_window_are_ignored():
    # final_move = 2.0 → floor window is [spot - 2.0*2, spot - 0.35*2] = [96.0, 99.3]
    too_close = SPOT - LEVEL_SNAP_MIN_SIGMA * BASE_MOVE + 0.2   # 99.5
    too_far = SPOT - LEVEL_SNAP_MAX_SIGMA * BASE_MOVE - 0.5     # 95.5
    out = _fc(gex_df=_gex_df([(too_close, -5e9), (too_far, -4e9)]))
    assert out['floor'] == FLOOR_EDGE
    assert out['methodology']['floor_basis'] is None


# ── candidate collection ─────────────────────────────────────────────────────

def test_collect_candidates_respects_sides_and_gates():
    gex = _gex_df([(98.5, -5e9), (101.5, 3e9)])
    walls = {'put_walls': [(97.0, 9000), (98.0, 100)],
             'call_walls': [(103.0, 8000)]}
    neurals = {
        'support_zones': [
            {'center': 97.5, 'strength': 3},
            {'center': 96.0, 'strength': 1},     # below strength gate → excluded
        ],
        'resistance_zones': [{'center': 102.5, 'strength': 4}],
    }
    fractal_levels = {
        'support_levels': [('d1', 98.2), ('d2', 101.0)],   # 101 is above spot → excluded
        'resistance_levels': [('d3', 101.8)],
    }
    vectors = {
        'support_vector': {'role': 'support', 'current_value': 97.8},
        'resistance_vector': {'role': 'support', 'current_value': 103.5},  # wrong role
    }

    floor_c = _collect_level_candidates(SPOT, 'floor', gex_df=gex, walls=walls,
                                        neurals=neurals, fractal_levels=fractal_levels,
                                        vectors=vectors)
    ceil_c = _collect_level_candidates(SPOT, 'ceiling', gex_df=gex, walls=walls,
                                       neurals=neurals, fractal_levels=fractal_levels,
                                       vectors=vectors)

    assert (98.5, 'gex') in floor_c
    assert (97.0, 'oi_wall') in floor_c and (98.0, 'oi_wall') in floor_c
    assert (97.5, 'neural') in floor_c
    assert (96.0, 'neural') not in floor_c
    assert (98.2, 'fractal') in floor_c
    assert (101.0, 'fractal') not in floor_c
    assert (97.8, 'vector') in floor_c

    assert (101.5, 'gex') in ceil_c
    assert (103.0, 'oi_wall') in ceil_c
    assert (102.5, 'neural') in ceil_c
    assert (101.8, 'fractal') in ceil_c
    assert all(src != 'vector' for _, src in ceil_c)   # flipped vector excluded


def test_snap_level_handles_degenerate_inputs():
    assert _snap_level(SPOT, 98.0, 0.0, [(98.5, 'gex')], 'floor') is None
    assert _snap_level(SPOT, 98.0, BASE_MOVE, [], 'floor') is None


# ── nearest reaction levels (first defended level, no distance window) ───────

def test_reaction_levels_pick_nearest_defended_either_side():
    gex = _gex_df([(99.8, -5e9), (97.0, -3e9), (100.4, 4e9), (103.0, 2e9)])
    rl = nearest_reaction_levels(SPOT, gex_df=gex)
    # Nearest defended levels win even though both sit far inside the snap
    # window's minimum distance — reaction levels have no window.
    assert rl['support']['level'] == 99.8
    assert rl['support']['sources'] == ['gex']
    assert rl['resistance']['level'] == 100.4
    assert rl['resistance']['distance'] == 0.4


def test_lone_fractal_pivot_is_not_defended():
    fl = {'support_levels': [('d', 99.7)], 'resistance_levels': []}
    rl = nearest_reaction_levels(SPOT, fractal_levels=fl)
    assert rl['support'] is None
    # ...but the same pivot clustered with a vector (2 sources) is defended.
    vec = {'support_vector': {'role': 'support', 'current_value': 99.68},
           'resistance_vector': None}
    rl2 = nearest_reaction_levels(SPOT, fractal_levels=fl, vectors=vec)
    assert rl2['support'] is not None
    assert rl2['support']['sources'] == ['fractal', 'vector']


def test_reaction_levels_empty_inputs_are_safe():
    rl = nearest_reaction_levels(SPOT)
    assert rl == {'support': None, 'resistance': None}


def test_reaction_level_at_spot_is_ignored():
    # A wall strike sitting exactly at spot is not a reaction level — the next
    # defended level out takes its place.
    walls = {'put_walls': [(100.0, 9000), (98.9, 5000)], 'call_walls': []}
    rl = nearest_reaction_levels(SPOT, walls=walls)
    assert rl['support']['level'] == 98.9
