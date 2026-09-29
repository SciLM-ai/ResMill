"""Smoke tests for the rewritten Alluvsim-faithful channel engine.

Validates basic shape/dtype/range invariants on the public output arrays
across both ``ChannelLayer`` and ``ChannelLayer`` for
binary and full-Alluvsim output modes. Statistical/visual parity vs the
Alluvsim binary lives in ``test_alluvsim_parity.py``.
"""
import numpy as np
import pytest

from resmill.layers.channel import (
    ChannelLayer, ChannelLayer,
    PV_SHOESTRING, CB_JIGSAW,
)


# Common small grid for fast iteration
GRID = dict(nx=64, ny=32, nz=16, x_len=640, y_len=320, z_len=8, top_depth=1000)


# === Public output shape & invariants ===

def test_channel_shapes():
    layer = ChannelLayer(**GRID)
    layer.create_geology(seed=0)
    assert layer.poro_mat.shape == (64, 32, 16)
    assert layer.perm_mat.shape == (64, 32, 16)
    assert layer.active.shape == (64, 32, 16)
    assert layer.facies.shape == (64, 32, 16)


def test_channel_has_nonzero_facies():
    layer = ChannelLayer(**GRID)
    layer.create_geology(seed=42)
    assert layer.active.sum() > 0, "engine should produce some sand"


def test_channel_poro_in_bounds():
    layer = ChannelLayer(**GRID)
    layer.create_geology(seed=1)
    assert (layer.poro_mat >= 0).all()
    assert (layer.poro_mat <= 1).all()


def test_channel_perm_positive_where_active():
    layer = ChannelLayer(**GRID)
    layer.create_geology(seed=2)
    assert (layer.perm_mat[layer.active == 1] > 0).all()


# === Facies output: 6-class Alluvsim codes everywhere ===

def test_facies_is_alluvsim_6class():
    """``layer.facies`` always has Alluvsim codes (-1..4); active is the
    binary 0/1 collapse derived from it."""
    layer = ChannelLayer(**GRID)
    layer.create_geology(seed=3)
    assert set(np.unique(layer.facies)).issubset({-1, 0, 1, 2, 3, 4})
    assert set(np.unique(layer.active)).issubset({0, 1})
    assert ((layer.active == (layer.facies >= 1).astype(np.int8))).all()
    # No legacy duplicate
    assert not hasattr(layer, "facies_alluvsim")


def test_alluvsim_codes_with_splays():
    """Turning on splays should make CS=1 appear in the facies array."""
    layer = ChannelLayer(**GRID)
    layer.create_geology(seed=4,
                         **{**PV_SHOESTRING, "mCSnum": 1.0, "stdevCSnum": 0.5,
                            "mCSnumlobe": 1.0, "stdevCSnumlobe": 0.5})
    codes = set(np.unique(layer.facies))
    assert -1 in codes  # FF
    assert 4 in codes   # CH


# === Preset constants ===

def test_pv_shoestring_preset():
    layer = ChannelLayer(**GRID)
    layer.create_geology(seed=0, **PV_SHOESTRING)
    # PV-shoestring is low-NTG; sand fraction should be modest. Loose
    # cap because per-event K-C mult draws consume RNG and shift NTG
    # slightly per realisation; 0.45 still discriminates from CB/SH.
    ntg = float(layer.active.mean())
    assert 0.01 < ntg < 0.45


def test_cb_jigsaw_preset_via_braided():
    layer = ChannelLayer(**GRID)
    layer.create_geology(seed=0)  # uses CB_JIGSAW defaults
    ntg = float(layer.active.mean())
    # CB-jigsaw is moderate-NTG; should hit > pv_shoestring
    assert ntg > 0.05


# === Reservoir composition (DeltaLayer back-compat sanity) ===

def test_reservoir_stacking_with_channel():
    from resmill.layers.gaussian import GaussianLayer
    from resmill.reservoir import Reservoir

    g = GaussianLayer(nx=64, ny=32, nz=8, x_len=640, y_len=320,
                      z_len=4, top_depth=1000)
    g.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03,
                     perm_std=0.5, ntg=0.7)

    c = ChannelLayer(nx=64, ny=32, nz=8, x_len=640, y_len=320,
                                z_len=4, top_depth=1004)
    c.create_geology(seed=0)

    res = Reservoir([g, c])
    assert res.poro_mat.shape == (64, 32, 16)


def test_channel_several_entry_points():
    """``n_sources=3``: three entry positions on the upstream edge, spaced by at
    least ``source_spacing_min`` of the edge; ``n_sources=1`` is the default and
    leaves the random stream untouched."""
    import resmill as rm
    from resmill.layers.channel import PV_SHOESTRING
    kw = dict(PV_SHOESTRING)
    a = rm.ChannelLayer(nx=64, ny=64, nz=32, x_len=640, y_len=640, z_len=32, top_depth=0)
    a.create_geology(seed=5, azimuth=0.0, n_sources=3, **kw)
    src = a._engine._sources
    assert len(src) == 3
    assert min(np.diff(sorted(src))) >= 0.15 * 640 - 1e-6
    b = rm.ChannelLayer(nx=64, ny=64, nz=32, x_len=640, y_len=640, z_len=32, top_depth=0)
    b.create_geology(seed=5, azimuth=0.0, **kw)
    c = rm.ChannelLayer(nx=64, ny=64, nz=32, x_len=640, y_len=640, z_len=32, top_depth=0)
    c.create_geology(seed=5, azimuth=0.0, n_sources=1, **kw)
    assert np.array_equal(b.active, c.active)
    assert c._engine._sources is None


def test_channel_porosity_noise_textures_sand_only():
    """poro_noise_std > 0 adds cell-scale texture to sand cells and leaves mud alone;
    0 reproduces the smooth-ramp output exactly."""
    from resmill.layers.channel import ChannelLayer, PV_SHOESTRING
    kw = dict(PV_SHOESTRING, seed=7)
    a = ChannelLayer(nx=40, ny=40, nz=24, x_len=400, y_len=400, z_len=24, top_depth=0.0)
    a.create_geology(**kw)
    b = ChannelLayer(nx=40, ny=40, nz=24, x_len=400, y_len=400, z_len=24, top_depth=0.0)
    b.create_geology(**kw, poro_noise_std=0.0)
    assert np.array_equal(a.poro_mat, b.poro_mat) and np.array_equal(a.perm_mat, b.perm_mat)
    c = ChannelLayer(nx=40, ny=40, nz=24, x_len=400, y_len=400, z_len=24, top_depth=0.0)
    c.create_geology(**kw, poro_noise_std=0.1, poro_noise_range=3.0)
    assert np.array_equal(a.facies, c.facies)
    sand = a.facies >= 1; mud = ~sand
    assert sand.sum() > 100
    assert np.array_equal(a.poro_mat[mud], c.poro_mat[mud])
    ratio = c.poro_mat[sand] / np.maximum(a.poro_mat[sand], 1e-6)
    assert 0.04 < ratio.std() < 0.2 and abs(ratio.mean() - 1.0) < 0.03
    # perm follows the porosity texture (K-C slope 3 in log10)
    lp = np.log10(c.perm_mat[sand]) - np.log10(a.perm_mat[sand])
    assert np.allclose(lp, 3.0 * np.log10(ratio), atol=1e-2)


def test_event_cap_scaling_is_opt_in():
    """``scale_ntime`` is off by default: the cap is the ``ntime`` given, so the
    published dataset is unchanged; switched on, a grid larger than the presets'
    800 x 800 m gets the cap scaled by sqrt(area / 800^2)."""
    import math
    kw = dict(PV_SHOESTRING, seed=3)
    grid = dict(nx=20, ny=20, nz=4, x_len=3200.0, y_len=1600.0, z_len=8.0, top_depth=0.0)
    off = ChannelLayer(**grid)
    off.create_geology(**kw)
    on = ChannelLayer(**grid)
    on.create_geology(**kw, scale_ntime=True)
    assert off._engine.ntime == PV_SHOESTRING["ntime"]
    assert on._engine.ntime == math.ceil(PV_SHOESTRING["ntime"] * math.sqrt(3200.0 * 1600.0 / 800.0 ** 2))


def test_scaled_event_cap_fills_field_size_layer():
    """With ``scale_ntime=True`` a field-size layer keeps depositing until its NTG
    target instead of being cut off with its upper levels still empty."""
    from resmill.layers.channel import SH_DISTAL
    layer = ChannelLayer(nx=40, ny=30, nz=12, x_len=4000.0, y_len=3000.0,
                         z_len=30.0, top_depth=0.0)
    layer.create_geology(seed=11, scale_ntime=True, **SH_DISTAL)
    sand = layer.active > 0
    assert sand.mean() > 0.9 * SH_DISTAL["NTGtarget"]
    assert sand[:, :, sand.shape[2] // 2:].mean() > 0.5 * SH_DISTAL["NTGtarget"]


def test_level_inherit_continues_the_previous_path(monkeypatch):
    """``level_inherit=1`` keeps each level on the previous channel's path, so with
    avulsion off the engine draws a path from its pool only once; by default it
    draws a fresh one at every level."""
    from resmill.layers import _fluvial
    draws = []
    original = _fluvial.fluvial._draw_from_pool

    def counting(self):
        draws.append(1)
        return original(self)

    monkeypatch.setattr(_fluvial.fluvial, "_draw_from_pool", counting)
    grid = dict(nx=24, ny=24, nz=8, x_len=480.0, y_len=480.0, z_len=8.0, top_depth=0.0)
    kw = dict(PV_SHOESTRING, seed=3, nlevel=4, ntime=2, ntime_per_level=True,
              probAvulOutside=0.0, probAvulInside=0.0)
    ChannelLayer(**grid).create_geology(**kw)
    redrawn = len(draws)
    draws.clear()
    ChannelLayer(**grid).create_geology(**kw, level_inherit=1.0)
    assert redrawn == 4
    assert len(draws) == 1


def test_ntime_can_be_given_per_level():
    """A list gives one event cap per level (with ``ntime_per_level``), scaled
    element-wise by ``scale_ntime``; a wrong length or a global cap is refused."""
    import math
    grid = dict(nx=20, ny=20, nz=6, x_len=3200.0, y_len=1600.0, z_len=6.0, top_depth=0.0)
    kw = dict(PV_SHOESTRING, seed=3, nlevel=3, ntime_per_level=True)
    layer = ChannelLayer(**grid)
    layer.create_geology(**{**kw, "ntime": [5, 2, 1]})
    assert layer._engine.ntime == [5, 2, 1]
    scaled = ChannelLayer(**grid)
    scaled.create_geology(**{**kw, "ntime": [5, 2, 1]}, scale_ntime=True)
    factor = math.sqrt(3200.0 * 1600.0 / 800.0 ** 2)
    assert scaled._engine.ntime == [math.ceil(n * factor) for n in (5, 2, 1)]
    with pytest.raises(ValueError):
        ChannelLayer(**grid).create_geology(**{**kw, "ntime": [5, 2]})
    with pytest.raises(ValueError):
        ChannelLayer(**grid).create_geology(**{**kw, "ntime": [5, 2, 1], "ntime_per_level": False})


def _right_hand_bend(**kw):
    """One channel, then its levee, stamped alone on a right-hand bend (flow east,
    turning south, radius 3 km, so the outer bank is north); the sections through
    the bend apex at x = 2 km and the cell-centre y of their columns."""
    from resmill.layers._fluvial import CH, FF
    layer = ChannelLayer(nx=200, ny=150, nz=60, x_len=4000.0, y_len=3000.0, z_len=60.0, top_depth=0.0)
    layer.create_geology(seed=1, nlevel=1, ntime=1, mCHdepth=14.0, mCHwdratio=21.0, stdevCHdepth=0.0,
                         stdevCHdepth2=0.0, stdevCHwdratio=0.0, probAvulOutside=0.0,
                         probAvulInside=0.0, **kw)
    eng = layer._engine
    th = np.radians(np.linspace(125.0, 55.0, 600))       # decreasing angle: clockwise, a right turn
    eng.cx, eng.cy = 2000.0 + 3000.0 * np.cos(th), -1500.0 + 3000.0 * np.sin(th)
    eng.chelev, eng.chelev_arr = 30.0, np.full(600, 30.0)
    eng._chwidth_arr, eng._chwidth_state_n = np.full(600, 147.0), 600
    eng.cal_curv()
    eng.facies[:] = FF
    eng._stamp_channel(facies_code=CH, erode_above=False)
    channel = eng.facies[100].copy()
    eng.facies[:] = FF
    eng._stamp_levee(2.0, 600.0, 6.0, 0.3, 0.0)
    return channel, eng.facies[100].copy(), (np.arange(150) + 0.5) * 20.0


@pytest.mark.parametrize("outer", [False, True])
def test_cutbank_side_of_a_bend(outer):
    """``cutbank_outer=True`` puts the channel's deepest point and the wider levee on
    the outer bank of a bend, the side the channel migrates to; the default keeps
    Alluvsim's rules, which put both on the inner bank."""
    from resmill.layers._fluvial import CH, LV
    channel, levee, y = _right_hand_bend(**({"cutbank_outer": True} if outer else {}))
    base = np.where((channel == CH).any(axis=1), np.argmax(channel == CH, axis=1), channel.shape[1])
    deepest_north = y[np.argmin(base)] > 1500.0
    reach = y[(levee == LV).any(axis=1)]
    levee_wider_north = reach.max() - 1500.0 > 1500.0 - reach.min()
    assert deepest_north == outer
    assert levee_wider_north == outer


def _arc(radius, degrees, spacing=10.0):
    """Points every ``spacing`` m along a circular arc, with their segment lengths."""
    th = np.arange(0.0, np.radians(degrees), spacing / radius)
    cx, cy = radius * np.sin(th), radius * (1.0 - np.cos(th))
    return cx, cy, np.r_[0.0, np.hypot(np.diff(cx), np.diff(cy))]


@pytest.mark.parametrize("ratio, bend_cut", [(1.0, True), (3.0, False)])
def test_cutoff_loop_ratio_keeps_bends_and_cuts_loops(ratio, bend_cut):
    """Alluvsim's neck-cutoff rule (loop ratio 1) also removes a plain 120-degree bend
    whose shortcut stays inside a 200 m wide channel; with a loop ratio of 3 only a
    real loop, several times longer than its neck, is cut off, however short it is."""
    from resmill.layers._make_cutoff import make_cutoff
    ctol = 300.0                                      # the engine's 3 half-widths
    cx, cy, dl = _arc(200.0, 120.0)
    assert (make_cutoff(cx, cy, dl, ctol, loop_ratio=ratio) < dl.size) == bend_cut
    cx, cy, dl = _arc(200.0, 330.0)
    assert make_cutoff(cx, cy, dl, ctol, loop_ratio=ratio) < dl.size
    cx, cy, dl = _arc(60.0, 330.0)                    # a small loop whose neck has closed
    assert make_cutoff(cx, cy, dl, ctol, loop_ratio=ratio) < dl.size


@pytest.mark.parametrize("extend", [False, True])
def test_extend_to_boundary_keeps_channels_flowing_out(extend):
    """A channel whose downstream end lies inside the grid after a migration step
    (here a path that stops mid-grid) is walked on until it leaves the grid with
    ``extend_to_boundary=True``; by default it keeps its dead end."""
    layer = ChannelLayer(nx=100, ny=50, nz=10, x_len=2000.0, y_len=1000.0, z_len=10.0, top_depth=0.0)
    layer.create_geology(seed=1, nlevel=1, ntime=1, probAvulOutside=0.0, probAvulInside=0.0,
                         **({"extend_to_boundary": True} if extend else {}))
    eng = layer._engine
    x = np.linspace(0.0, 1000.0, 200)                 # a gently sinuous path ending mid-grid
    eng.cx, eng.cy = x, 500.0 + 20.0 * np.sin(x / 150.0)
    eng.chelev_arr = np.full(200, 5.0)
    eng._chwidth_arr, eng._chwidth_state_n = np.full(200, 20.0), 200
    eng.cal_curv()
    assert eng._migrate_one_step(5.0) == 1
    to_edge = min(eng.cx[-1], 2000.0 - eng.cx[-1], eng.cy[-1], 1000.0 - eng.cy[-1])
    assert (to_edge < 2 * eng.step) == extend
