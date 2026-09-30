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


def _straight_channel(**kw):
    """A straight channel (flow east along y = 1500 m, 294 m wide, 14 m deep, top at z = 20 m)
    on an empty 4 x 3 km grid of 20 m x 1 m cells; with the columns of the section x = 2 km
    north of its bank and their distance beyond the bank."""
    layer = ChannelLayer(nx=200, ny=150, nz=60, x_len=4000.0, y_len=3000.0, z_len=60.0, top_depth=0.0)
    layer.create_geology(seed=1, nlevel=1, ntime=1, mCHdepth=14.0, mCHwdratio=21.0, stdevCHdepth=0.0,
                         stdevCHdepth2=0.0, stdevCHwdratio=0.0, probAvulOutside=0.0, probAvulInside=0.0, **kw)
    eng = layer._engine
    eng.cx, eng.cy = np.linspace(-500.0, 4500.0, 500), np.full(500, 1500.0)
    eng.chelev, eng.chelev_arr = 20.0, np.full(500, 20.0)
    eng._chwidth_arr, eng._chwidth_state_n = np.full(500, 147.0), 500
    eng.cal_curv()
    y = (np.arange(150) + 0.5) * 20.0
    north = y > 1647.0
    return eng, north, y[north] - 1647.0


@pytest.mark.parametrize("width", [1200.0, 3000.0])
def test_continuous_banks_carry_the_channel_wall_into_the_levee(width):
    """With ``continuous_banks`` a levee starts at the bank with the channel wall's own slope
    and rounds over into its crest, the same distance out however wide the levee; by default
    its crest sits a sixth of ``LVwidth`` out, whatever the channel's wall."""
    from resmill.layers._fluvial import FF, LV
    from resmill.layers._genchannel import bank_slope
    s = bank_slope(0.5, 14.0, 294.0, False)
    dc = np.e * (25.0 / np.e) / s                          # crest distance for a crest of 25/e m
    for kw in ({}, {"continuous_banks": True}):
        eng, north, d = _straight_channel(**kw)
        eng.facies[:] = FF
        eng._stamp_levee(0.0, width, 25.0, 0.0, 0.0)
        thick = (eng.facies[100][north] == LV).sum(axis=1).astype(float)
        if not kw:
            assert thick[np.argmin(abs(d - width / 6.0))] == thick.max()
        else:
            near = d < 0.5 * dc
            assert np.allclose(thick[near], s * d[near] * np.exp(-d[near] / dc), atol=1.0)
            assert thick[np.argmin(abs(d - dc))] == thick.max()


def test_continuous_banks_trim_older_levees_along_the_wall():
    """The active channel clears the space above it: by default straight up from its banks,
    so an older levee beside it is cut vertically; with ``continuous_banks`` along its walls
    carried up past the banks, so that levee is trimmed to the wall's slope."""
    from resmill.layers._fluvial import CH, FF, LV
    from resmill.layers._genchannel import bank_slope
    s = bank_slope(0.5, 14.0, 294.0, False)
    for kw in ({}, {"continuous_banks": True}):
        eng, north, d = _straight_channel(**kw)
        eng.facies[:] = FF
        eng.facies[:, :, 20:30] = LV                           # an older levee, z = 20-30 m everywhere
        eng._stamp_channel(facies_code=CH, erode_above=True)
        lv = eng.facies[100][north] == LV
        top = np.where(lv.any(axis=1), lv.shape[1] - np.argmax(lv[:, ::-1], axis=1), 20).astype(float)
        want = np.minimum(30.0, 20.0 + s * d) if kw else np.full(d.size, 30.0)
        assert np.allclose(top, want, atol=1.0)


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


@pytest.mark.parametrize("buffer", [0.0, 600.0])
def test_path_buffer_draws_the_path_beyond_the_grid_at_both_ends(buffer):
    """With ``path_buffer`` a fresh channel path starts that many metres upstream
    of its entry and keeps going that far past the edge where it leaves the grid,
    with nodes added in proportion so their spacing stays the same; by default it
    starts and stops at the edge."""
    layer = ChannelLayer(nx=100, ny=50, nz=10, x_len=2000.0, y_len=1000.0, z_len=10.0, top_depth=0.0)
    layer.create_geology(seed=1, nlevel=1, ntime=1, probAvulOutside=0.0, probAvulInside=0.0,
                         **({"path_buffer": buffer} if buffer else {}))
    eng = layer._engine
    assert eng.ndis0 == round(200 * (1.0 + 2.0 * buffer / 2000.0))
    np.random.seed(3)
    assert eng.generate_streamline(x0=0.0, y0=500.0, chazi=90.0, chsinu=1.2) == 1
    assert eng.cx[0] == pytest.approx(-buffer)
    seg = np.hypot(np.diff(eng.cx), np.diff(eng.cy))
    outside = (eng.cx < 0.0) | (eng.cx > 2000.0) | (eng.cy < 0.0) | (eng.cy > 1000.0)
    first_in = int(np.argmin(outside))
    first_out = first_in + int(np.argmax(outside[first_in:])) if outside[first_in:].any() else eng.cx.size
    assert seg[:first_in].sum() == pytest.approx(buffer, abs=3 * eng.step)
    assert seg[first_out - 1:].sum() == pytest.approx(buffer, abs=2 * eng.step)


def test_path_buffer_keeps_both_ends_outside():
    """Migrating a path drawn beyond the grid leaves both its ends outside, so the
    channel neither starts nor stops inside the volume."""
    layer = ChannelLayer(nx=100, ny=50, nz=10, x_len=2000.0, y_len=1000.0, z_len=10.0, top_depth=0.0)
    layer.create_geology(seed=2, nlevel=1, ntime=30, probAvulOutside=0.0, probAvulInside=0.0,
                         mCHdepth=5.0, mCHwdratio=15.0, mdistMigrate=15.0, NTGtarget=0.99, path_buffer=400.0)
    eng = layer._engine
    for x, y in [(eng.cx[0], eng.cy[0]), (eng.cx[-1], eng.cy[-1])]:
        assert x < 0.0 or x > 2000.0 or y < 0.0 or y > 1000.0


def test_path_step_gives_the_same_channel_on_any_grid():
    """Alluvsim spaces a channel path's points one grid cell apart, and its bend rules
    count points (migration looks 30 upstream, curvature is smoothed over 10), so the same
    river bends differently on another grid. With ``path_step`` the points are that many
    metres apart whatever the cells, and one seed gives one channel on 20 m and 40 m cells
    (without splays, whose walk still takes one step per cell and so draws other numbers)."""
    eng = {}
    for cell in (20.0, 40.0):
        for kw in ({}, {"path_step": 10.0}):
            layer = ChannelLayer(nx=int(2000.0 / cell), ny=int(1000.0 / cell), nz=10, x_len=2000.0,
                                 y_len=1000.0, z_len=10.0, top_depth=0.0)
            layer.create_geology(seed=1, nlevel=1, ntime=20, probAvulOutside=0.0, probAvulInside=0.0,
                                 NTGtarget=0.99, mCHdepth=5.0, mCHwdratio=15.0, mdistMigrate=15.0,
                                 mCSnum=0.0, stdevCSnum=0.0, **kw)
            eng[cell, bool(kw)] = layer._engine
    assert (eng[20.0, False].step, eng[40.0, False].step) == (20.0, 40.0)
    assert eng[20.0, False].ndis0 != eng[40.0, False].ndis0
    assert eng[20.0, True].step == eng[40.0, True].step == 10.0
    assert eng[20.0, True].ndis0 == eng[40.0, True].ndis0 == 400
    np.testing.assert_allclose(eng[20.0, True].cx, eng[40.0, True].cx)
    np.testing.assert_allclose(eng[20.0, True].cy, eng[40.0, True].cy)


def test_facies_props_apply_to_their_layer_only():
    """A layer's ``facies_props`` override ResMill's table for that layer only: the module
    table, and so the next layer built without them, keep the defaults."""
    import copy
    from resmill.layers import channel as channel_module
    before = copy.deepcopy(channel_module.FACIES_PROPS)
    layer = ChannelLayer(nx=20, ny=15, nz=10, x_len=2000.0, y_len=1500.0, z_len=10.0, top_depth=0.0)
    layer.create_geology(seed=1, nlevel=1, ntime=2, probAvulOutside=0.0, probAvulInside=0.0,
                         facies_props={-1: {"log10_perm": -5.0}})
    assert np.allclose(np.asarray(layer.perm_mat)[np.asarray(layer.facies) == -1], 1e-5)
    assert channel_module.FACIES_PROPS == before


def test_rock_spread_slope_and_cap_are_opt_in():
    """``facies_props`` may give a facies a permeability spread at a given porosity
    (``log10_perm_sd``) and a relative porosity spread (``poro_sd``); ``perm_poro_slope`` ties
    sand permeability to porosity in decades per porosity unit around each facies' average;
    ``poro_max`` caps porosity softly."""
    layer = ChannelLayer(nx=60, ny=40, nz=12, x_len=1200.0, y_len=800.0, z_len=12.0, top_depth=0.0)
    layer.create_geology(seed=3, nlevel=2, ntime=20, probAvulOutside=0.0, probAvulInside=0.0,
                         mCHdepth=4.0, mCHwdratio=10.0, NTGtarget=0.9, perm_poro_slope=0.19, poro_max=0.40,
                         facies_props={4: {"log10_perm_sd": 0.4},
                                       -1: {"poro": 0.12, "log10_perm": -4.0, "log10_perm_sd": 0.5, "poro_sd": 0.25}})
    fac, poro = np.asarray(layer.facies), np.asarray(layer.poro_mat)
    log_k = np.log10(np.asarray(layer.perm_mat))
    ch = fac == 4
    slope, intercept = np.polyfit(100.0 * poro[ch], log_k[ch], 1)
    assert slope == pytest.approx(0.19, abs=0.03)
    assert np.std(log_k[ch] - (100.0 * slope * poro[ch] + intercept)) == pytest.approx(0.4, abs=0.08)
    assert np.mean(poro[ch]) == pytest.approx(0.30, abs=0.005)   # the facies value is its average porosity
    event_offset = np.asarray(layer._engine.log_perm_offset_field)[ch].mean()   # each body keeps its quality
    assert np.mean(log_k[ch]) == pytest.approx(3.3 + event_offset, abs=0.1)   # the facies value is its average
    mud = fac == -1
    assert np.std(log_k[mud]) == pytest.approx(0.5, abs=0.1)
    assert np.std(poro[mud]) == pytest.approx(0.03, abs=0.01) and poro[mud].min() > 0.0
    assert poro.max() < 0.40 and np.mean(poro > 0.399) < 0.001      # soft cap: no pile-up


def test_perm_max_bends_permeability_below_the_cap():
    """``perm_max`` (mD) bends log-permeability within half a decade of it smoothly towards
    it: nothing reaches the cap and nothing piles up just below it."""
    layer = ChannelLayer(nx=60, ny=40, nz=12, x_len=1200.0, y_len=800.0, z_len=12.0, top_depth=0.0)
    layer.create_geology(seed=3, nlevel=2, ntime=20, probAvulOutside=0.0, probAvulInside=0.0,
                         mCHdepth=4.0, mCHwdratio=10.0, NTGtarget=0.9, perm_max=1000.0)
    log_k = np.log10(np.asarray(layer.perm_mat))
    assert log_k.max() < 3.0
    assert np.mean(log_k > 2.99) < 0.001


def test_fining_amplitude_and_event_spreads_are_settings():
    """``fining_amplitude`` sets the fining-upward ramp (1 -+ it; Alluvsim's 0.3 by default) and
    ``event_poro_sd`` / ``event_log_perm_sd`` the spread of each flow event's rock."""
    stds = []
    for kw in ({}, {"fining_amplitude": 0.1, "event_poro_sd": 0.02, "event_log_perm_sd": 0.25}):
        layer = ChannelLayer(nx=60, ny=40, nz=12, x_len=1200.0, y_len=800.0, z_len=12.0, top_depth=0.0)
        layer.create_geology(seed=3, nlevel=2, ntime=20, probAvulOutside=0.0, probAvulInside=0.0,
                             mCHdepth=4.0, mCHwdratio=10.0, NTGtarget=0.9, **kw)
        stds.append(np.std(np.asarray(layer.poro_mat)[np.asarray(layer.facies) == 4]))
    assert stds[1] < 0.5 * stds[0]
    assert (layer._engine.poro_mult_std, layer._engine.log_perm_offset_std) == (0.02, 0.25)


def _bend_thalweg(radius, **kw):
    """How far the deepest point sits from the channel's centre (fraction of its
    width) along the middle of a 1.5 km right-hand bend of ``radius`` m (None:
    straight), for a 294 m wide channel. Flow starts east and turns south, so the
    heading never passes due north, where the engine's azimuth wraps."""
    layer = ChannelLayer(nx=200, ny=150, nz=10, x_len=4000.0, y_len=3000.0, z_len=10.0, top_depth=0.0)
    layer.create_geology(seed=1, nlevel=1, ntime=1, probAvulOutside=0.0, probAvulInside=0.0, **kw)
    eng = layer._engine
    if radius is None:
        cx, cy = 1000.0 + np.linspace(0.0, 1500.0, 400), np.full(400, 2500.0)
    else:
        th = np.linspace(0.0, 1500.0 / radius, 400)
        cx, cy = 1000.0 + radius * np.sin(th), 2500.0 - radius * (1.0 - np.cos(th))
    eng.cx, eng.cy = cx, cy
    eng.chelev_arr = np.full(400, 5.0)
    eng._chwidth_arr, eng._chwidth_state_n = np.full(400, 147.0), 400
    eng.cal_curv()
    return np.abs(eng.thalweg[eng.ndis // 4:3 * eng.ndis // 4] - 0.5)


def test_thalweg_max_follows_each_bends_own_tightness():
    """With ``thalweg_max`` each bend's asymmetry depends on its own tightness: a
    straight reach stays symmetric, a bend of radius 1.5 channel widths gets the full
    ``thalweg_max`` and a gentle one (10 widths) about a sixth of it. By default every
    bend is scaled by the sharpest point of its channel, so a lone bend always gets
    Alluvsim's maximum, 0.75, however gentle."""
    w = 294.0
    assert np.allclose(_bend_thalweg(None, thalweg_max=0.9), 0.0, atol=1e-6)
    assert np.allclose(_bend_thalweg(1.5 * w, thalweg_max=0.9), 0.4, atol=0.01)
    assert np.allclose(_bend_thalweg(10.0 * w, thalweg_max=0.9), 0.4 * 0.15, atol=0.01)
    assert np.allclose(_bend_thalweg(10.0 * w), 0.25, atol=0.01)


def test_unwrap_azimuth_keeps_curvature_through_due_north():
    """On a circular bend whose heading passes due north, smoothing the compass
    heading across its 360 -> 0 jump fakes a curvature spike of the wrong sign;
    with ``unwrap_azimuth=True`` the curvature stays constant, as on any circle."""
    def curvature(**kw):
        layer = ChannelLayer(nx=200, ny=150, nz=10, x_len=4000.0, y_len=3000.0, z_len=10.0, top_depth=0.0)
        layer.create_geology(seed=1, nlevel=1, ntime=1, probAvulOutside=0.0, probAvulInside=0.0, **kw)
        eng = layer._engine
        th = np.linspace(0.0, np.pi, 400)                 # east, through north, to west: a left turn
        eng.cx, eng.cy = 2000.0 + 500.0 * np.sin(th), 500.0 + 500.0 * (1.0 - np.cos(th))
        eng.chelev_arr = np.full(400, 5.0)
        eng._chwidth_arr, eng._chwidth_state_n = np.full(400, 147.0), 400
        eng.cal_curv()
        return eng.curv[eng.ndis // 4:3 * eng.ndis // 4]
    true = -np.degrees(1.0 / 500.0)                       # deg/m; negative: a left turn
    assert np.allclose(curvature(unwrap_azimuth=True), true, rtol=0.02)
    assert np.abs(curvature()).max() > 5 * abs(true)


def _s_bend_thalweg(**kw):
    """Thalweg position (> 0.5: deep side right of the flow) along an S-bend of a
    294 m wide channel with deepwater hydraulics: 900 m turning right, then 1500 m
    turning left, both of radius 600 m, nodes 12 m apart; and each resampled node's
    distance past the inflection."""
    layer = ChannelLayer(nx=200, ny=150, nz=10, x_len=4000.0, y_len=3000.0, z_len=10.0, top_depth=0.0)
    layer.create_geology(seed=1, nlevel=1, ntime=1, probAvulOutside=0.0, probAvulInside=0.0,
                         mCHdepth=14.0, mCHwdratio=21.0, Q=5000.0, **kw)
    eng = layer._engine
    s = np.arange(0.0, 2400.0, 12.0)
    heading = np.cumsum(np.where(s < 900.0, -1.0, 1.0) * 12.0 / 600.0)
    eng.cx = 500.0 + np.cumsum(12.0 * np.cos(heading))
    eng.cy = 1500.0 + np.cumsum(12.0 * np.sin(heading))
    eng.chelev_arr = np.full(s.size, 5.0)
    eng._chwidth_arr, eng._chwidth_state_n = np.full(s.size, 147.0), s.size
    eng.cal_curv()
    return eng.thalweg, eng.length - 888.0


@pytest.mark.parametrize("lag", [False, True])
def test_thalweg_lag_keeps_the_pool_where_the_channel_erodes(lag):
    """The migration model moves the channel by the curvature upstream (its memory),
    so just past an inflection the channel still erodes toward the previous bend's
    outer bank. With ``thalweg_lag=True`` the deep side follows the same memory and
    stays there; by default it flips at the inflection, leaving the point bar next
    to the pool. Well into the next bend both agree."""
    a, past = _s_bend_thalweg(thalweg_max=0.8, **({"thalweg_lag": True} if lag else {}))
    just_past = (past > 20.0) & (past < 110.0)
    far = (past > 700.0) & (past < 1000.0)
    assert np.all(a[just_past] > 0.5) == lag
    assert np.all(a[just_past] < 0.5) == (not lag)
    assert np.all(a[far] < 0.5)


def _fining_layer(**kw):
    layer = ChannelLayer(nx=60, ny=40, nz=16, x_len=1200.0, y_len=800.0, z_len=16.0, top_depth=0.0)
    layer.create_geology(seed=3, nlevel=2, ntime=20, probAvulOutside=0.0, probAvulInside=0.0,
                         mCHdepth=6.0, mCHwdratio=10.0, NTGtarget=0.9, perm_poro_slope=0.05, **kw)
    fac = np.asarray(layer.facies)
    return layer, fac, (fac == 3) | (fac == 4), np.asarray(layer._engine.depth_norm)


def test_fining_upward_is_a_permeability_drop_above_the_clean_base():
    """``fining_perm_decades`` lowers channel-fill (CH, LA) permeability linearly from the top of
    the fill's clean lower part (``fining_clean_fraction`` of its depth) to its top, where the
    drop is the full ``fining_perm_decades``; porosity follows by ``fining_poro_per_decade``
    units per decade, and each facies keeps its value as its average."""
    layer, fac, fill, dn = _fining_layer(fining_perm_decades=1.5, fining_clean_fraction=0.5)
    log_k, poro = np.log10(np.asarray(layer.perm_mat)), np.asarray(layer.poro_mat)
    offset = np.asarray(layer._engine.log_perm_offset_field)
    k = log_k - offset                                   # each body's own quality aside
    clean, top = fill & (dn >= 0.6), fill & (dn < 0.1)
    assert np.mean(k[clean & (fac == 3)]) - np.mean(k[top & (fac == 3)]) == pytest.approx(1.35, abs=0.15)
    lower = fill & (fac == 3) & (dn >= 0.5)             # flat in the clean part
    assert abs(np.polyfit(dn[lower], k[lower], 1)[0]) < 0.15
    for code, value in ((3, 2.7), (4, 3.3)):
        assert np.mean(k[fac == code]) == pytest.approx(value, abs=0.1)   # the facies value is its average
    assert np.mean(poro[clean & (fac == 3)]) - np.mean(poro[top & (fac == 3)]) == pytest.approx(0.0, abs=0.005)
    layer, fac, fill, dn = _fining_layer(fining_perm_decades=1.5, fining_clean_fraction=0.5,
                                         fining_poro_per_decade=4.0)
    poro = np.asarray(layer.poro_mat)
    clean, top = fill & (fac == 3) & (dn >= 0.6), fill & (fac == 3) & (dn < 0.1)
    assert np.mean(poro[clean]) - np.mean(poro[top]) == pytest.approx(0.04 * 1.5 * 0.9, abs=0.01)
    assert np.mean(poro[fac == 3]) == pytest.approx(0.25, abs=0.005)
    log_k = np.log10(np.asarray(layer.perm_mat)) - np.asarray(layer._engine.log_perm_offset_field)
    assert np.mean(log_k[clean]) - np.mean(log_k[top]) == pytest.approx(1.35, abs=0.15)   # the total drop


def test_fining_probability_picks_whole_storeys():
    """With ``fining_probability`` p, each channel storey (one aggradation level: every flow
    event of it) fines upward or stays blocky as a whole, about p of them fining."""
    layer = ChannelLayer(nx=60, ny=40, nz=36, x_len=1200.0, y_len=800.0, z_len=36.0, top_depth=0.0)
    layer.create_geology(seed=3, nlevel=6, ntime=8, ntime_per_level=True, probAvulOutside=0.0, probAvulInside=0.0,
                         mCHdepth=6.0, mCHwdratio=10.0, NTGtarget=0.9, perm_poro_slope=0.05,
                         fining_perm_decades=1.5, fining_clean_fraction=0.5, fining_probability=0.5)
    fac, eng = np.asarray(layer.facies), layer._engine
    fill = (fac == 3) | (fac == 4)
    dn, log_k = np.asarray(eng.depth_norm)[fill], np.log10(np.asarray(layer.perm_mat))[fill]
    pm, po = np.asarray(eng.poro_mult_field)[fill], np.asarray(eng.log_perm_offset_field)[fill]
    level = np.array([eng.event_levels[(a, b)] for a, b in zip(pm, po)])
    event = np.unique(np.stack([pm, po], axis=1), axis=0, return_inverse=True)[1].ravel()
    kinds = {}
    for e in np.unique(event):
        sel = event == e
        if (dn[sel] >= 0.5).sum() >= 3 and (dn[sel] < 0.2).sum() >= 3:
            drop = log_k[sel][dn[sel] >= 0.5].mean() - log_k[sel][dn[sel] < 0.2].mean()
            assert abs(drop) < 0.3 or drop > 0.8                          # blocky or fining, nothing between
            kinds.setdefault(int(level[sel][0]), set()).add(drop > 0.8)
    assert len(kinds) >= 4
    assert all(len(k) == 1 for k in kinds.values())                      # one choice per storey
    assert {True, False} <= set().union(*kinds.values())


def test_noise_ranges_in_metres_keep_the_same_texture_on_any_grid():
    """``noise_range_m`` gives the rock noise's horizontal and vertical correlation lengths in
    metres, so a finer grid shows the same texture: the correlation at a fixed distance does
    not depend on the cell size."""
    from resmill.layers.channel import _correlated_noise
    corr = []
    for dx in (10.0, 20.0):
        np.random.seed(0)
        noise = _correlated_noise((int(4000 / dx), 40, 8), 3.0, sigma=(120.0 / 3.46 / dx, 0.5, 0.2))
        lag = int(60 / dx)
        corr.append(np.corrcoef(noise[:-lag].ravel(), noise[lag:].ravel())[0, 1])
    assert corr[0] == pytest.approx(corr[1], abs=0.05)
    layer = ChannelLayer(nx=60, ny=40, nz=12, x_len=1200.0, y_len=800.0, z_len=12.0, top_depth=0.0)
    layer.create_geology(seed=3, nlevel=2, ntime=20, probAvulOutside=0.0, probAvulInside=0.0,
                         mCHdepth=4.0, mCHwdratio=10.0, NTGtarget=0.9, perm_poro_slope=0.05,
                         noise_range_m=(200.0, 0.5), facies_props={4: {"log10_perm_sd": 0.3}})
    log_k = np.log10(np.asarray(layer.perm_mat))
    fac = np.asarray(layer.facies)
    both = (fac[:-1] == 4) & (fac[1:] == 4)
    lateral = np.corrcoef(log_k[:-1][both], log_k[1:][both])[0, 1]
    both_z = (fac[:, :, :-1] == 4) & (fac[:, :, 1:] == 4)
    vertical = np.corrcoef(log_k[:, :, :-1][both_z], log_k[:, :, 1:][both_z])[0, 1]
    assert lateral > vertical + 0.2                                  # long thin beds


def _grdecl_array(path, keyword):
    """One keyword's values from a GRDECL file (plain numbers, no n* repeats)."""
    words = open(path).read().split()
    start = words.index(keyword) + 1
    return np.array([float(w) for w in words[start:words.index("/", start)]])


def test_kvkh_per_facies_sets_vertical_permeability(tmp_path):
    """A ``kvkh`` entry in ``facies_props`` gives each facies its own vertical-to-horizontal
    permeability ratio, which the GRDECL export writes as PERMZ = kvkh x PERMX; facies
    without one keep the layer's ``kzkx``."""
    from resmill.export import to_grdecl
    layer = ChannelLayer(nx=30, ny=20, nz=8, x_len=600.0, y_len=400.0, z_len=8.0, top_depth=0.0, kzkx=0.1)
    ratios = {4: 0.8, 3: 0.7, 2: 1e-3, -1: 0.2}
    layer.create_geology(seed=3, nlevel=2, ntime=10, probAvulOutside=0.0, probAvulInside=0.0,
                         mCHdepth=3.0, mCHwdratio=10.0, NTGtarget=0.9,
                         facies_props={code: {"kvkh": r} for code, r in ratios.items()})
    fac, kvkh = np.asarray(layer.facies), np.asarray(layer.kvkh_mat)
    for code in np.unique(fac):
        assert np.allclose(kvkh[fac == code], ratios.get(int(code), 0.1))
    to_grdecl(layer, tmp_path / "m.grdecl")
    permx, permz = _grdecl_array(tmp_path / "m.grdecl", "PERMX"), _grdecl_array(tmp_path / "m.grdecl", "PERMZ")
    expected = kvkh[:, :, ::-1].ravel(order="F")
    assert np.allclose(permz / permx, expected, rtol=1e-4)


def test_fining_tops_lower_kvkh_towards_the_top():
    """With ``fining_top_kvkh``, kv/kh in the fining part of a fill falls log-linearly from the
    facies value at the top of the clean part to ``fining_top_kvkh`` at the fill's top; blocky
    events and the clean parts keep the facies value."""
    layer, fac, fill, dn = _fining_layer(fining_perm_decades=1.5, fining_clean_fraction=0.5,
                                         fining_top_kvkh=0.05, facies_props={3: {"kvkh": 0.8}, 4: {"kvkh": 0.8}})
    kvkh = np.asarray(layer.kvkh_mat)
    assert np.allclose(kvkh[fill & (dn >= 0.5)], 0.8)                 # the clean part keeps the facies value
    top = fill & (dn < 0.05)
    assert np.all(kvkh[top] < 0.8 * (0.05 / 0.8) ** 0.85)                # near 0.05 at the very top
    mid = fill & (dn >= 0.24) & (dn < 0.26)                               # halfway up the fining part
    assert np.allclose(kvkh[mid], 0.8 * (0.05 / 0.8) ** 0.5, rtol=0.1)


def test_kvkh_is_off_by_default():
    """Without ``kvkh`` entries or ``fining_top_kvkh`` a layer has no kvkh_mat, and the export keeps
    PERMZ = kzkx x PERMX."""
    layer = ChannelLayer(nx=30, ny=20, nz=8, x_len=600.0, y_len=400.0, z_len=8.0, top_depth=0.0)
    layer.create_geology(seed=3, nlevel=2, ntime=10, probAvulOutside=0.0, probAvulInside=0.0,
                         mCHdepth=3.0, mCHwdratio=10.0, NTGtarget=0.9)
    assert layer.kvkh_mat is None
