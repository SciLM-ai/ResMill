"""Fusion of channels that meet (``fuse_prob``): a path that reaches an older path of its own level joins it.

The pieces are tested alone first (the raster of older paths, the crossing, the bend), then through the engine: the crossings
of distinct channels of one level fall, those between levels stay, a level looks only at its own paths, nothing changes when it
is off, and the build is repeatable.
"""
import numpy as np
import pytest

from resmill.layers import _fluvial, _fusion
from resmill.layers.channel import ChannelLayer

from .channel_paths import PathLog, crossing_sites

STEP = 10.0
WIDTH = 40.0


def _belts():
    """A 100 x 100 raster of 10 m cells holding one older path: a gentle curve along +x at y = 500 m."""
    belts = _fusion.Belts(100, 100, 0.0, 0.0, 10.0, 10.0)
    x = np.arange(0.0, 1000.0, STEP)
    belts.add(x, 500.0 + 20.0 * np.sin(x / 300.0), x, 500.0 + 20.0 * np.sin(x / 300.0))
    return belts


def _line(angle_deg, through=(500.0, 500.0), back=400.0, ahead=300.0):
    """A straight path through ``through`` heading ``angle_deg`` degrees from +x, ``STEP`` m between its nodes."""
    a = np.radians(angle_deg)
    s = np.arange(-back, ahead, STEP)
    return through[0] + s * np.cos(a), through[1] + s * np.sin(a)


def test_burn_writes_the_cells_a_path_passes_through():
    owner, node = np.full((10, 10), -1, np.int32), np.zeros((10, 10), np.int32)
    _fusion.burn(owner, node, 3, np.array([5.0, 95.0]), np.array([5.0, 5.0]), 0.0, 0.0, 10.0, 10.0)
    assert (owner[:, 0] == 3).all() and (owner[:, 1:] == -1).all()
    assert (node == 0).all()


def test_a_crossing_is_found_where_it_is_with_the_angle_it_has():
    belts = _belts()
    x, y = _line(60.0)
    j, u, pid, s, v = belts.first_crossing(x, y, x, y, start=0)
    px, py = x[j] + u * (x[j + 1] - x[j]), y[j] + u * (y[j + 1] - y[j])
    ox, oy = belts.path(pid)
    assert (px, py) == pytest.approx((ox[s] + v * (ox[s + 1] - ox[s]), oy[s] + v * (oy[s + 1] - oy[s])), abs=1e-6)
    assert abs(px - 500.0) < 20.0 and abs(py - 500.0) < 20.0 and pid == 0


def test_a_path_running_along_an_older_one_does_not_cross_it():
    belts = _belts()
    x, y = _line(8.0, through=(500.0, 480.0))
    assert belts.first_crossing(x, y, x, y, start=0) is None


def test_only_the_path_from_its_start_node_on_is_searched():
    belts = _belts()
    x, y = _line(60.0)
    j = belts.first_crossing(x, y, x, y, start=0)[0]
    assert belts.first_crossing(x, y, x, y, start=j + 1) is None


def test_the_raster_is_read_in_the_frame_the_path_is_stamped_in():
    """Paths are kept as walked and found by the cells of their stamped image: rotate both and the crossing stays."""
    a = np.radians(30.0)
    rot = lambda x, y: (500.0 + (x - 500.0) * np.cos(a) + (y - 500.0) * np.sin(a),
                        500.0 - (x - 500.0) * np.sin(a) + (y - 500.0) * np.cos(a))
    belts = _fusion.Belts(100, 100, 0.0, 0.0, 10.0, 10.0)
    ox = np.arange(100.0, 900.0, STEP)
    oy = 500.0 + 20.0 * np.sin(ox / 300.0)
    belts.add(ox, oy, *rot(ox, oy))
    x, y = _line(60.0)
    assert belts.first_crossing(x, y, *rot(x, y), start=0) is not None


def test_a_fused_path_bends_into_the_older_one_and_follows_it():
    belts = _belts()
    x, y = _line(60.0)
    fx, fy = _fusion.fuse(belts, x, y, x, y, WIDTH, STEP, 0, 70.0)
    ox, oy = belts.path(0)
    assert (fx[0], fy[0]) == (x[0], y[0])                                    # the path upstream of the junction is its own
    assert (fx[-1], fy[-1]) == (ox[-1], oy[-1])                              # and downstream it is the older path
    tail = np.flatnonzero(np.isclose(fx, ox[-30]))
    assert tail.size and np.allclose(fy[tail[-1]:][:10], oy[-30:][:10])
    heading = np.arctan2(np.diff(fy), np.diff(fx))
    turns = np.abs(np.degrees(np.unwrap(heading)[3:] - np.unwrap(heading)[:-3]))
    assert turns.max() < 30.0                                                # no kink at the junction
    step = np.hypot(np.diff(fx), np.diff(fy))
    assert step.max() < 1.5 * STEP and step.min() > 0.3 * STEP


def _own_nodes(fx, fy, x, y):
    """How many leading nodes of the fused path are the new path's own."""
    n = min(fx.size, x.size)
    return int(np.argmax((fx[:n] != x[:n]) | (fy[:n] != y[:n])))


def test_the_bend_does_not_cross_the_older_path_before_it_joins():
    belts = _belts()
    x, y = _line(75.0)
    fx, fy = _fusion.fuse(belts, x, y, x, y, WIDTH, STEP, 0, 70.0)
    ox, oy = belts.path(0)
    gap = np.interp(fx, ox, oy) - fy
    joined = np.flatnonzero(np.abs(gap) < 1e-6)[0]
    assert (np.sign(gap[:joined]) == np.sign(gap[0])).all() and joined > _own_nodes(fx, fy, x, y)


@pytest.mark.parametrize("angle,leaves", [(50.0, (47.0, 53.0)), (88.0, (60.0, 78.0))])
def test_the_junction_angle_is_capped(angle, leaves):
    """A path meeting the older one at more than ``max_angle`` is turned towards its heading before the bend (a confluence
    joins at an acute angle): the older path runs at 3.8 degrees there, so a 88-degree path leaves at most at 74 (the first
    chord of the bend lies a little inside its tangent), a 50-degree one as it came."""
    belts = _belts()
    x, y = _line(angle)
    fx, fy = _fusion.fuse(belts, x, y, x, y, WIDTH, STEP, 0, 70.0)
    i = _own_nodes(fx, fy, x, y)
    assert leaves[0] < np.degrees(np.arctan2(fy[i] - fy[i - 1], fx[i] - fx[i - 1])) < leaves[1]


def test_nothing_to_join_is_none():
    belts = _belts()
    x, y = _line(8.0, through=(500.0, 480.0))
    assert _fusion.fuse(belts, x, y, x, y, WIDTH, STEP, 0, 70.0) is None


# === through the engine ===

GRID = dict(nx=64, ny=48, nz=8, x_len=1600.0, y_len=1200.0, z_len=12.0, top_depth=0.0)
RIVER = dict(nlevel=2, ntime=70, ntime_per_level=True, NTGtarget=0.99, probAvulOutside=0.25, probAvulInside=0.25,
             mCHdepth=3.0, mCHwdratio=16.0, stdevCHdepth=0.2, mCHsinu=1.25, mdistMigrate=8.0, stdevdistMigrate=2.0,
             path_step=8.0, stdevCHsource=150.0, mCSnum=0.0, mCSnumlobe=0.0, mLVwidth=10.0)
"""Two levels of 70 events on a 1.6 x 1.2 km model, 48 m wide channels, a quarter of the events avulsions out of the pool and a
quarter new tails: about 60 lineages a build, a second each."""


def _layer(seed, **kw):
    layer = ChannelLayer(**GRID)
    layer.create_geology(seed=seed, **{**RIVER, **kw})
    return layer


def _sites(monkeypatch, seeds, **kw):
    """Same-level and different-level crossing sites summed over ``seeds``."""
    total = {"same": 0, "different": 0}
    for seed in seeds:
        with monkeypatch.context() as m:
            log = PathLog(m)
            engine = _layer(seed, **kw)._engine
            for kind, found in crossing_sites(log, engine.mCHdepth * engine.mCHwdratio, engine.step).items():
                total[kind] += found[0].size
    return total


def test_fusion_removes_the_crossings_of_distinct_channels_of_one_level(monkeypatch):
    plain = _sites(monkeypatch, range(3))
    fused = _sites(monkeypatch, range(3), fuse_prob=1.0)
    assert plain["same"] > 60
    assert fused["same"] < 0.2 * plain["same"]
    assert fused["different"] > 0.1 * plain["different"]        # the crossings between generations stay
    half = _sites(monkeypatch, range(3), fuse_prob=0.5)
    assert fused["same"] < half["same"] < plain["same"]


def test_a_level_looks_only_at_its_own_paths(monkeypatch):
    """The first lineage of every level starts with no older path: the raster is new at each level."""
    seen = {}
    original = _fluvial.fluvial._begin_lineage

    def begin(engine, shared_m):
        seen.setdefault(engine._level, engine._belts.n_paths + len(engine._events))
        return original(engine, shared_m)

    monkeypatch.setattr(_fluvial.fluvial, "_begin_lineage", begin)
    _layer(0, nlevel=3, fuse_prob=1.0)
    assert seen == {0: 0, 1: 0, 2: 0}


def test_a_merging_lineage_ends_on_an_older_path(monkeypatch):
    """A path that joined follows an older one to its end, so its last node is the last node of a kept path."""
    joined, original = [], _fluvial.fluvial._merged

    def merged(engine, x, y, shared_m):
        out = original(engine, x, y, shared_m)
        if out[0] is not x:
            ends = [engine._belts.path(p)[0][-1] for p in range(engine._belts.n_paths)]
            joined.append(np.isclose(out[0][-1], ends).any())
        return out

    monkeypatch.setattr(_fluvial.fluvial, "_merged", merged)
    _layer(1, fuse_prob=1.0)
    assert len(joined) > 10 and all(joined)


def test_a_build_with_fusion_is_repeatable():
    a, b = _layer(4, fuse_prob=0.7), _layer(4, fuse_prob=0.7)
    assert np.array_equal(a.facies, b.facies) and np.array_equal(a.perm_mat, b.perm_mat)


def test_fusion_off_changes_nothing_and_draws_nothing():
    """``fuse_prob`` 0 is the default: same model and the same global random stream afterwards."""
    ChannelLayer(**GRID).create_geology(seed=2, **RIVER)                     # compile
    np.random.seed(0)
    ChannelLayer(**GRID).create_geology(seed=2, **RIVER)
    plain_state = np.random.get_state()
    plain = _layer(2)
    off = _layer(2, fuse_prob=0.0, fuse_max_angle=50.0)
    np.random.seed(0)
    ChannelLayer(**GRID).create_geology(seed=2, fuse_prob=0.0, **RIVER)
    off_state = np.random.get_state()
    assert np.array_equal(plain.facies, off.facies) and np.array_equal(plain.perm_mat, off.perm_mat)
    assert np.array_equal(plain_state[1], off_state[1]) and plain_state[2] == off_state[2]


def test_fusion_is_refused_where_it_means_nothing():
    with pytest.raises(ValueError):
        _layer(0, fuse_prob=1.5)
    with pytest.raises(ValueError):
        _fluvial.fluvial(nx=8, ny=8, nz=2, xsiz=10.0, ysiz=10.0, zsiz=1.0, bifurcate=True, fuse_prob=0.5)
