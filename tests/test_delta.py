"""Tests for the fluvial-engine-driven DeltaLayer."""
import numpy as np
import pytest

from resmill.layers.delta import DeltaLayer


def _make_layer(**kwargs):
    defaults = dict(nx=64, ny=64, nz=24, x_len=64 * 16, y_len=64 * 16,
                    z_len=24 * 3, top_depth=1000)
    defaults.update(kwargs)
    return DeltaLayer(**defaults)


def test_delta_shapes():
    layer = _make_layer()
    layer.create_geology(seed=0)
    assert layer.poro_mat.shape == (64, 64, 24)
    assert layer.perm_mat.shape == (64, 64, 24)
    assert layer.active.shape == (64, 64, 24)
    assert layer.facies.shape == (64, 64, 24)


def test_delta_has_sand():
    layer = _make_layer()
    layer.create_geology(seed=0)
    assert layer.active.sum() > 50


def test_delta_poro_in_bounds():
    layer = _make_layer()
    layer.create_geology(seed=0)
    assert np.all(layer.poro_mat >= 0)
    assert np.all(layer.poro_mat <= 1)


def test_delta_perm_positive_where_active():
    layer = _make_layer()
    layer.create_geology(seed=0)
    active_perm = layer.perm_mat[layer.active == 1]
    assert np.all(active_perm > 0)


def test_delta_facies_is_alluvsim_6class():
    """``self.facies`` is Alluvsim 6-class (-1..4); active is the 0/1 collapse."""
    layer = _make_layer()
    layer.create_geology(seed=0)
    assert set(np.unique(layer.facies)).issubset({-1, 0, 1, 2, 3, 4})
    assert set(np.unique(layer.active)).issubset({0, 1})


def test_delta_fan_spreads_in_y():
    """Fan signature: distal Y-spread > proximal Y-spread.

    Averaged over a handful of seeds to absorb the AR(2)-walk variance.
    """
    ratios = []
    for seed in range(0, 5):
        layer = _make_layer()
        layer.create_geology(seed=seed)
        a = layer.active
        nx_ = a.shape[0]
        prox = a[: nx_ // 4, :, :].sum(axis=(0, 2))
        dist = a[3 * nx_ // 4 :, :, :].sum(axis=(0, 2))
        prox_span = float((prox > 0).sum())
        dist_span = float((dist > 0).sum())
        ratios.append(dist_span / max(prox_span, 1.0))
    assert np.mean(ratios) > 1.5, (
        f"Expected mean Y-spread(distal)/Y-spread(proximal) > 1.5; "
        f"got ratios={ratios} mean={np.mean(ratios):.2f}"
    )


def test_delta_azimuth_rotates_fan():
    """At azimuth=270° (compass), the fan should progradate in +y."""
    def y_centroid(az):
        layer = _make_layer()
        layer.create_geology(seed=0, azimuth=az)
        a = layer.active
        total = a.sum()
        if total == 0:
            return 0.0
        ys = np.arange(a.shape[1])
        cy = (a.sum(axis=(0, 2)) * ys).sum() / total
        return float(cy)

    cy0 = y_centroid(0.0)        # fan opens to +x; y-centroid ≈ y_center
    cy270 = y_centroid(270.0)    # fan opens to +y; y-centroid > y_center
    ny = 64
    assert cy270 > cy0, (
        f"az=270 should pull Y centroid above az=0; got cy0={cy0:.2f}, cy270={cy270:.2f}"
    )
    assert cy270 > ny / 2, (
        f"az=270 centroid should be above midline; got cy270={cy270:.2f}"
    )


def test_delta_in_reservoir():
    from resmill.layers.gaussian import GaussianLayer
    from resmill.reservoir import Reservoir

    g = GaussianLayer(nx=64, ny=64, nz=8, x_len=64 * 16, y_len=64 * 16,
                      z_len=24, top_depth=1000)
    g.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03,
                     perm_std=0.5, ntg=0.7)

    d = _make_layer(top_depth=1024)
    d.create_geology(seed=0)

    res = Reservoir([g, d])
    assert res.poro_mat.shape == (64, 64, 32)


def test_delta_tree_is_a_tree():
    """``bifurcate=True``: discharge falls with branch order, so widths do too,
    splits happen beyond the trunk (order >= 2 exists), some branches end on
    the plain and some rejoin, and the channel presets are untouched."""
    layer = DeltaLayer(nx=64, ny=64, nz=32, x_len=640, y_len=640, z_len=32, top_depth=0)
    layer.create_geology(seed=3, azimuth=0.0, bifurcate=True, n_trees=2, front_radius=0.8, paint_mouth_bars=False)
    tb = layer.tree_branches
    assert layer.active.sum() > 50
    assert max(b['order'] for b in tb) >= 2
    mean_q = {o: np.mean([b['q'] for b in tb if b['order'] == o]) for o in sorted({b['order'] for b in tb})}
    orders = sorted(mean_q)
    assert all(mean_q[a] > mean_q[b] for a, b in zip(orders, orders[1:])), mean_q
    assert all(b['q'] < 1.0 for b in tb if b['order'] >= 1)
    assert any(b['tip'] for b in tb) and any(b['merged'] for b in tb)


def test_delta_tree_flag_off_is_default():
    a = DeltaLayer(nx=48, ny=48, nz=24, x_len=480, y_len=480, z_len=24, top_depth=0)
    a.create_geology(seed=7, azimuth=0.0)
    b = DeltaLayer(nx=48, ny=48, nz=24, x_len=480, y_len=480, z_len=24, top_depth=0)
    b.create_geology(seed=7, azimuth=0.0, bifurcate=False)
    assert np.array_equal(a.active, b.active)
    assert a.tree_branches == []


def test_delta_bottom_generation_sits_on_the_floor():
    """The lowest generation's channel base is on the floor (no one-cell sand sliver
    from a generation buried below it)."""
    from resmill.layers.delta import DeltaLayer
    layer = DeltaLayer(nx=48, ny=48, nz=16, x_len=480, y_len=480, z_len=16, top_depth=0.0)
    layer.create_geology(n_generations=2, mCHdepth=4.0, bifurcate=True, seed=3)
    sand = (layer.facies >= 1).mean(axis=(0, 1))
    assert sand[0] > 0                     # bottom generation reaches the floor ...
    assert sand[0] <= 1.5 * sand[2] + 1e-6  # ... as a full body, not a sliver over mud


def test_delta_passes_splay_step_and_max_sinuosity_to_every_generation(monkeypatch):
    """The fluvial passthrough of the CHANGELOG: both options reach the engine of each generation."""
    from resmill.layers import _fluvial
    seen = []
    original = _fluvial.fluvial.__init__

    def record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        seen.append((self.splay_step, self.max_sinuosity))

    monkeypatch.setattr(_fluvial.fluvial, "__init__", record)
    layer = DeltaLayer(nx=32, ny=32, nz=8, x_len=512, y_len=512, z_len=16, top_depth=0.0)
    layer.create_geology(seed=1, n_generations=2, ntime_per_gen=3, splay_step=20.0, max_sinuosity=2.2)
    assert seen == [(20.0, 2.2)] * 2


# --------------------------------------------------------------------------
# opt-in stop at the net-to-gross: trees are added until the level holds its sand
# --------------------------------------------------------------------------

TREE = dict(bifurcate=True, n_generations=3, n_bifurcations=12, mCHdepth=4.0, mCHwdratio=20.0, stdevCHdepth=0.4,
            stdevCHwdratio=1.0, mCHsinu=1.08, trunk_length_fraction=0.3, paint_mouth_bars=True,
            mouth_bar_length_factor=2.5, mouth_bar_width_factor=1.0, q_min=0.03, front_radius=1.0,
            azimuth=0.0, seed=3)


def _tree_delta(**extra):
    layer = DeltaLayer(nx=48, ny=48, nz=12, x_len=1920.0, y_len=1920.0, z_len=24.0, top_depth=1000)
    layer.create_geology(**{**TREE, **extra})
    return layer


def _tree_calls(**fluvial_args):
    """Run the engine of one level with a spy on every tree it grows; returns the ``old`` flag of each."""
    from resmill.layers._fluvial import fluvial

    class Spy(fluvial):
        calls = []

        def _simulate_tree(self, old=False):
            self.calls.append(old)
            super()._simulate_tree(old=old)

    engine = Spy(nx=48, ny=48, nz=12, xsiz=40.0, ysiz=40.0, zsiz=2.0, xmn=20.0, ymn=20.0, nlevel=1, level_z=[24.0],
                 NTGtarget=0.99, bifurcate=True, n_bifurcations=12, mCHdepth=4.0, mCHwdratio=20.0, q_min=0.03,
                 stdevCHsource=0.2, seed=3, **fluvial_args)
    engine.calls = []
    engine.simulation()
    return engine.calls


def test_the_after_tree_hook_ends_a_levels_trees_and_leaves_none_abandoned():
    """Without it ``n_trees`` networks are grown and every one but the last is abandoned (``old``); with it
    the hook is asked after each tree and the first True ends the level, no tree being abandoned."""
    assert _tree_calls(n_trees=3) == [True, True, False]
    asked = []
    assert _tree_calls(n_trees=5, after_tree=lambda e: asked.append(e) or len(asked) == 2) == [False, False]
    assert len(asked) == 2


def test_the_stop_at_ntg_adds_networks_until_the_sand_share_is_reached():
    """NTGtarget is the sand share of the whole layer (facies 1 or more, mouth bars included): generation g of 3
    asks for (g + 1) / 3 of it, counted once over the layer, so the layer ends at its target plus at most
    the last network, a fraction of a percent here; ``n_trees`` is only the cap."""
    shares = {}
    for target in (0.15, 0.30):
        layer = _tree_delta(tree_ntg_stop=True, n_trees=200, NTGtarget=target)
        shares[target] = float((layer.facies >= 1).mean())
        assert target <= shares[target] < target + 0.015, (target, shares[target])
        if target == 0.15:     # the cap would be 3 generations of 200 networks, 3000 branches or more
            assert len(layer.tree_branches) < 3 * 200 * 5
    assert shares[0.30] > shares[0.15] + 0.12


def test_the_stop_at_ntg_counts_sand_once_where_bars_cross_generations():
    """Bars 0.35 of their half-width thick (four times the default) reach down through several generations; a
    generation's own sand then overstates what it adds to the layer, and the share still ends at the target."""
    layer = _tree_delta(tree_ntg_stop=True, n_trees=200, NTGtarget=0.25, mouth_bar_hw_ratio=0.15, mouth_bar_dw_ratio=0.2)
    assert 0.25 <= (layer.facies >= 1).mean() < 0.27


def test_the_stop_at_ntg_stamps_older_networks_as_sand_not_mud_fill():
    """With mFFCHprop = 1 the default mud-fills every abandoned network (FFCH cells); the stop leaves none."""
    default = _tree_delta(n_trees=6, mFFCHprop=1.0, stdevFFCHprop=0.0)
    stopped = _tree_delta(tree_ntg_stop=True, n_trees=6, NTGtarget=0.99, mFFCHprop=1.0, stdevFFCHprop=0.0)
    assert (default.facies == 0).sum() > 0 and (stopped.facies == 0).sum() == 0
    assert (stopped.facies >= 1).mean() > (default.facies >= 1).mean()


def test_the_stop_at_ntg_counts_the_mouth_bars():
    """Bars are painted after every tree, so bigger bars reach the same share with fewer networks: the same
    target is met by a network share about the size of a bar's."""
    small = _tree_delta(tree_ntg_stop=True, n_trees=200, NTGtarget=0.2, mouth_bar_length_factor=1.0, mouth_bar_width_factor=0.5)
    large = _tree_delta(tree_ntg_stop=True, n_trees=200, NTGtarget=0.2, mouth_bar_length_factor=4.0, mouth_bar_width_factor=1.5)
    for layer in (small, large):
        assert 0.18 < (layer.facies >= 1).mean() < 0.3
    assert (large.facies == 3).sum() > (small.facies == 3).sum()
    assert len(large.tree_branches) < len(small.tree_branches)


def test_the_stop_at_ntg_needs_the_tree_mode():
    with pytest.raises(ValueError, match="bifurcate"):
        _tree_delta(tree_ntg_stop=True, bifurcate=False)


# --------------------------------------------------------------------------
# opt-in levels: networks one to a level, levels added until the layer holds its sand
# --------------------------------------------------------------------------

def test_spread_levels_come_both_ends_first_then_halve_the_gaps():
    """Channel tops from the floor to the roof in the order 0, 1, 1/2, 1/4, 3/4, 1/8, ...: any prefix of
    2^m + 1 levels is evenly spaced, so a stop part-way leaves no part of the layer bare."""
    from resmill.layers.delta import _spread_levels

    z = _spread_levels(4.0, 20.0, 9)
    np.testing.assert_allclose(z, [4.0, 20.0, 12.0, 8.0, 16.0, 6.0, 14.0, 10.0, 18.0])
    for m in (1, 2, 3):
        np.testing.assert_allclose(np.diff(np.sort(z[:2 ** m + 1])), 16.0 / 2 ** m)
    assert len(set(_spread_levels(0.0, 1.0, 100))) == 100


def test_a_level_stop_adds_levels_until_the_layer_holds_its_sand():
    """One network a level and three levels leave the layer at 2 %; with ``max_levels`` the stop adds levels (the
    gaps halved in turn) until the share is the target's, to within the last network (0.7 % of the layer here)."""
    few = _tree_delta(tree_ntg_stop=True, n_trees=1, NTGtarget=0.12)
    many = _tree_delta(tree_ntg_stop=True, n_trees=1, NTGtarget=0.12, max_levels=60)
    assert (few.facies >= 1).mean() < 0.06
    assert 0.12 <= (many.facies >= 1).mean() < 0.14
    assert len({b["gen"] for b in many.tree_branches}) > 10


def test_a_level_stop_spreads_its_networks_through_the_whole_layer():
    """Every third of the layer, the floor, the middle and the roof, holds sand of the same order."""
    layer = _tree_delta(tree_ntg_stop=True, n_trees=1, NTGtarget=0.12, max_levels=60)
    sand = (layer.facies >= 1).mean(axis=(0, 1))
    thirds = [sand[:4].mean(), sand[4:8].mean(), sand[8:].mean()]
    assert min(thirds) > 0.5 * max(thirds) and min(thirds) > 0.04, thirds


def test_a_level_stop_ends_at_the_cap_on_levels():
    layer = _tree_delta(tree_ntg_stop=True, n_trees=1, NTGtarget=0.9, max_levels=4)
    assert {b["gen"] for b in layer.tree_branches} == {0, 1, 2, 3}
    assert (layer.facies >= 1).mean() < 0.1


def test_max_levels_needs_the_stop():
    with pytest.raises(ValueError, match="tree_ntg_stop"):
        _tree_delta(max_levels=10)


# --------------------------------------------------------------------------
# opt-in levees that follow their own branch
# --------------------------------------------------------------------------

def _levee_calls(**extra):
    """(depth of the branch, levee width, levee height) of every levee a tree's branches are given."""
    from resmill.layers._fluvial import fluvial

    seen = []

    class Spy(fluvial):
        def _stamp_levee(self, LV_depth, LV_width, LV_height, LV_asym, LV_thin):
            seen.append((self.CHdepth, LV_width, LV_height))
            super()._stamp_levee(LV_depth, LV_width, LV_height, LV_asym, LV_thin)

    engine = Spy(nx=48, ny=48, nz=12, xsiz=40.0, ysiz=40.0, zsiz=2.0, xmn=20.0, ymn=20.0, nlevel=1, level_z=[24.0],
                 NTGtarget=0.99, bifurcate=True, n_bifurcations=12, mCHdepth=4.0, mCHwdratio=20.0, q_min=0.03,
                 mLVwidth=60.0, stdevLVwidth=0.0, mLVheight=0.5, stdevLVheight=0.0, stdevCHdepth=0.0,
                 stdevCHsource=0.2, seed=3, **extra)
    engine.simulation()
    return np.array(seen)


def test_levees_follow_their_branch_when_asked():
    """Off (the default) every branch gets the trunk's levee, 60 m wide and 0.5 m high. On, a branch of depth
    d = 4 q^0.4 has width 20 x 4 q^0.5 and a levee scaled by the same q: width 60 q^0.5, height 0.5 q^0.4."""
    default = _levee_calls()
    assert len(default) > 10 and np.all(default[:, 1] == 60.0) and np.all(default[:, 2] == 0.5)
    scaled = _levee_calls(branch_levees=True)
    q = (scaled[:, 0] / 4.0) ** (1.0 / 0.4)
    np.testing.assert_allclose(scaled[:, 1], 60.0 * q ** 0.5, rtol=1e-6)
    np.testing.assert_allclose(scaled[:, 2], 0.5 * q ** 0.4, rtol=1e-6)
    assert scaled[:, 1].max() == pytest.approx(60.0) and scaled[:, 1].min() < 30.0


# --------------------------------------------------------------------------
# the bar's footprint as asked
# --------------------------------------------------------------------------

def test_a_mouth_bar_is_as_long_and_as_wide_as_asked():
    """A bar of length 600 m and half-width 150 m (12 and 6 tip widths of 50 and 25 m, say) at a tip heading +x:
    its footprint runs 600 m along the heading and 300 m across at its widest, a third of the way along."""
    from types import SimpleNamespace
    from resmill.layers.delta import _paint_mouth_bar_into_engine

    n, size = 120, 10.0
    canvas = SimpleNamespace(facies=np.full((n, n, 4), -1, dtype=np.int8), x=np.arange(n) * size + size / 2,
                             y=np.arange(n) * size + size / 2, xsiz=size, ysiz=size, zsiz=1.0)
    _paint_mouth_bar_into_engine(canvas, 200.0, 600.0, 3.9, 0.0, 600.0, 150.0, 0.1, 0.1)
    plan = (canvas.facies >= 1).any(axis=2)
    xs, ys = np.where(plan)
    assert (xs.max() - xs.min() + 1) * size == pytest.approx(600.0, abs=2 * size)
    widest = plan.sum(axis=1).max() * size
    assert widest == pytest.approx(300.0, abs=2 * size)
    assert plan.sum(axis=1).argmax() * size + size / 2 - 200.0 == pytest.approx(200.0, abs=60.0)
