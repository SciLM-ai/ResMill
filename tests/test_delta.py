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
    layer.create_geology(seed=3, azimuth=0.0, bifurcate=True, n_trees=2, paint_mouth_bars=False)
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
