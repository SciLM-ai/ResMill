import numpy as np
import pytest
from resmill.layers.gaussian import GaussianLayer
from resmill.layers.base import Layer
from resmill.reservoir import Reservoir


def _make_gaussian(nx, ny, nz, top_depth, z_len, dip=0):
    layer = GaussianLayer(nx=nx, ny=ny, nz=nz, x_len=100, y_len=100,
                          z_len=z_len, top_depth=top_depth, dip=dip)
    layer.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03, perm_std=0.5, ntg=0.7)
    return layer


def test_reservoir_stacking():
    l1 = _make_gaussian(10, 10, 5, top_depth=1000, z_len=10)
    l2 = _make_gaussian(10, 10, 8, top_depth=1010, z_len=16)
    res = Reservoir([l1, l2])
    assert res.poro_mat.shape == (10, 10, 13)
    assert res.perm_mat.shape == (10, 10, 13)
    assert res.active.shape == (10, 10, 13)
    assert res.nz == 13
    assert res.n_layers == 2


def test_reservoir_stacks_depth_coherently():
    l1 = _make_gaussian(10, 10, 5, top_depth=1000, z_len=10)   # shallow layer
    l2 = _make_gaussian(10, 10, 8, top_depth=1010, z_len=16)   # deep layer
    res = Reservoir([l1, l2])
    # k increases upward through the stack: the deep layer fills global
    # k=0..7 and the shallow layer k=8..12, each in its internal order.
    assert np.array_equal(res.poro_mat[:, :, :8], l2.poro_mat)
    assert np.array_equal(res.poro_mat[:, :, 8:], l1.poro_mat)
    assert np.array_equal(res.active[:, :, :8], l2.active)


def test_reservoir_single_layer():
    l1 = _make_gaussian(10, 10, 5, top_depth=1000, z_len=10)
    res = Reservoir(l1)
    assert res.n_layers == 1
    assert res.poro_mat.shape == (10, 10, 5)


def test_reservoir_rejects_mismatched_nx():
    l1 = _make_gaussian(10, 10, 5, top_depth=1000, z_len=10)
    l2 = GaussianLayer(nx=12, ny=10, nz=5, x_len=100, y_len=100, z_len=10, top_depth=1010)
    l2.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03, perm_std=0.5, ntg=0.7)
    with pytest.raises(ValueError, match="nx"):
        Reservoir([l1, l2])


def test_reservoir_rejects_z_gap():
    l1 = _make_gaussian(10, 10, 5, top_depth=1000, z_len=10)
    l2 = _make_gaussian(10, 10, 5, top_depth=1050, z_len=10)  # gap: 1010 to 1050
    with pytest.raises(ValueError, match="does not match"):
        Reservoir([l1, l2])
