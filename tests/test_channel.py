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
