"""Fault seal against independent references: hand-built pairs of columns whose multipliers and shale gouge ratios are
worked out outside the code (the series resistance behind Manzocchi et al. 1999, the slipped interval of Yielding et al.
1997), so that a wrong length, window or dip fails here and not only a golden number."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from resmill.fault_seal import Seal, face_multipliers

DZ, TOP = 0.5, 1000.0


def columns(depth, side, vsh=0.5, perm=100.0, dip=90.0):
    """The arguments of ``face_multipliers`` (bar the cell sizes) for columns with the cell interfaces ``depth``
    (nx, ny, nz + 1; m, k top-down), each wholly on its ``side`` (nx, ny; +1 hanging wall, -1 footwall) of one fault of
    ``dip``. ``vsh`` and ``perm`` (one value for the three directions, or three) broadcast to (nx, ny, nz); cells
    thinner than 5 mm are inactive, as the exporter makes them."""
    depth = np.asarray(depth, dtype=float)
    nx, ny, nz = depth.shape[0], depth.shape[1], depth.shape[2] - 1
    full = lambda v: np.broadcast_to(np.asarray(v, dtype=float), (nx, ny, nz))
    perms = tuple(perm) if isinstance(perm, tuple) else (perm,) * 3
    return dict(faces=[(SimpleNamespace(name="F1", dip=dip), np.broadcast_to(
        np.asarray(side, dtype=np.int8)[:, :, None], (nx, ny, nz)).copy())],
        zc=np.repeat(np.repeat(depth, 2, axis=0), 2, axis=1), act=(np.diff(depth, axis=2) > 5e-3).astype(np.int32),
        vsh=full(vsh), perms=tuple(full(p) for p in perms))


def layered(shifts, nz=40):
    """Flat cells of ``DZ`` m, the stack of each column ``shifts`` (nx, ny) m deeper than the top at ``TOP``."""
    return TOP + np.asarray(shifts, dtype=float)[:, :, None] + DZ * np.arange(nz + 1)


def step(throw, up_first=True, along="x", nz=40, **kw):
    """Two columns, the downthrown one ``throw`` m deeper, along x (X faces) or y (Y faces); the upthrown one is the
    lower-index column when ``up_first`` (the downthrown one is the hanging wall of a normal fault)."""
    shift, side = np.array([0.0, throw]), np.array([-1, 1])
    if not up_first:
        shift, side = shift[::-1], side[::-1]
    if along == "y":
        shift, side = shift[None, :], side[None, :]
    else:
        shift, side = shift[:, None], side[:, None]
    return columns(layered(shift, nz), side, **kw)


def fault_rock_k(sgr, d):
    """Fault-rock permeability (mD) of Manzocchi et al. 1999: log10 k_f = -4 SGR - 1/4 log10(D) (1 - SGR)^5."""
    return 10.0 ** (-4.0 * sgr - 0.25 * math.log10(d) * (1.0 - sgr) ** 5)


def manzocchi(sgr, d, ki, kj, li, lj, dt_ratio=66.0):
    """The multiplier of one face from first principles: the series resistance between the two cell centres with the
    fault rock (thickness d / dt_ratio) in place of half its thickness of each cell, over the resistance without it.
    ``li`` and ``lj`` are the whole cells across the face."""
    tf = d / dt_ratio
    plain = 0.5 * li / ki + 0.5 * lj / kj
    faulted = 0.5 * (li - tf) / ki + tf / fault_rock_k(sgr, d) + 0.5 * (lj - tf) / kj
    return plain / faulted


def mults(c, dx=25.0, dy=25.0, seal=None):
    """``face_multipliers`` of a stack built by :func:`columns`."""
    return face_multipliers(**c, dx=dx, dy=dy, seal=seal or Seal())


def test_the_reference_resistance_gives_the_numbers_worked_by_hand():
    """Two 100 mD cells 25 m wide, a fault rock of 1 mD and 0.5 m thick: 0.25 over 0.745 (the cell halves 12.25 m of
    100 mD each, 0.1225 + 0.1225, and the slab 0.5 / 1 = 0.5), T = 0.3356; and with t_f / 2 = 0.25 m of each cell
    replaced, the same value from the closed form 1 / (1 + t_f (2/k_f - 1/k_i - 1/k_j) / (L_i/k_i + L_j/k_j))."""
    plain, faulted = 0.5 * 25.0 / 100.0 * 2.0, 2.0 * 0.5 * (25.0 - 0.5) / 100.0 + 0.5 / 1.0
    assert plain / faulted == pytest.approx(0.25 / 0.745, rel=1e-12)
    assert 1.0 / (1.0 + 0.5 * (2.0 / 1.0 - 2.0 / 100.0) / (50.0 / 100.0)) == pytest.approx(0.25 / 0.745, rel=1e-12)


@pytest.mark.parametrize("up_first", [True, False])
@pytest.mark.parametrize("along,dx,dy", [("x", 25.0, 50.0), ("y", 25.0, 50.0), ("x", 50.0, 25.0), ("y", 50.0, 25.0)])
def test_the_multiplier_uses_the_whole_cell_across_the_face(along, dx, dy, up_first):
    """Manzocchi's L_i and L_j are the whole cells across the face (the half in the bracket is already in t_f): dx for
    an X face, dy for a Y face, never the other. Uniform clay 0.5 gives SGR 0.5 whichever beds slid past; the faces
    lie between 100 mD cells."""
    c = step(10.0, up_first, along, vsh=0.5)
    out = mults(c, dx, dy)
    length = dx if along == "x" else dy
    expected = manzocchi(0.5, 10.0, 100.0, 100.0, length, length)
    key = "MULTX" if along == "x" else "MULTY"
    assert len(out["faults"][0]["mult"]) == 20 and np.allclose(out["faults"][0]["mult"], expected, rtol=1e-9)
    faced = out[key] < 1.0
    assert faced.sum() == 20 and np.allclose(out[key][faced], expected, rtol=1e-9)
    assert np.all(out["MULTZ"] == 1.0)


def test_the_multiplier_is_the_published_number_for_a_typical_face():
    """SGR 0.5 and 10 m of displacement between 100 mD cells 25 m wide: k_f = 0.00982 mD, t_f = 0.1515 m, so the
    resistance 15.68 against 0.25 gives 0.01595 (the half-length form gave 0.00804)."""
    out = mults(step(10.0, vsh=0.5))
    assert np.allclose(out["faults"][0]["mult"], 0.01595, rtol=1e-3)


@pytest.mark.parametrize("ki,kj", [(100.0, 100.0), (100.0, 400.0), (5.0, 2000.0)])
def test_the_multiplier_weights_each_cell_by_its_own_permeability(ki, kj):
    """The two cells have their own permeability (the lower-index column's first): the multiplier is the series
    resistance of both, as the closed form's L_i/k_i + L_j/k_j."""
    c = step(10.0, vsh=0.5)
    c["perms"] = tuple(np.where(np.arange(2)[:, None, None] == 0, ki, kj) * np.ones(c["vsh"].shape) for _ in range(3))
    out = mults(c, 25.0, 25.0)
    assert np.allclose(out["faults"][0]["mult"], manzocchi(0.5, 10.0, ki, kj, 25.0, 25.0), rtol=1e-9)


@pytest.mark.parametrize("dip", [90.0, 60.0, 45.0, 30.0])
def test_the_displacement_along_the_plane_is_the_throw_over_sin_dip(dip):
    """A vertical throw T on a plane dipping at ``dip`` slips the rock T / sin(dip) along it: that D sets the fault
    rock's permeability and its thickness D / 66 (10 m of throw: D = 10, 11.55, 14.14 and 20 m)."""
    out = mults(step(10.0, vsh=0.5, dip=dip))
    expected = manzocchi(0.5, 10.0 / math.sin(math.radians(dip)), 100.0, 100.0, 25.0, 25.0)
    assert np.allclose(out["faults"][0]["mult"], expected, rtol=1e-9)
