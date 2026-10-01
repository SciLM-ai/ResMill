"""Finite faults: throw dying out at an elliptical tip line, reverse drag, dip, curved traces, FAULTS/MULTFLT."""
import numpy as np
import pytest

from resmill.export import _build_geometry, to_grdecl
from resmill.faults import Fault
from resmill.layers.base import Layer

NX, NY, NZ, DX, DZ, TOP = 60, 40, 10, 50.0, 5.0, 2000.0


def flat_layer():
    layer = Layer(NX, NY, NZ, NX * DX, NY * DX, NZ * DZ, top_depth=TOP, kzkx=0.1)
    layer.poro_mat = np.full((NX, NY, NZ), 0.2)
    layer.perm_mat = np.full((NX, NY, NZ), 100.0)
    return layer


def top_offset(zc, j):
    """Offset of the top surface across the fault along doubled-corner row ``j`` (max minus min over x)."""
    top = zc[:, j, 0]
    return float(top.max() - top.min())


def test_no_faults_changes_nothing():
    layer = flat_layer()
    a = _build_geometry([layer])
    b = _build_geometry([layer], faults=[])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_a_fault_offsets_the_layers_by_its_throw_at_its_centre_and_dies_at_its_tips():
    """A 1.2 km fault striking along y through the model's middle, throw 20 m: across its centre the top
    surface is offset by about the throw; beyond its tips it is not offset at all (Walsh & Watterson 1987)."""
    layer = flat_layer()
    f = Fault(center=(1500.0, 1000.0), strike=90.0, length=1200.0, throw=20.0, dip=60.0, drag=(0.0, 0.0))
    _, _, zc, act = _build_geometry([layer], faults=[f])
    j_mid = 2 * int(1000.0 // DX)
    assert top_offset(zc, j_mid) == pytest.approx(20.0, rel=0.1)
    j_far = 2 * int(1750.0 // DX)                                  # 750 m from the centre: beyond the 600 m tip
    assert top_offset(zc, j_far) == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(zc, axis=2) >= -1e-9)


def test_the_trace_moves_with_depth_by_the_dip():
    """A 60 degree fault's trace at the base of a 200 m stack lies 200 / tan(60) = 115 m further toward
    the hanging wall than at the top (25 m cells: measured to half a cell)."""
    nx, ny, nz, dx, dz = 120, 40, 20, 25.0, 10.0
    layer = Layer(nx, ny, nz, nx * dx, ny * dx, nz * dz, top_depth=TOP, kzkx=0.1)
    f = Fault(center=(1500.0, 500.0), strike=90.0, length=4000.0, throw=15.0, dip=60.0, drag=(0.0, 0.0))
    _, _, zc, _ = _build_geometry([layer], faults=[f])
    j = 2 * (ny // 2)
    step = lambda k: float(np.argmax(np.abs(np.diff(zc[:, j, k])) > 1.0))
    moved = (step(nz) - step(0)) * 0.5 * dx                          # doubled corners: half a cell per index
    assert moved == pytest.approx(200.0 / np.tan(np.radians(60.0)), abs=0.75 * dx)
    assert moved > 80.0


def test_the_grdecl_lists_the_faults_faces_and_its_multiplier(tmp_path):
    """to_grdecl writes FAULTS (the stair-stepped cell faces between hanging wall and footwall) and MULTFLT."""
    layer = flat_layer()
    f = Fault(center=(1500.0, 1000.0), strike=90.0, length=1200.0, throw=20.0, dip=60.0, mult=0.01, name="F1")
    to_grdecl(layer, tmp_path / "m.grdecl", faults=[f])
    text = (tmp_path / "m.grdecl").read_text()
    assert "\nFAULTS\n" in text and "\nMULTFLT\n" in text
    assert "'F1' 0.01 /" in text
    rows = [line.split() for line in text.split("\nFAULTS\n")[1].split("\n/\n")[0].splitlines() if line.strip()]
    assert len(rows) > 5 and all(r[0] == "'F1'" and r[7] in ("'X'", "'Y'") for r in rows)
    i_cols = {int(r[1]) for r in rows}
    assert i_cols <= set(range(28, 34))                              # the faces sit along x = 1.5 km (+ dip shift)


def test_the_faults_faces_are_continuous_where_the_plane_cuts_a_cell():
    """Every layer's hanging wall meets its footwall along an unbroken line of faces: a cell the dipping plane
    cuts goes with the side of its middle, so no row leaves a gap in FAULTS (a leak for MULTFLT)."""
    layer = flat_layer()
    faces = []
    f = Fault(center=(1520.0, 1000.0), strike=90.0, length=1200.0, throw=20.0, dip=60.0, drag=(0.0, 0.0))
    _build_geometry([layer], faults=[f], _faces=faces)                # off the cell edge: the plane cuts cells
    side = faces[0][1]
    broken = 0
    for k in range(side.shape[2]):
        for j in range(side.shape[1]):
            row = side[:, j, k]
            if (row > 0).any() and (row < 0).any() and not np.any(row[:-1] * row[1:] == -1):
                broken += 1
    assert broken == 0
