"""Finite faults: throw dying out at an elliptical tip line, reverse drag, dip, curved traces, FAULTS/MULTFLT."""
import math

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
    assert len(rows) > 5 and all(r[0] == "'F1'" and r[7] in ("'X'", "'Y'", "'Z'") for r in rows)
    i_cols = {int(r[1]) for r in rows}
    assert i_cols <= set(range(28, 34))                              # the faces sit along x = 1.5 km (+ dip shift)


def trace_offsets(f, nx=200, ny=160, dx=25.0):
    """The fault's top-surface trace on a fine flat grid: for each row j, the x (m) where the side flips."""
    layer = Layer(nx, ny, 4, nx * dx, ny * dx, 20.0, top_depth=TOP, kzkx=0.1)
    faces = []
    _build_geometry([layer], faults=[f], _faces=faces)
    side = faces[0][1][:, :, 0]
    rows, xs = [], []
    for j in range(ny):
        flips = np.flatnonzero(side[:-1, j] * side[1:, j] == -1)
        if flips.size:
            rows.append((j + 0.5) * dx)
            xs.append((flips[0] + 1) * dx)
    return np.array(rows), np.array(xs)


def test_bends_move_the_trace_by_their_rms_and_keep_its_centre_and_strike():
    """bends = 0.03 on a 3 km straight fault striking along y: the trace wanders about 0.03 x 3 km = 90 m
    (rms) about its chord, whose position and direction stay those asked for; bends = 0 draws the straight
    trace exactly as before, and the same seed draws the same bends."""
    base = dict(center=(2500.0, 2000.0), strike=90.0, length=3000.0, throw=20.0, dip=89.0, drag=(0.0, 0.0))
    y0, x0 = trace_offsets(Fault(**base))
    assert np.ptp(x0) <= 25.0
    y1, x1 = trace_offsets(Fault(**base, bends=0.03, seed=4))
    assert np.ptp(y1) > 2900.0
    dev = x1 - 2500.0
    assert np.sqrt(np.mean(dev ** 2)) == pytest.approx(90.0, rel=0.1)
    slope, mean = np.polyfit(y1 - 2000.0, dev, 1)
    assert abs(mean) < 10.0 and abs(slope) < 0.01
    y2, x2 = trace_offsets(Fault(**base, bends=0.03, seed=4))
    assert np.array_equal(x1, x2)
    _, x3 = trace_offsets(Fault(**base, bends=0.03, seed=5))
    assert not np.array_equal(x1, x3)


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


def test_a_reverse_fault_raises_its_hanging_wall_without_stretching_cells():
    """A k-ordered column cannot hold a reverse fault's repeated section, so each column goes whole to one
    side: the hanging wall (+x here) rises by about the throw, and no cell is stretched across the plane
    (5 m cells stay 5 m to within the few per cent the tip-line profile bends them)."""
    layer = flat_layer()
    f = Fault(center=(1520.0, 1000.0), strike=90.0, length=1200.0, throw=20.0, dip=60.0, drag=(0.0, 0.0),
              reverse=True)
    faces = []
    _, _, zc, _ = _build_geometry([layer], faults=[f], _faces=faces)
    thickness = np.diff(zc, axis=2)
    assert thickness.min() > 4.7 and thickness.max() < 5.3
    j_mid = 2 * int(1000.0 // DX)
    top = zc[:, j_mid, 0]
    assert top[:40].mean() - top[-40:].mean() == pytest.approx(20.0, rel=0.15)   # east (hanging wall) higher
    side = faces[0][1]
    assert np.all(side == side[:, :, :1])                                         # whole columns


def test_faults_lists_every_face_across_which_the_two_sides_touch(tmp_path):
    """Flow multiplies a connection by the face of its lower-index cell, also a connection across the throw
    (layer k against layer k'), so FAULTS must list every face on which a cell meets the fault's other side in
    any layer. Listing only same-layer pairs let a sealing dipping normal fault leak fully in Flow."""
    layer = flat_layer()
    f = Fault(center=(1520.0, 1000.0), strike=90.0, length=1200.0, throw=20.0, dip=60.0, drag=(0.0, 0.0), name="F1")
    faces = []
    _, _, zc, _ = _build_geometry([layer], faults=[f], _faces=faces)
    side = faces[0][1]
    to_grdecl(layer, tmp_path / "m.grdecl", faults=[f])
    rows = [r.split() for r in (tmp_path / "m.grdecl").read_text().split("\nFAULTS\n")[1].split("\n/\n")[0].splitlines()]
    listed = {(int(r[1]) - 1, int(r[3]) - 1, k) for r in rows if r[7] == "'X'" for k in range(int(r[5]) - 1, int(r[6]))}
    missing = []
    for i in range(NX - 1):
        for j in range(NY):
            for k in range(NZ):
                a = zc[2 * i + 1, 2 * j:2 * j + 2, k:k + 2]                    # the shared face, west cell
                for kk in range(NZ):
                    b = zc[2 * i + 2, 2 * j:2 * j + 2, kk:kk + 2]              # the shared face, east cell
                    touch = np.any(np.minimum(a[:, 1], b[:, 1]) - np.maximum(a[:, 0], b[:, 0]) > 1e-6)
                    if touch and side[i, j, k] * side[i + 1, j, kk] == -1 and (i, j, k) not in listed:
                        missing.append((i, j, k, kk))
    assert not missing


def test_faults_lists_the_tread_where_the_hanging_wall_sits_on_the_footwall(tmp_path):
    """In the column a dipping normal fault cuts, hanging-wall cells sit on footwall cells, joined across the cut-out
    (zero-thickness) cells by PINCH. That vertical connection crosses the fault too, so the hanging-wall cell's
    +Z face is listed ('Z'); without it a sealing fault leaked fully in Flow."""
    layer = flat_layer()
    f = Fault(center=(1520.0, 1000.0), strike=90.0, length=1200.0, throw=20.0, dip=60.0, drag=(0.0, 0.0), name="F1")
    faces = []
    _, _, _, act = _build_geometry([layer], faults=[f], _faces=faces)
    side = faces[0][1]
    to_grdecl(layer, tmp_path / "m.grdecl", faults=[f])
    rows = [r.split() for r in (tmp_path / "m.grdecl").read_text().split("\nFAULTS\n")[1].split("\n/\n")[0].splitlines()]
    listed = {(int(r[1]) - 1, int(r[3]) - 1, k) for r in rows if r[7] == "'Z'" for k in range(int(r[5]) - 1, int(r[6]))}
    treads = set()
    for i in range(NX):
        for j in range(NY):
            live = [k for k in range(NZ) if act[i, j, k]]
            treads |= {(i, j, a) for a, b in zip(live, live[1:]) if side[i, j, a] * side[i, j, b] == -1}
    assert treads and treads <= listed


def thin_stack(nx=60, ny=60, nz=50, dx=50.0, dz=1.0):
    layer = Layer(nx, ny, nz, nx * dx, ny * dx, nz * dz, top_depth=TOP, kzkx=0.1)
    layer.poro_mat = np.full((nx, ny, nz), 0.2)
    layer.perm_mat = np.full((nx, ny, nz), 100.0)
    return layer


TWISTED = dict(center=(1510.0, 1500.0), strike=75.0, length=1000.0, throw=40.0, dip=60.0)


def test_no_cell_is_stretched_next_to_a_normal_faults_tread():
    """Hanging wall, footwall and cut-out are decided at every corner, since the throw changes across a column
    (tip-line profile, drag); a column-wide choice stretched the cell beside the tread (1 m cells to 4.6 m)."""
    _, _, zc, act = _build_geometry([thin_stack()], faults=[Fault(**TWISTED)])
    thick = np.diff(zc, axis=2).reshape(60, 2, 60, 2, 50).max(axis=(1, 3))
    assert thick[act > 0].max() < 1.5


def test_reverse_drag_peaks_at_the_fault_at_every_depth():
    """The drag taper is measured from the plane at each horizon's own depth: on top of a 300 m stack the hanging
    wall drops most right at the fault, and at its base the footwall rises most there (both peaked 80 m inside)."""
    nx, ny, nz = 300, 6, 60
    layer = Layer(nx, ny, nz, nx * 10.0, ny * 10.0, nz * 5.0, top_depth=TOP, kzkx=0.1)
    faces = []
    _, _, zc, _ = _build_geometry([layer], faults=[Fault(center=(1500.0, 30.0), strike=90.0, length=4000.0, throw=40.0,
                                                         dip=60.0)], _faces=faces)
    side, j = faces[0][1], ny // 2
    drop = (zc[0::2, 2 * j, 0] - TOP)[side[:, j, 0] > 0]                      # top, hanging wall, from the fault east
    rise = (TOP + nz * 5.0 - zc[0::2, 2 * j, nz])[side[:, j, nz - 1] < 0][::-1]   # base, footwall, from the fault west
    for move, least in ((drop, 20.0), (rise, 12.0)):      # past the column or two where the plane meets the horizon
        k = int(np.argmax(move[:30]))
        assert k <= 2 and move[k] > least and np.all(np.diff(move[k:k + 20]) <= 1e-6)


def test_the_default_tip_line_centre_is_the_stacks_middle_by_depth():
    """Over a 50 m zone of 0.5 m cells and a 200 m zone of 20 m cells, the tip ellipse centres on the stack's middle
    depth (125 m down), not on its middle interface by count (27.5 m down)."""
    upper = Layer(NX, NY, 100, NX * DX, NY * DX, 50.0, top_depth=TOP, kzkx=0.1)
    lower = Layer(NX, NY, 10, NX * DX, NY * DX, 200.0, top_depth=TOP + 50.0, kzkx=0.1)
    f = Fault(center=(1520.0, 1000.0), strike=90.0, length=400.0, throw=10.0, dip=60.0, drag=(0.0, 0.0))
    _, _, zc, _ = _build_geometry([upper, lower], faults=[f])
    j = 2 * int(1000.0 // DX)
    assert top_offset(zc, j) == pytest.approx(0.0, abs=1e-6)
    assert np.ptp(zc[:, j, 104]) == pytest.approx(10.0, rel=0.15)        # the interface at 2130 m


@pytest.mark.parametrize("hanging_wall,radius,strike,reverse",
                         [(1, math.inf, 90.0, False), (-1, math.inf, 90.0, False), (1, -2500.0, 30.0, False),
                          (-1, 2500.0, 135.0, True)])
def test_the_hanging_wall_lies_where_asked_and_moves_down_or_up_if_reverse(hanging_wall, radius, strike, reverse):
    """Whole hanging-wall cells move down (up on a reverse fault), whole footwall cells up (down), and the hanging wall
    lies on the side hanging_wall x the strike's normal points to. (A cell the plane truncates is thinner, and its
    middle may move either way.)"""
    layer = flat_layer()
    f = Fault(center=(1520.0, 1000.0), strike=strike, length=3000.0, throw=20.0, dip=60.0, hanging_wall=hanging_wall,
              radius=radius, reverse=reverse)
    faces = []
    xc, yc, z0, _ = _build_geometry([layer])
    _, _, zc, _ = _build_geometry([layer], faults=[f], _faces=faces)
    side = faces[0][1]
    centre = lambda z: (0.5 * (z[..., 1:] + z[..., :-1])).reshape(NX, 2, NY, 2, NZ).mean(axis=(1, 3))
    move = (centre(zc) - centre(z0)) * (-1.0 if reverse else 1.0)
    whole = np.diff(zc, axis=2).reshape(NX, 2, NY, 2, NZ).min(axis=(1, 3)) > 0.9 * DZ
    assert move[whole & (side > 0)].min() > -1e-9 and move[side > 0].max() > 5.0
    assert move[whole & (side < 0)].max() < 1e-9 and move[side < 0].min() < -2.0
    n = np.array([math.sin(math.radians(strike)), math.cos(math.radians(strike))])
    xm, ym = xc.reshape(NX, 2, NY, 2).mean(axis=(1, 3)), yc.reshape(NX, 2, NY, 2).mean(axis=(1, 3))
    along_n = (xm * n[0] + ym * n[1])[..., None] * np.ones(NZ)
    assert hanging_wall * (along_n[side > 0].mean() - along_n[side < 0].mean()) > 0.0


def test_plot_section_and_to_pyvista_draw_faulted_models():
    """Both drawing paths take faults (several, crossing)."""
    import matplotlib
    matplotlib.use("Agg")
    from resmill.plotting import plot_section
    layer = flat_layer()
    faults = [Fault(center=(1500.0, 1000.0), strike=90.0, length=1500.0, throw=15.0),
              Fault(center=(1500.0, 1000.0), strike=0.0, length=1500.0, throw=10.0, reverse=True)]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plot_section(layer, prop="poro_mat", faults=faults, ax=ax)
    assert ax.collections
    plt.close(fig)
    pv = pytest.importorskip("pyvista")
    from resmill.export import to_pyvista
    grid = to_pyvista(layer, faults=faults)
    assert grid.n_cells == NX * NY * NZ
