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


def lateral_contacts(side, zc):
    """Brute force: (face, i, j, k) of every cell meeting a cell of the other side across its + face, each shared face
    sampled at 201 points between its pillars."""
    t = np.linspace(0.0, 1.0, 201)[:, None]
    out = set()
    nx, ny, _ = side.shape
    for axis, face in ((0, "X"), (1, "Y")):
        for i in range(nx - (axis == 0)):
            for j in range(ny - (axis == 1)):
                lo, hi = side[i, j], side[i + 1, j] if axis == 0 else side[i, j + 1]
                other = lo[:, None] * hi[None, :] == -1
                if not other.any():
                    continue
                if axis == 0:
                    a, b = zc[2 * i + 1, 2 * j:2 * j + 2], zc[2 * i + 2, 2 * j:2 * j + 2]
                else:
                    a, b = zc[2 * i:2 * i + 2, 2 * j + 1], zc[2 * i:2 * i + 2, 2 * j + 2]
                A, B = a[0] + t * (a[1] - a[0]), b[0] + t * (b[1] - b[0])     # (201, nk)
                overlap = (np.minimum(A[:, 1:, None], B[:, None, 1:]) - np.maximum(A[:, :-1, None], B[:, None, :-1]))
                for k in np.flatnonzero(((overlap > 1e-6).any(axis=0) & other).any(axis=1)):
                    out.add((face, i, j, int(k)))
    return out


@pytest.mark.parametrize("case", ["twisted", "crossing faults and erosion"])
def test_faults_lists_every_contact_across_each_fault(tmp_path, case):
    """Every face on which a cell meets the fault's other side, wherever along the face (a twisted face may touch only
    between its pillars), and every cell resting on one of the other side, are listed, for each of several faults on
    the final geometry (here also cut by erosion)."""
    layer = thin_stack()
    if case == "twisted":
        faults, kw = [Fault(**TWISTED, name="F1")], {}
    else:
        faults = [Fault(center=(1500.0, 1500.0), strike=20.0, length=2500.0, throw=15.0, dip=55.0, name="F1"),
                  Fault(center=(1400.0, 1600.0), strike=110.0, length=2000.0, throw=10.0, dip=50.0, hanging_wall=-1,
                        reverse=True, name="F2")]
        kw = dict(erode_above=TOP + 12.0)
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=faults, _faces=faces, **kw)
    to_grdecl(layer, tmp_path / "m.grdecl", faults=faults, **kw)
    rows = [r.split() for r in (tmp_path / "m.grdecl").read_text().split("\nFAULTS\n")[1].split("\n/\n")[0].splitlines()]
    for fault, side in faces:
        listed = {(r[7].strip("'"), int(r[1]) - 1, int(r[3]) - 1, k) for r in rows if r[0] == f"'{fault.name}'"
                  for k in range(int(r[5]) - 1, int(r[6]))}
        assert lateral_contacts(side, zc) <= listed
        for i, j in zip(*np.nonzero((side > 0).any(axis=2) & (side < 0).any(axis=2))):
            live = [k for k in range(side.shape[2]) if act[i, j, k]]
            for a, b in zip(live, live[1:]):
                if side[i, j, a] * side[i, j, b] == -1:
                    assert ("Z", i, j, a) in listed


@pytest.mark.parametrize("names", [("NORTH_FAULT_1", "NORTH_FAULT_2"), ("A", "A"), ("F02", "")])
def test_fault_names_are_unique_within_eight_characters(tmp_path, names):
    """OPM keeps 8 characters of a fault name and merges equal ones (a sealing fault then leaks), so to_grdecl refuses
    long or repeated names, including one equal to an unnamed fault's default (F01, F02, ...)."""
    faults = [Fault(center=(1500.0, 500.0 + 800.0 * n), strike=90.0, length=600.0, throw=10.0, name=nm)
              for n, nm in enumerate(names)]
    with pytest.raises(ValueError, match="name"):
        to_grdecl(flat_layer(), tmp_path / "m.grdecl", faults=faults)


@pytest.mark.parametrize("bad", [dict(throw=-20.0), dict(throw=0.0), dict(length=0.0), dict(radius=0.0),
                                 dict(radius=300.0), dict(aspect=0.0), dict(dip=0.0), dict(dip=95.0),
                                 dict(hw_share=1.5), dict(hanging_wall=2), dict(drag=(-0.1, 0.2)), dict(mult=-1.0),
                                 dict(bends=-0.01), dict(bends=0.02)])
def test_a_fault_refuses_values_that_would_corrupt_the_grid(bad):
    """Sampling with noise reaches odd values: a negative throw, zero length or radius, a trace curled past a half
    circle, a dip outside (0, 90], shares outside [0, 1], or bends without a seed (every fault would wander alike)."""
    with pytest.raises(ValueError):
        Fault(**{**dict(center=(1500.0, 1000.0), strike=90.0, length=1200.0, throw=20.0), **bad})


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


# ----- listric faults (Fault.detach): a circular plane flattening at a detachment, the hanging wall by vertical shear -----

def research_rollover(delta_s, z_d, z_r, throw):
    """The research's closed form (design_notes/structure_research/step4_rollover_calc.py), written out again: a circular
    fault of surface dip ``delta_s`` (deg) turning flat at depth ``z_d`` (m), x measured from its surface trace; the horizon
    at ``z_r`` has ``throw`` at the fault and, by vertical shear (constant heave), depth z_r + F(x) - F(x - H) beyond the
    hanging-wall cutoff. Returns the fault's dip at z_r (deg), the heave H, and the depth as a function of the distance p
    (m) east of the footwall cutoff."""
    ds = math.radians(delta_s)
    radius = z_d / (1.0 - math.cos(ds))
    xb = radius * math.sin(ds)

    def plane(x):
        x = np.asarray(x, float)
        u = np.clip((xb - x) / radius, -1.0, 1.0)
        return np.where(x >= xb, z_d, z_d - radius * (1.0 - np.sqrt(1.0 - u ** 2)))

    th = math.acos(1.0 - (z_d - z_r) / radius)
    x_r = radius * (math.sin(ds) - math.sin(th))
    xs = np.linspace(x_r, x_r + 20000.0, 400001)
    heave = float(xs[int(np.argmin(np.abs(plane(xs) - (z_r + throw))))] - x_r)
    return math.degrees(th), heave, lambda p: z_r + plane(x_r + p) - plane(x_r + p - heave)


def rollover_section(dip, detach, z_r, throw, dx=25.0, nx=320, **kw):
    """Top-interface corner depths along the middle row of a 10 m thick layer cut by a huge listric fault (so its throw does
    not taper) whose trace at z_r passes x = 1000 m, the hanging wall to the east; returns x of the corners and depths."""
    layer = Layer(nx, 6, 2, nx * dx, 6 * dx, 10.0, top_depth=z_r, kzkx=0.1)
    f = Fault(center=(1000.0, 3 * dx), strike=90.0, length=4.0e5, throw=throw, dip=dip, detach=detach, z_center=z_r, **kw)
    xc, _, zc, _ = _build_geometry([layer], faults=[f])
    return xc[:, 6], zc[:, 6, 0]


def test_a_listric_fault_with_a_far_detachment_is_the_planar_fault():
    """As the detachment recedes the circle straightens: a listric fault with its detachment 10^9 m down gives the planar
    fault's layers to 1 mm and the same sides, the planar fault with the whole throw on the hanging wall and no drag (what
    vertical shear over a plane is)."""
    layer = thin_stack()
    kw = dict(center=(1510.0, 1500.0), strike=75.0, length=1500.0, throw=40.0, dip=60.0, z_center=TOP + 20.0)
    planar, listric = [], []
    _, _, za, _ = _build_geometry([layer], faults=[Fault(hw_share=1.0, drag=(0.0, 0.0), **kw)], _faces=planar)
    _, _, zb, _ = _build_geometry([layer], faults=[Fault(detach=1.0e9, **kw)], _faces=listric)
    assert np.abs(za - zb).max() < 1.0e-3
    assert np.array_equal(planar[0][1], listric[0][1])


def test_the_listric_plane_is_the_circle_through_the_dip_at_z_center_flat_at_the_detachment():
    """The plane is a circular arc: at z_center it is where the trace is and dips ``dip`` there, it flattens at ``detach``
    (a horizontal run beyond R sin(dip) of the trace), never turns back, and trace(plane(h)) gives h back to 1e-9 m. R is
    (detach - z_center) / (1 - cos dip), the arc's radius."""
    from resmill.faults import _listric
    dip, zc, detach = 55.0, 2500.0, 4500.0
    plane, trace = _listric(dip, zc, detach)
    radius = (detach - zc) / (1.0 - math.cos(math.radians(dip)))
    h = np.linspace(-1500.0, 3.0 * radius, 4001)
    z = plane(h)
    assert float(plane(0.0)) == pytest.approx(zc, abs=1e-9)
    slope = (plane(1e-3) - plane(-1e-3)) / 2e-3
    assert math.degrees(math.atan(slope)) == pytest.approx(dip, abs=1e-6)
    assert np.all(np.diff(z) >= 0.0) and z[-1] == pytest.approx(detach, abs=1e-9)
    assert np.all(z[h >= radius * math.sin(math.radians(dip))] == pytest.approx(detach, abs=1e-9))
    on_arc = (z > zc - 400.0) & (z < detach - 1e-6)
    assert np.abs(trace(z[on_arc]) - h[on_arc]).max() < 1e-9
    # the circle itself: every point of the arc is R from its centre, which lies R above the flat's tangent point
    cx, cz = radius * math.sin(math.radians(dip)), detach - radius
    on = (h > cx - radius) & (h < cx)                                  # past the vertical point the plane stays vertical
    assert np.abs(np.hypot(h[on] - cx, z[on] - cz) - radius).max() < 1e-6


@pytest.mark.parametrize("dip, zc, detach", [(35.0, 3000.0, 6000.0), (40.0, 2700.0, 7000.0), (55.0, 2500.0, 4500.0),
                                             (30.0, 2500.0, 8000.0), (60.0, 2400.0, 5200.0)])
def test_the_listric_plane_and_trace_are_finite_far_into_the_footwall_and_below_the_detachment(dip, zc, detach):
    """The arc's two square roots reach 0 at its vertical point (landward) and at its flat (below the detachment), where
    rounding made them -1e-17 and the whole footwall side, or every corner below the flat, NaN: they stay finite and
    monotone over any distance and depth."""
    from resmill.faults import _listric
    plane, trace = _listric(dip, zc, detach)
    h, z = np.linspace(-2.0e5, 2.0e5, 40001), np.linspace(0.0, 3.0 * detach, 40001)
    assert np.isfinite(plane(h)).all() and np.isfinite(trace(z)).all()
    assert np.all(np.diff(plane(h)) >= 0.0) and np.all(np.diff(trace(z)) >= 0.0)


def test_a_rollover_has_the_closed_form_amplitude_heave_and_width_of_vertical_shear():
    """Circular fault, surface dip 55 deg, detachment 4.5 km, horizon at 2.5 km with 300 m of throw (the research's table:
    dip at the horizon 35.9 deg, heave 438 m, steepest drag dip 4.2 deg, 90 % of the drag gone 5.24 km from the cutoff):
    the grid's top surface follows the closed form depth z + F(h) - F(h - H) at every corner clear of the fault, drags down
    by the throw at the cutoff, and measures those numbers."""
    dip, heave, depth = research_rollover(55.0, 4500.0, 2500.0, 300.0)
    assert dip == pytest.approx(35.9, abs=0.05) and heave == pytest.approx(438.0, abs=1.0)
    x, z = rollover_section(dip, 4500.0, 2500.0, 300.0)
    east = x > 1000.0 + heave + 50.0
    assert np.abs(z[east] - depth(x[east] - 1000.0)).max() < 0.05
    west = x < 1000.0 - 50.0
    assert np.abs(z[west] - 2500.0).max() < 1e-6
    first = np.argmax(east)                                           # the drag starts at the throw and falls no faster than its steepest dip
    assert 0.0 <= 300.0 - (z[first] - 2500.0) <= math.tan(math.radians(4.3)) * (x[first] - 1000.0 - heave)
    drag = z - 2500.0
    pos = x[east] - 1000.0 - heave
    dips = np.degrees(np.arctan(np.diff(z[east]) / np.diff(x[east])))
    assert -dips.min() == pytest.approx(4.2, abs=0.15)                                               # steepest drag dip
    width = pos[np.argmax(drag[east] < 0.1 * drag[east][0])]
    assert width == pytest.approx(5240.0, abs=100.0)
    gap = (x > 1000.0) & (x < 1000.0 + heave) & (np.abs(z - 2500.0) > 1.0)
    assert 0.5 * np.count_nonzero(gap) * 25.0 == pytest.approx(heave, abs=3 * 25.0)                  # treads span the heave


def test_a_deeper_detachment_gives_a_broader_flatter_roll_and_a_bigger_throw_a_bigger_one():
    """The research's table: with 300 m of throw the steepest drag dip is 4.2, 3.2 and 2.7 deg for detachments at 4.5, 6
    and 7.5 km, and 8.5 deg for 600 m of throw at 4.5 km."""
    for z_d, throw, steepest in ((4500.0, 300.0, 4.2), (6000.0, 300.0, 3.2), (7500.0, 300.0, 2.7), (4500.0, 600.0, 8.5)):
        dip, heave, _ = research_rollover(55.0, z_d, 2500.0, throw)
        x, z = rollover_section(dip, z_d, 2500.0, throw, nx=480)
        east = x > 1000.0 + heave + 50.0
        dips = np.degrees(np.arctan(np.diff(z[east]) / np.diff(x[east])))
        assert -dips.min() == pytest.approx(steepest, abs=0.2)


def test_the_listric_hanging_wall_drags_by_the_throw_at_the_fault_and_lies_still_beyond_the_flat():
    """Vertical shear: no drag past the fault's flat (R sin(dip) + heave from the trace), the drag falls monotonically from
    the fault, no cell has negative thickness, and the footwall is not moved at all (its share of the throw is zero)."""
    x, z = rollover_section(35.9, 4500.0, 2500.0, 300.0, nx=480)
    east = x > 1000.0 + 438.0 + 50.0
    drag = z[east] - 2500.0
    assert np.all(np.diff(drag) <= 1e-9) and drag[-1] == pytest.approx(0.0, abs=1e-9)
    assert np.all(drag[x[east] > 1000.0 + 438.0 + 6.4e3] == pytest.approx(0.0, abs=1e-9))
    assert np.all(z[x < 950.0] == 2500.0)


def test_a_listric_fault_refuses_values_that_make_no_arc():
    """A detachment at or above the tip ellipse's centre, a non-positive one, and a reverse listric fault have no arc."""
    base = dict(center=(1500.0, 1000.0), strike=90.0, length=1200.0, throw=20.0)
    for bad in (dict(detach=0.0), dict(detach=-5.0), dict(detach=2000.0, z_center=2000.0), dict(detach=3000.0, reverse=True)):
        with pytest.raises(ValueError):
            Fault(**{**base, **bad})
    with pytest.raises(ValueError):                                    # the default centre is the stack's middle, known only there
        _build_geometry([flat_layer()], faults=[Fault(**base, detach=TOP + 10.0)])


def test_a_listric_fault_lists_every_contact_across_it_and_every_tread(tmp_path):
    """FAULTS for a listric fault: every face on which a cell meets the other side, wherever along the face, and every cell
    resting on the other side, as for a planar one (twisted trace, so a face may touch only between its pillars)."""
    layer = thin_stack()
    f = Fault(**{**TWISTED, "dip": 50.0}, detach=TOP + 1500.0, z_center=TOP + 25.0, name="F1")
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=[f], _faces=faces)
    to_grdecl(layer, tmp_path / "m.grdecl", faults=[f])
    rows = [r.split() for r in (tmp_path / "m.grdecl").read_text().split("\nFAULTS\n")[1].split("\n/\n")[0].splitlines()]
    listed = {(r[7].strip("'"), int(r[1]) - 1, int(r[3]) - 1, k) for r in rows for k in range(int(r[5]) - 1, int(r[6]))}
    side = faces[0][1]
    assert lateral_contacts(side, zc) and lateral_contacts(side, zc) <= listed
    treads = 0
    for i, j in zip(*np.nonzero((side > 0).any(axis=2) & (side < 0).any(axis=2))):
        live = [k for k in range(side.shape[2]) if act[i, j, k]]
        for a, b in zip(live, live[1:]):
            if side[i, j, a] * side[i, j, b] == -1:
                treads += 1
                assert ("Z", i, j, a) in listed
    assert treads
