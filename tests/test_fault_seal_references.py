"""Fault seal against independent references: hand-built pairs of columns whose multipliers and shale gouge ratios are
worked out outside the code (the series resistance behind Manzocchi et al. 1999, the slipped interval of Yielding et al.
1997), so that a wrong length, window or dip fails here and not only a golden number."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from resmill.export import _build_geometry
from resmill.fault_seal import (Capillary, Seal, bretan_pressure, column_height, fault_blocks,
                                fault_transmissibility, face_multipliers, seal_capacity)
from resmill.faults import Fault
from resmill.layers.base import Layer

DZ, TOP = 0.5, 1000.0


def columns(depth, side, vsh=0.5, perm=100.0, dip=90.0):
    """The arguments of ``face_multipliers`` (bar the cell sizes) for columns with the cell interfaces ``depth``
    (nx, ny, nz + 1; m, k top-down), each wholly on its ``side`` (nx, ny; +1 hanging wall, -1 footwall; or one per cell,
    nx, ny, nz) of one fault of ``dip``. ``vsh`` and ``perm`` (one value for the three directions, or three) broadcast
    to (nx, ny, nz); cells thinner than 5 mm are inactive, as the exporter makes them."""
    depth = np.asarray(depth, dtype=float)
    nx, ny, nz = depth.shape[0], depth.shape[1], depth.shape[2] - 1
    full = lambda v: np.broadcast_to(np.asarray(v, dtype=float), (nx, ny, nz))
    perms = tuple(perm) if isinstance(perm, tuple) else (perm,) * 3
    side = np.asarray(side, dtype=np.int8)
    side = np.broadcast_to(side if side.ndim == 3 else side[:, :, None], (nx, ny, nz)).copy()
    zc = np.repeat(np.repeat(depth, 2, axis=0), 2, axis=1)
    act = (np.diff(depth, axis=2) > 5e-3).astype(np.int32)
    return dict(faces=[(SimpleNamespace(name="F1", dip=dip), side)], zc=zc, act=act, vsh=full(vsh),
                perms=tuple(full(p) for p in perms))


def layered(shifts, nz=40):
    """Flat cells of ``DZ`` m, the stack of each column ``shifts`` (nx, ny) m deeper than the top at ``TOP``."""
    return TOP + np.asarray(shifts, dtype=float)[:, :, None] + DZ * np.arange(nz + 1)


def per_column(v, along):
    """``v`` as ``columns`` takes it: one value, one per cell (nz,), or one per cell for each of the two columns
    (2, nz), the lower-index column first."""
    v = np.asarray(v, dtype=float)
    return (v[:, None, :] if along == "x" else v[None, :, :]) if v.ndim == 2 else v


def step(throw, up_first=True, along="x", nz=40, vsh=0.5, perm=100.0, dip=90.0):
    """Two columns, the downthrown one ``throw`` m deeper, along x (X faces) or y (Y faces); the upthrown one is the
    lower-index column when ``up_first`` (the downthrown one is the hanging wall of a normal fault). ``vsh`` and
    ``perm`` as :func:`per_column` says."""
    shift, side = np.array([0.0, throw]), np.array([-1, 1])
    if not up_first:
        shift, side = shift[::-1], side[::-1]
    shift, side = (shift[None, :], side[None, :]) if along == "y" else (shift[:, None], side[:, None])
    perm = tuple(per_column(p, along) for p in perm) if isinstance(perm, tuple) else per_column(perm, along)
    return columns(layered(shift, nz), side, per_column(vsh, along), perm, dip)


def fault_rock_k(sgr, d):
    """Fault-rock permeability (mD) of Manzocchi et al. 1999: log10 k_f = -4 SGR - 1/4 log10(D) (1 - SGR)^5."""
    return 10.0 ** (-4.0 * sgr - 0.25 * math.log10(d) * (1.0 - sgr) ** 5)


def resistances(sgr, d, ki, kj, li, lj, dt_ratio=66.0):
    """The series resistance between the two cell centres (per unit area) without the fault, and with the fault rock
    (thickness d / dt_ratio) in place of half its thickness of each cell. ``li`` and ``lj`` are the whole cells
    across the face."""
    tf = d / dt_ratio
    plain = 0.5 * li / ki + 0.5 * lj / kj
    return plain, 0.5 * (li - tf) / ki + tf / fault_rock_k(sgr, d) + 0.5 * (lj - tf) / kj


def manzocchi(sgr, d, ki, kj, li, lj, dt_ratio=66.0):
    """The multiplier of one face from first principles: the resistance without the fault over that with it."""
    plain, faulted = resistances(sgr, d, ki, kj, li, lj, dt_ratio)
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


@pytest.mark.parametrize("along", ["x", "y"])
@pytest.mark.parametrize("ki,kj", [(100.0, 100.0), (100.0, 400.0), (5.0, 2000.0)])
def test_the_multiplier_weights_each_cell_by_its_own_permeability(ki, kj, along):
    """The two cells have their own permeability (the lower-index column's first): the multiplier is the series
    resistance of both, as the closed form's L_i/k_i + L_j/k_j, on X and on Y faces."""
    perm = per_column(np.array([[ki] * 40, [kj] * 40]), along)
    out = mults(step(10.0, vsh=0.5, along=along, perm=(perm, perm, perm)), 25.0, 25.0)
    assert np.allclose(out["faults"][0]["mult"], manzocchi(0.5, 10.0, ki, kj, 25.0, 25.0), rtol=1e-9)


@pytest.mark.parametrize("along,k", [("x", 100.0), ("y", 400.0)])
def test_a_lateral_face_takes_the_permeability_in_its_own_direction(along, k):
    """PERMX 100, PERMY 400 and PERMZ 7 mD in every cell: flow across an X face is along x, between 100 mD cells, and
    across a Y face along y, between 400 mD cells (PERMZ plays no part in either)."""
    out = mults(step(10.0, vsh=0.5, along=along, perm=(100.0, 400.0, 7.0)), 25.0, 25.0)
    assert np.allclose(out["faults"][0]["mult"], manzocchi(0.5, 10.0, k, k, 25.0, 25.0), rtol=1e-9)


def test_a_sealed_face_keeps_a_multiplier_that_is_not_zero():
    """A fault factor of 10^-15 would take every multiplier far below 1e-12: the least a face gets, so that the face is
    sealed in practice and still a number the deck reads."""
    out = mults(step(10.0, vsh=0.5), seal=Seal(offset=-15.0))
    assert np.all(np.array(out["faults"][0]["mult"]) == 1e-12) and out["MULTX"].min() == 1e-12


def test_faults_on_the_same_face_multiply_their_multipliers():
    """Two faults that list the same faces (as crossing faults do at their junction) each give them their multiplier and
    the grid's MULTX is the product; each reads the throw off the grid, so it is not counted twice."""
    single = mults(step(10.0, vsh=0.5))
    both = step(10.0, vsh=0.5)
    both["faces"] = both["faces"] * 2
    out = mults(both)
    assert all(np.allclose(f["mult"], single["faults"][0]["mult"], rtol=1e-12) for f in out["faults"])
    assert np.allclose(out["MULTX"], single["MULTX"] ** 2, rtol=1e-12) and out["MULTX"].min() < 0.001


@pytest.mark.parametrize("dip", [90.0, 60.0, 45.0, 30.0])
def test_the_displacement_along_the_plane_is_the_throw_over_sin_dip(dip):
    """A vertical throw T on a plane dipping at ``dip`` slips the rock T / sin(dip) along it: that D sets the fault
    rock's permeability and its thickness D / 66 (10 m of throw: D = 10, 11.55, 14.14 and 20 m)."""
    out = mults(step(10.0, vsh=0.5, dip=dip))
    expected = manzocchi(0.5, 10.0 / math.sin(math.radians(dip)), 100.0, 100.0, 25.0, 25.0)
    assert np.allclose(out["faults"][0]["mult"], expected, rtol=1e-9)


REVIEW = [(0.9, 10.0), (0.05, 15.0), (0.9, 10.0), (0.05, 15.0)]       # (clay, thickness in m) top down: 50 m of beds


def beds(stack):
    """Clay fraction of each ``DZ``-thick cell, top down, of a list of (clay, thickness) beds."""
    return np.concatenate([np.full(int(round(h / DZ)), v) for v, h in stack])


def clay_in(depth, cells, lo, hi):
    """Clay thickness and thickness (m) of the part of the window [lo, hi] inside a column of clay fractions ``cells``
    between the interface depths ``depth``, by adding up each cell's overlap."""
    clay = thick = 0.0
    for top, base, v in zip(depth[:-1], depth[1:], cells):
        overlap = max(0.0, min(hi, base) - max(lo, top))
        clay, thick = clay + overlap * v, thick + overlap
    return clay, thick


def slipped_sgr(up, down, z, throw):
    """SGR at depth z of a fault with ``throw``: the beds that slid past the point are the upthrown column's [z - throw,
    z] and the downthrown column's [z, z + throw] (the same beds, a throw lower); each column's thickness-weighted clay
    fraction, averaged (Yielding, Freeman & Needham 1997; Lyon et al. 2005; Dincau 1998). ``up`` and ``down`` are
    (interface depths, clay fractions)."""
    means = [c / t for c, t in (clay_in(*up, z - throw, z), clay_in(*down, z, z + throw)) if t > 0.0]
    return sum(means) / len(means)


def reference_sgrs(zc, vsh, rec, along="x"):
    """The SGR of each face record ``rec`` on an X (``along`` "x") or Y edge, from the interface depths ``zc`` and clay
    ``vsh`` of its two columns: the throw is the largest offset of an interface across the pair and the upthrown
    column the shallower one there."""
    got = []
    for r in rec:
        i, j = int(r["i"]), int(r["j"])
        if along == "x":
            a, b = zc[2 * i + 1, 2 * j:2 * j + 2].mean(0), zc[2 * i + 2, 2 * j:2 * j + 2].mean(0)
            va, vb = vsh[i, j], vsh[i + 1, j]
        else:
            a, b = zc[2 * i:2 * i + 2, 2 * j + 1].mean(0), zc[2 * i:2 * i + 2, 2 * j + 2].mean(0)
            va, vb = vsh[i, j], vsh[i, j + 1]
        k = int(np.argmax(np.abs(a - b)))
        up, down = ((a, va), (b, vb)) if a[k] < b[k] else ((b, vb), (a, va))
        got.append(slipped_sgr(up, down, float(r["depth"]), float(np.abs(a - b).max())))
    return np.array(got)


@pytest.mark.parametrize("up_first", [True, False])
@pytest.mark.parametrize("throw,faces,expected", [(6.0, 36, 0.05), (10.0, 20, 0.05), (16.0, 12, 0.58125)])
def test_sgr_of_sand_against_sand_is_the_clay_of_the_beds_that_slid_past(throw, faces, expected, up_first):
    """15 m sands between 10 m shales (clay 0.05 and 0.9). With a throw of 6 or 10 m every sand-on-sand face has slid
    only sand past it (SGR 0.05; a window of 2 T centred on the face would also take in shale); with 16 m, more than a
    sand's thickness, 6 m of sand and the whole 10 m shale have (0.05 x 6 + 0.9 x 10) / 16 = 0.58125. The sand-on-sand
    faces are those where the sands overlap: 2 x (15 - T) m of them, 2 x 6 m at 16 m."""
    cells = beds(REVIEW)
    out = mults(step(throw, up_first, nz=len(cells), vsh=cells, perm=np.where(cells > 0.5, 0.01, 100.0)))
    rec = out["face_records"]
    net = rec["perm"] >= 1.0
    assert net.sum() == faces
    assert np.allclose(rec["sgr"][net], expected, rtol=1e-9)


@pytest.mark.parametrize("up_first", [True, False])
def test_sgr_at_a_depth_of_a_20_m_throw_is_worked_by_hand(up_first):
    """20 m of throw, the face at 40.25 m below the top of the upthrown beds: the beds that slid past it span 20.25 to
    40.25 m, 4.75 m of sand, the 10 m shale and 5.25 m of sand: (0.05 x 10 + 0.9 x 10) / 20 = 0.475."""
    cells = beds(REVIEW)
    out = mults(step(20.0, up_first, nz=len(cells), vsh=cells))
    rec = out["face_records"]
    here = np.isclose(rec["depth"], TOP + 40.25)
    assert here.sum() == 1 and rec["sgr"][here] == pytest.approx(0.475, rel=1e-9)


@pytest.mark.parametrize("along", ["x", "y"])
@pytest.mark.parametrize("up_first", [True, False])
@pytest.mark.parametrize("throw", [4.0, 7.3, 10.0, 16.0, 25.0, 40.0])
def test_sgr_averages_what_slid_past_in_each_column(throw, up_first, along):
    """The two columns hold different beds (the lower one's the other's upside down): every face's SGR is the mean of
    the upthrown column's [z - T, z] and the downthrown one's [z, z + T], for throws that are, and are not, a whole
    number of cells or of beds, on X and on Y faces."""
    cells = beds(REVIEW)
    c = step(throw, up_first, along, nz=len(cells), vsh=np.stack([cells, cells[::-1]]))
    rec = mults(c)["face_records"]
    assert len(rec) > 10
    assert np.allclose(rec["sgr"], reference_sgrs(c["zc"], c["vsh"], rec, along), rtol=1e-9)


def test_sgr_takes_what_the_window_holds_where_it_runs_off_the_stack():
    """A throw that dies upward (4 m near the top, 10 m below 11 m) with the window set by the largest offset, 10 m:
    near the top it reaches above the stack, and the clay of what is there is the mean over the thickness it holds, not
    over 10 m."""
    cells = beds(REVIEW)
    nz = len(cells)
    depth = layered(np.zeros((2, 1)), nz)
    depth[1, 0] += np.interp(np.arange(nz + 1), [10, 22], [4.0, 10.0])
    c = columns(depth, np.array([[-1], [1]]), vsh=per_column(np.stack([cells, cells[::-1]]), "x"))
    rec = mults(c)["face_records"]
    assert np.any(rec["depth"] - 10.0 < TOP) and np.any(rec["depth"] - 10.0 > TOP)
    assert np.allclose(rec["sgr"], reference_sgrs(c["zc"], c["vsh"], rec), rtol=1e-9)


@pytest.mark.parametrize("reverse,wall,share", [(False, 1, 0.2), (False, -1, 0.7), (True, 1, 0.7), (True, -1, 0.2)])
def test_sgr_follows_the_slipped_interval_on_normal_and_reverse_faults(reverse, wall, share):
    """On faults of the real geometry (normal and reverse, either wall on the +x side, the hanging wall taking 20 or
    70 % of the displacement) the upthrown column is the lower-index one in two of the four: every face's SGR is the
    slipped-interval value read from the depths, and the sand-on-sand faces of a 10 m throw in 15 m sands have 0.05."""
    cells = beds(REVIEW)
    nz = len(cells)
    layer = Layer(40, 8, nz, 1000.0, 200.0, nz * DZ, top_depth=TOP, kzkx=0.1)
    layer.poro_mat = np.ones((40, 8, nz)) * 0.2
    layer.perm_mat = np.ones((40, 8, nz)) * np.where(cells[::-1] > 0.5, 0.01, 100.0)
    vsh = np.ones((40, 8, nz)) * cells
    f = Fault(center=(510.0, 100.0), strike=90.0, length=2e7, throw=10.0, dip=90.0, hanging_wall=wall, hw_share=share,
              drag=(0.0, 0.0), reverse=reverse, name="F1")
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=[f], _faces=faces)
    perm = np.asarray(layer.perm_mat)[:, :, ::-1]
    rec = face_multipliers(faces, zc, act, vsh, (perm, perm, perm), 25.0, 25.0, Seal())["face_records"]
    assert len(rec) > 100 and np.all(rec["axis"] == 0)
    assert np.allclose(rec["sgr"], reference_sgrs(zc, vsh, rec), rtol=1e-9)
    net = rec["perm"] >= 1.0
    assert net.sum() == 8 * 20 and np.allclose(rec["sgr"][net], 0.05, atol=1e-6)


def tread_columns(dip=60.0, throw=3.0):
    """A normal fault whose plane crosses the left of two 1 m-celled columns (``throw`` 3 m, or another below 4). Left:
    the upper five cells are hanging wall, ``throw`` lower than the right column's (k = 0 to 4, down to 5 + throw m);
    the plane cuts out the next three (collapsed, inactive); its last two (to 10 m) are footwall. Right: footwall
    throughout. The two columns' clay fractions differ. The face where the left column's hanging wall rests on its
    footwall, at 5 + throw m, is a tread (a Z face) through cells of 1 m (the lower one 1 - (throw - 3) m when the
    throw is not 3); PERMZ is 20 mD, PERMX and PERMY 100."""
    left = np.maximum.accumulate(np.concatenate([throw + np.arange(6), np.arange(6, 11)]))
    depth = TOP + np.stack([left, np.arange(11)]).astype(float)[:, None, :]
    side = np.stack([[1] * 5 + [-1] * 5, [-1] * 10])[:, None, :]
    vsh = np.stack([[0.9] * 8 + [0.2, 0.1], np.arange(10) / 10.0])
    return columns(depth, side, per_column(vsh, "x"), (100.0, 100.0, 20.0), dip)


def test_a_tread_takes_the_sgr_of_its_column_pair_at_its_depth_and_the_thickness_of_its_cells():
    """The tread at 8 m counts for the pair of columns beside it (the left, downthrown, one deeper by 3 m): the beds
    that slid past it are the right column's 5 to 8 m (clay 0.5, 0.6, 0.7: mean 0.6) and the left column's 8 to 11 m,
    of which it holds 8 to 10 m (0.2, 0.1: 0.15), so SGR (0.6 + 0.15) / 2 = 0.375; the cells above and below it are 1 m
    thick: the multiplier of Manzocchi's formula with L_i = L_j = 1 m and D = 3 / sin 60 degrees between 20 mD cells."""
    out = mults(tread_columns(), 25.0, 50.0)
    rec = out["face_records"]
    tread = np.isclose(rec["depth"], TOP + 8.0)
    assert tread.sum() == 1 and rec["sgr"][tread][0] == pytest.approx(0.375, rel=1e-9)
    d = 3.0 / math.sin(math.radians(60.0))
    assert out["MULTZ"][0, 0, 4] == pytest.approx(manzocchi(0.375, d, 20.0, 20.0, 1.0, 1.0), rel=1e-9)
    assert np.count_nonzero(out["MULTZ"] != 1.0) == 1 and np.count_nonzero(out["MULTX"] != 1.0) == 5
    assert np.allclose(rec["sgr"], reference_sgrs(tread_columns()["zc"], tread_columns()["vsh"], rec), rtol=1e-9)


@pytest.mark.parametrize("throw", [3.0, 3.3])
def test_a_faults_effective_multiplier_weights_each_face_by_its_unfaulted_transmissibility(throw):
    """The five X faces (1 m of the cell juxtaposed, however many cells it is juxtaposed with, x dy = 50 m, between
    100 mD cells 25 m wide: 200 mD m) and the tread (dx x dy = 1250 m2, between 20 mD cells 1 m and, with 3.3 m of
    throw, 0.7 m thick: 25,000 and 29,400) weigh by the flow each would carry without the fault, so the one multiplier
    for the whole fault follows the tread."""
    c = tread_columns(throw=throw)
    out = mults(c, 25.0, 50.0)
    rec, info = out["face_records"], out["faults"][0]
    d = throw / math.sin(math.radians(60.0))
    lateral = rec["depth"] < TOP + 5.0 + throw - 0.25
    lj = 1.0 - (throw - 3.0)
    sgr = reference_sgrs(c["zc"], c["vsh"], rec)
    m = np.array([manzocchi(s, d, 100.0, 100.0, 25.0, 25.0) for s in sgr[lateral]]
                 + [manzocchi(sgr[~lateral][0], d, 20.0, 20.0, 1.0, lj)])
    w = np.array([50.0 / (0.5 * 25.0 / 100.0 * 2.0)] * 5 + [1250.0 / (0.5 / 20.0 + 0.5 * lj / 20.0)])
    assert lateral.sum() == 5 and np.allclose(info["mult"], m, rtol=1e-9)
    assert info["effective"] == pytest.approx(float(np.dot(w, m) / w.sum()), rel=1e-9)
    assert info["effective"] == pytest.approx(m[-1], rel=0.2)


def test_a_fault_rock_thicker_than_a_tight_cell_and_more_permeable_leaves_the_face_open():
    """Clean gouge (k_f 0.56 mD at 10 m of displacement, t_f 0.15 m) against 0.01 mD cells 0.05 m wide: the fault rock
    would replace more than the whole cell with a better conductor, so the formula's bracket falls to -1.98 (a negative
    resistance). The face is open (1), not sealed by the sign: also when the fault's factor would scale it."""
    for seal in (Seal(), Seal(offset=-0.6)):
        out = mults(step(10.0, vsh=0.0, perm=0.01), dx=0.05, dy=25.0, seal=seal)
        assert len(out["faults"][0]["mult"]) == 20
        assert np.all(np.array(out["faults"][0]["mult"]) == 1.0) and out["faults"][0]["effective"] == 1.0
        assert np.all(out["MULTX"] == 1.0)


def test_a_fault_rock_a_little_better_than_the_cells_gives_more_than_1_which_the_factor_scales_before_the_cap():
    """The same gouge against 0.01 mD cells 25 m wide: the bracket 0.994, T = 1.006. Without a factor the cap holds it
    at 1; with the fault's factor 10^-0.6 it is 1.006 x 0.251 = 0.253, as the multiplier of the physics was always
    scaled before it was capped."""
    base = mults(step(10.0, vsh=0.0, perm=0.01), dx=25.0, dy=25.0)["faults"][0]["mult"]
    scaled = mults(step(10.0, vsh=0.0, perm=0.01), dx=25.0, dy=25.0, seal=Seal(offset=-0.6))["faults"][0]["mult"]
    assert len(base) == 20 and np.all(np.array(base) == 1.0)
    bracket = 1.0 + (10.0 / 66.0) * (2.0 / fault_rock_k(0.0, 10.0) - 2.0 / 0.01) / (2.0 * 25.0 / 0.01)
    assert 0.99 < bracket < 1.0 and np.allclose(scaled, 10.0 ** -0.6 / bracket, rtol=1e-9)


@pytest.mark.parametrize("along", ["x", "y"])
def test_a_cells_net_status_is_its_horizontal_permeability_on_every_lateral_face(along):
    """Net reservoir is a property of the bed, not of the face's direction: the record's permeability, which
    ``Capillary.net_perm`` is tested against, is the lower of the two cells' horizontal permeabilities, the geometric
    mean of PERMX and PERMY (sqrt(25 x 4) = 10 mD in both columns here, on X and on Y faces alike, however low PERMZ
    is), and the lower of the two cells' where they differ."""
    rec = mults(step(10.0, True, along, perm=(25.0, 4.0, 0.001)))["face_records"]
    assert len(rec) == 20 and np.all(rec["perm"] == 10.0)
    px, py = np.array([[25.0] * 40, [1.0] * 40]), np.array([[4.0] * 40, [1.0] * 40])
    rec = mults(step(10.0, True, along, perm=(per_column(px, along), per_column(py, along), 0.001)))["face_records"]
    assert len(rec) == 20 and np.all(rec["perm"] == 1.0)          # sqrt(25 x 4) = 10 above, sqrt(1 x 1) = 1 below


def test_a_tread_is_net_on_the_horizontal_permeability_of_its_cells():
    """The tread's cells have PERMZ 20 mD against 100 mD horizontally (and 0.1 mD against 100 below): they are net
    reservoir on every face they have, the tread included."""
    for kz in (20.0, 0.1):
        c = tread_columns()
        c["perms"] = (c["perms"][0], c["perms"][1], np.full(c["vsh"].shape, kz))
        rec = mults(c, 25.0, 50.0)["face_records"]
        assert len(rec) == 6 and np.all(rec["perm"] == 100.0)


def test_fault_transmissibility_is_the_series_resistance_ratio_where_the_resistance_is_positive():
    """For 400 random faces (SGR, displacement from 0.1 to 300 m, permeabilities over seven decades, cells 0.1 to 60 m
    long, so the fault rock is sometimes thicker than the cell) the closed form of Manzocchi et al. is the resistance
    without the fault over the resistance with it, element by element on arrays; where the resistance with the fault
    would be zero or negative it is infinite, an open face."""
    rng = np.random.default_rng(3)
    sgr, d = rng.uniform(0.0, 0.9, 400), 10.0 ** rng.uniform(-1.0, 2.5, 400)
    ki, kj = 10.0 ** rng.uniform(-3.0, 4.0, (2, 400))
    li, lj = 10.0 ** rng.uniform(-1.0, 1.8, (2, 400))
    got = fault_transmissibility(sgr, d, ki, kj, li, lj)
    plain, faulted = np.array([resistances(*args) for args in zip(sgr, d, ki, kj, li, lj)]).T
    ok = faulted > 0.0
    assert 3 <= (~ok).sum() < 100
    assert np.allclose(got[ok], plain[ok] / faulted[ok], rtol=1e-9) and np.all(np.isinf(got[~ok]))
    assert fault_transmissibility(0.5, 10.0, 100.0, 100.0, 25.0, 25.0) == pytest.approx(0.01595, rel=1e-3)
    assert fault_transmissibility(0.5, 10.0, 100.0, 100.0, 25.0, 25.0, dt_ratio=170.0) > 0.01595


@pytest.mark.parametrize("delta_rho,metres", [(100.0, 102.0), (200.0, 51.0), (300.0, 34.0), (400.0, 25.5),
                                              (500.0, 20.4), (700.0, 14.6), (800.0, 12.7)])
def test_a_bar_holds_the_oil_column_of_the_researchs_table(delta_rho, metres):
    """capillary_fault_seal.md section 2: 1 bar supports 102 m of oil at a density contrast of 100 kg/m3, 51 m at 200,
    34 at 300, 25.5 at 400, 20.4 at 500, 14.6 at 700 and 12.7 at 800, and 4 bar four times as much."""
    assert column_height(1.0, delta_rho) == pytest.approx(metres, abs=0.1)
    assert column_height(np.array([2.0, 4.0]), delta_rho) == pytest.approx([2.0 * metres, 4.0 * metres], abs=0.4)


# Bretan, Yielding & Jones (2003) eq. 1 in bar (capillary_fault_seal.md section 2): SGR in %, then C = 0.5, 0.25 and 0
BRETAN_TABLE = [(0, 0.32, 0.56, 1.00), (10, 0.74, 1.32, 2.35), (15, 1.14, 2.02, 3.59), (20, 1.74, 3.10, 5.50),
                (25, 2.67, 4.74, 8.43), (30, 4.08, 7.26, 12.9), (40, 9.58, 17.0, 30.3), (50, 22.5, 40.0, 71.1),
                (60, 52.8, 93.8, 167.0)]


@pytest.mark.parametrize("percent,shallow,middle,deep", BRETAN_TABLE)
def test_bretans_envelope_reproduces_the_researchs_table(percent, shallow, middle, deep):
    """All 27 entries of the research's table of eq. 1 (burial below 3 km, 3 to 3.5 km and deeper; the table rounds to
    three figures, 0.316 to 0.32 being the worst, 1.2 %)."""
    got = bretan_pressure(percent / 100.0, np.array([2000.0, 3200.0, 4000.0]))
    assert got == pytest.approx([shallow, middle, deep], rel=0.013)


def test_bretans_envelope_reaches_the_oil_plateau_of_3_bar_where_the_research_says():
    """Eq. 1 reaches 3 bar of oil at SGR 26, 20 and 13 % for burial below 3 km, 3 to 3.5 km and deeper."""
    for sgr, burial in ((0.26, 2000.0), (0.20, 3200.0), (0.13, 4000.0)):
        assert float(bretan_pressure(sgr, burial)) == pytest.approx(3.0, abs=0.15)


def zema_column(sgr, cap):
    """The column (m) of Zema's fluids that a face of ``sgr`` holds at 1,500 m burial (C 0.5): a 15 m oil leg
    (4.3 kPa/m) under gas (8.0 kPa/m) share the face's capacity, the oil's 0.645 bar first."""
    leg, oil, gas = 15.0, 0.043, 0.080                                   # m, bar/m, bar/m
    return leg + (float(seal_capacity(sgr, 1500.0, cap)) - oil * leg) / gas


def test_zemas_capacity_gives_the_published_72_m_of_column():
    """Lyon et al. (2005), the Zema prospect, Otway Basin (under 3 km of burial): slipped-interval SGR 33 % at the leak
    point at the top of the fault, oil 0.53, gas 0.143 and water 0.97 g/cm3 (4.3 and 8.0 kPa/m), a 15 m oil leg under
    the gas. Eq. 1 gives 5.27 bar, 0.645 of them held by the oil leg and the rest by 57.8 m of gas: 72.9 m, the paper's
    72 m within 6 m; 32 and 34 % give the 67.5 and 78.7 m of the research."""
    cap = Capillary(delta_rho=440.0, plateau=100.0)                       # the plateau out of the way
    assert zema_column(0.33, cap) == pytest.approx(72.9, abs=0.15) and abs(zema_column(0.33, cap) - 72.0) <= 6.0
    assert zema_column(0.32, cap) == pytest.approx(67.5, abs=0.15)
    assert zema_column(0.34, cap) == pytest.approx(78.7, abs=0.15)


def ramp_blocks(throw, delta_rho, nx=30, ny=7, nz=60, dz=1.0, along="x"):
    """A footwall closure against a fault, by hand: columns of 50 m on a map 30 by 7 whose middle row rises 4 m a column
    from the west edge (56 m deeper than at the fault) to the fault between columns 14 and 15, where the 60 m of clean
    sand (clay 0, 100 mD) of the footwall meets the hanging wall's, ``throw`` m lower, which rises 2 m a column away
    from it to the east edge (so the hanging wall holds no trap), and every row away from the middle is 20 m deeper than
    the one before (so nothing escapes sideways). ``fault_blocks`` of that under ``Capillary(delta_rho)``; with
    ``along="y"`` the map is turned over, the fault runs along x and its faces are Y faces."""
    i, j = np.arange(nx)[:, None], np.arange(ny)[None, :]
    top = np.where(i <= 14, 4.0 * (14 - i), throw - 2.0 * (i - 15)) + 20.0 * np.abs(j - ny // 2)
    side = np.where(i <= 14, -1, 1) + 0 * j
    if along == "y":
        top, side = top.T, side.T
    c = columns(TOP + top[:, :, None] + dz * np.arange(nz + 1), side, vsh=0.0)
    mult = face_multipliers(**c, dx=50.0, dy=50.0, seal=Seal())
    return fault_blocks(c["zc"], c["act"], c["faces"], mult, 50.0, 50.0, Capillary(delta_rho=delta_rho))


@pytest.mark.parametrize("along", ["x", "y"])
@pytest.mark.parametrize("delta_rho", [400.0, 300.0])
@pytest.mark.parametrize("throw", [10.0, 20.0, 30.0])
def test_a_self_juxtaposed_footwall_trap_holds_about_the_throw(throw, delta_rho, along):
    """Gibson (1994, Columbus Basin; research check 1): where the footwall sand is self-juxtaposed (sand against sand
    from the hanging wall's top down) the trap holds about the fault's throw, plus the seal at most about 20 m of clean
    sand gives (0.5 bar: 12.7 m at a density contrast of 400, 17.0 m at 300, and half a cell for the window's middle):
    the contact lies the throw plus the floor's column below the footwall's sand top at the fault, the oil leaking
    into a hanging wall that has no closure of its own (one block, limited by the leak); the same across X and Y
    faces."""
    blocks = ramp_blocks(throw, delta_rho, along=along)
    assert len(blocks) == 1 and blocks[0]["limited_by"] == "leak" and blocks[0]["point"][0 if along == "x" else 1] == 15
    held = blocks[0]["contact_depth"] - TOP                           # the footwall's sand top at the fault is TOP
    assert held == pytest.approx(throw + 0.5 + float(column_height(0.5, delta_rho)), abs=1e-9)
    assert throw < held <= throw + 20.0
    assert delta_rho != 400.0 or abs(held - throw) <= 15.0


def chain_blocks(throw1, throw2, delta_rho=400.0, nx=30, ny=7, nz=60):
    """Three fault blocks in a row, by hand, the ramp of :func:`ramp_blocks` twice: A (columns 0 to 9) rises 4 m a
    column to fault 1, between columns 9 and 10, where its 60 m of clean sand (clay 0, 100 mD) meets B's, ``throw1``
    lower; B (10 to 19) rises 2 m a column to fault 2, where it meets C's, ``throw2`` lower; C (20 to 29) rises on to
    the map's edge and holds no trap. Every row away from the middle is 20 m deeper than the one before. The blocks of
    ``fault_blocks`` under ``Capillary(delta_rho)``, shallowest crest first: B (its crest, at fault 2, lies ``throw1``
    - 18 m below the top of A at fault 1), then A."""
    i, j = np.arange(nx)[:, None], np.arange(ny)[None, :]
    west, middle, east = 4.0 * (9 - i), throw1 - 2.0 * (i - 10), throw1 - 18.0 + throw2 - 2.0 * (i - 20)
    top = np.select([i <= 9, i <= 19], [west, middle], east) + 20.0 * np.abs(j - ny // 2)
    c = columns(TOP + top[:, :, None] + np.arange(nz + 1.0), np.where(i <= 9, -1, 1) + 0 * j, vsh=0.0)
    second = np.broadcast_to((np.where(i <= 19, -1, 1) + 0 * j).astype(np.int8)[:, :, None], c["act"].shape).copy()
    c["faces"] = c["faces"] + [(SimpleNamespace(name="F2", dip=90.0), second)]
    mult = face_multipliers(**c, dx=50.0, dy=50.0, seal=Seal())
    return fault_blocks(c["zc"], c["act"], c["faces"], mult, 50.0, 50.0, Capillary(delta_rho=delta_rho))


@pytest.mark.parametrize("throw1,throw2,joined", [(10.0, 10.0, False), (10.0, 25.0, True)])
def test_oil_leaks_down_a_chain_of_fault_blocks_from_one_window_to_the_next(throw1, throw2, joined):
    """Each fault leaks at its own weakest window, the first sand against sand (the throw and half a cell below the
    sand top of the block it holds), once the contact stands the floor's column below it (0.5 bar at a density
    contrast of 400: 12.7 m). With 10 m of throw at both faults B's window, 2.5 m below the top of A, is shallower than
    A's, 10.5 m: B keeps its oil down to 15.2 m and passes the rest on to C, A spills across fault 1 into B at 23.2 m,
    and there are two accumulations, A's contact 8 m deeper than B's. With 25 m of throw at fault 2 its window, 17.5 m,
    is the deeper one: B fills to 30.2 m, A's oil stands as high in B as B's does, and one contact serves both."""
    b, a = chain_blocks(throw1, throw2)
    window = float(column_height(0.5, 400.0)) + 0.5
    leak_b = TOP + throw1 - 18.0 + throw2 + window                  # fault 2: B's top there, throw2, half a cell, floor
    leak_a = TOP + throw1 + window                                  # fault 1: A's top there is TOP
    assert b["crest_depth"] == TOP + throw1 - 18.0 and a["crest_depth"] == TOP
    assert b["contact_depth"] == pytest.approx(leak_b, abs=1e-9)
    assert a["contact_depth"] == pytest.approx(max(leak_a, leak_b), abs=1e-9)
    assert a["limited_by"] == b["limited_by"] == "leak" and (a["group"] == b["group"]) == joined
    assert a["height"] == pytest.approx(a["contact_depth"] - TOP)
    assert b["height"] == pytest.approx(leak_b - b["crest_depth"])
    if not joined:
        assert a["point"] == (10, 3) and b["point"] == (20, 3)          # A leaks into B, B into C
        assert (a["contact_depth"] - b["contact_depth"]) == pytest.approx(8.0, abs=1e-9)


def test_the_effective_multiplier_weighs_x_and_y_faces_by_their_own_areas():
    """A corner of a fault: one column (1, 0) 10 m down, between its neighbours, so the fault has 20 X faces on 25 m
    cells (face area 0.5 m x dy = 25 m2) and 20 Y faces on 50 m cells (0.5 m x dx = 12.5 m2): weights 25 / (25/100) =
    100 and 12.5 / (50/100) = 25 each, so the one multiplier is (100 m_x + 25 m_y) / 125."""
    shift, side = np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([[-1, -1], [1, -1]])
    out = mults(columns(layered(shift), side, vsh=0.5), dx=25.0, dy=50.0)
    m_x = manzocchi(0.5, 10.0, 100.0, 100.0, 25.0, 25.0)
    m_y = manzocchi(0.5, 10.0, 100.0, 100.0, 50.0, 50.0)
    assert len(out["faults"][0]["mult"]) == 40 and m_y > 1.5 * m_x
    assert out["faults"][0]["effective"] == pytest.approx((100.0 * m_x + 25.0 * m_y) / 125.0, rel=1e-9)


@pytest.mark.parametrize("dt_ratio", [30.0, 66.0, 170.0])
def test_the_seals_ratio_of_displacement_to_fault_rock_thickness_is_used(dt_ratio):
    """Manzocchi's median ratio is 66 and the harmonic mean 170: Seal.dt_ratio sets t_f = D / ratio."""
    out = mults(step(10.0, vsh=0.5), seal=Seal(dt_ratio=dt_ratio))
    assert np.allclose(out["faults"][0]["mult"], manzocchi(0.5, 10.0, 100.0, 100.0, 25.0, 25.0, dt_ratio), rtol=1e-9)


def tied_blocks(throw=10.0):
    """Two blocks whose contacts and the fault's weakest window lie at exactly the same depth, by hand: the ramp of
    :func:`ramp_blocks` with a trap of its own east of the fault (crest 4 m above the fault's top, a saddle at ``throw``
    + 0.5 m, then a gentle rise to the east edge, the crest still the block's shallowest point), and no seal at all
    (floor and plateau 0, so a face leaks at its own depth: the sand's top at the fault plus ``throw`` plus half a cell,
    also ``throw`` + 0.5). The depths are multiples of 0.5 m, exact in floating point."""
    nx, ny, nz = 30, 7, 60
    i, j = np.arange(nx)[:, None], np.arange(ny)[None, :]
    east = throw + np.array([0.0, -2.0, -4.0, -2.0, 0.5, -1.5, -2.0, -2.5, -3.0, -3.5, -3.5, -3.5, -3.5, -3.5, -3.5])
    top = np.where(i <= 14, 4.0 * (14 - i), east[np.clip(i - 15, 0, 14)]) + 20.0 * np.abs(j - ny // 2)
    c = columns(TOP + top[:, :, None] + np.arange(nz + 1.0), np.where(i <= 14, -1, 1) + 0 * j, vsh=0.0)
    mult = face_multipliers(**c, dx=50.0, dy=50.0, seal=Seal())
    cap = Capillary(delta_rho=300.0, floor=0.0, plateau=0.0)
    return fault_blocks(c["zc"], c["act"], c["faces"], mult, 50.0, 50.0, cap), mult


def test_blocks_that_meet_at_the_level_of_their_window_are_one_accumulation():
    """The footwall trap leaks at its window (floor 0: at the window's own depth, 10.5 m below the sand top at the
    fault) into a hanging-wall trap that spills over its saddle at that very depth: both contacts, and the edge's pass
    level between them, are TOP + 10.5 m exactly, and oil at that level is one accumulation, one group."""
    blocks, mult = tied_blocks()
    assert len(blocks) == 2
    assert [b["contact_depth"] for b in blocks] == [TOP + 10.5, TOP + 10.5]
    assert blocks[0]["group"] == blocks[1]["group"] == 0
    assert np.min(mult["face_records"]["depth"]) == TOP + 10.5


@pytest.mark.parametrize("along", ["x", "y"])
def test_a_fault_without_throw_has_no_face_to_seal_or_to_leak(along):
    """Cells meeting the other side's with nothing slipped past them (no offset: the throw has died out) have no SGR and
    no window: no multiplier, no record, the fault's effective multiplier 1."""
    out = mults(step(0.0, vsh=0.5, along=along))
    assert len(out["face_records"]) == 0 and out["faults"][0]["mult"] == [] and out["faults"][0]["effective"] == 1.0
    assert all(np.all(out[key] == 1.0) for key in ("MULTX", "MULTY", "MULTZ"))
