"""Fault seal against independent references: hand-built pairs of columns whose multipliers and shale gouge ratios are
worked out outside the code (the series resistance behind Manzocchi et al. 1999, the slipped interval of Yielding et al.
1997), so that a wrong length, window or dip fails here and not only a golden number."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from resmill.export import _build_geometry
from resmill.fault_seal import Seal, face_multipliers
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


@pytest.mark.parametrize("up_first", [True, False])
@pytest.mark.parametrize("throw", [4.0, 7.3, 10.0, 16.0, 25.0, 40.0])
def test_sgr_averages_what_slid_past_in_each_column(throw, up_first):
    """The two columns hold different beds (the lower one's the other's upside down): every face's SGR is the mean of
    the upthrown column's [z - T, z] and the downthrown one's [z, z + T], for throws that are, and are not, a whole
    number of cells or of beds."""
    cells = beds(REVIEW)
    c = step(throw, up_first, nz=len(cells), vsh=np.stack([cells, cells[::-1]]))
    out = mults(c)
    rec = out["face_records"]
    assert len(rec) > 10
    assert np.allclose(rec["sgr"], reference_sgrs(c["zc"], c["vsh"], rec), rtol=1e-9)


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


def tread_columns(dip=60.0):
    """A normal fault whose plane crosses the left of two 1 m-celled columns (3 m of throw). Left: the upper five cells
    are hanging wall, 3 m lower than the right column's (k = 0 to 4, down to 8 m); the plane cuts out the next three
    (collapsed, inactive); its last two (8 to 10 m) are footwall. Right: footwall throughout. The two columns' clay
    fractions differ. The face where the left column's hanging wall rests on its footwall, at 8 m, is a tread (a Z face)
    through cells of 1 m; PERMZ is 20 mD, PERMX and PERMY 100."""
    depth = TOP + np.stack([[3, 4, 5, 6, 7, 8, 8, 8, 8, 9, 10], np.arange(11)]).astype(float)[:, None, :]
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


def test_a_faults_effective_multiplier_weights_each_face_by_its_unfaulted_transmissibility():
    """The five X faces (1 m of overlap x dy = 50 m, between 100 mD cells 25 m wide: 200 mD m) and the tread (dx x dy =
    1250 m2, between 20 mD cells 1 m thick: 25,000) weigh by the flow each would carry without the fault, so the one
    multiplier for the whole fault follows the tread."""
    out = mults(tread_columns(), 25.0, 50.0)
    rec, info = out["face_records"], out["faults"][0]
    d = 3.0 / math.sin(math.radians(60.0))
    lateral = rec["depth"] < TOP + 8.0 - 1e-9
    m = np.array([manzocchi(s, d, 100.0, 100.0, 25.0, 25.0) for s in rec["sgr"][lateral]]
                 + [manzocchi(0.375, d, 20.0, 20.0, 1.0, 1.0)])
    w = np.array([50.0 / (0.5 * 25.0 / 100.0 * 2.0)] * 5 + [1250.0 / (0.5 * 1.0 / 20.0 * 2.0)])
    assert info["effective"] == pytest.approx(float(np.dot(w, m) / w.sum()), rel=1e-9)
    assert info["effective"] == pytest.approx(manzocchi(0.375, d, 20.0, 20.0, 1.0, 1.0), rel=0.05)


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
