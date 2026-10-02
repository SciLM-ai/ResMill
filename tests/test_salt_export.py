"""Salt in the exported grid: its cells are inactive (the contact is sealed), the wall is a staircase on cell faces, an
overhang leaves active cells beneath salt, faults and erosion still apply, and the grid must resolve the upturn."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from resmill import salt as sl
from resmill import structure as st
from resmill.export import _build_geometry, to_grdecl, to_pyvista
from resmill.faults import Fault
from resmill.layers.base import Layer
from resmill.plotting import plot_section
from tests.test_export import prop_cube, read_grdecl, zcorn_cube

NX, NY, DX, TOP = 60, 50, 20.0, 2000.0
CX, CY, R = 0.5 * NX * DX, 0.5 * NY * DX, 300.0


def layer(nz=4, dz=2.5, nx=NX, ny=NY, dx=DX):
    L = Layer(nx, ny, nz, nx * dx, ny * dx, nz * dz, top_depth=TOP, kzkx=0.1)
    L.poro_mat = np.full((nx, ny, nz), 0.2)
    L.perm_mat = np.full((nx, ny, nz), 100.0)
    return L


def centres(nx=NX, ny=NY, dx=DX):
    return np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dx, indexing="ij")


def test_cells_in_a_vertical_stock_are_inactive_and_the_wall_is_a_staircase():
    _, _, _, act = _build_geometry([layer()], salt=sl.salt_body((CX, CY), (R, R)))
    xc, yc = centres()
    inside = (xc - CX) ** 2 + (yc - CY) ** 2 < R ** 2                    # by hand, cell centres in the circle
    assert (act == (~inside)[:, :, None]).all()
    assert abs(inside.sum() * DX ** 2 - math.pi * R ** 2) < DX * 2.0 * math.pi * R   # staircase error under a cell x perimeter


def test_the_written_file_has_the_inactive_cells_and_untouched_properties(tmp_path):
    body = sl.salt_body((CX, CY), (R, R))
    L = layer()
    plain = read_grdecl(to_grdecl(L, tmp_path / "plain.grdecl"))
    salty = read_grdecl(to_grdecl(L, tmp_path / "salt.grdecl", salt=body))
    act = prop_cube(salty, "ACTNUM", NX, NY, 4)
    xc, yc = centres()
    assert (act == (~((xc - CX) ** 2 + (yc - CY) ** 2 < R ** 2))[:, :, None]).all()
    for key in ("PORO", "PERMX", "PERMY", "PERMZ", "ZCORN", "COORD"):
        assert salty[key] == plain[key]                                # the cells keep their values, only ACTNUM changes


def test_the_salt_has_no_pore_volume_and_the_rest_of_the_model_keeps_its_own(tmp_path):
    """A simulator's pore volume is porosity x cell volume over the active cells. Cell volumes come from the file's ZCORN
    (vertical pillars: plan area times the mean corner thickness) of a folded layer: the model's pore volume is the whole
    layer's less that of the cells whose centres lie in the stock (counted by hand), every salt cell holds none, and what
    the stock removes is pi R^2 x thickness x porosity to the staircase's error."""
    nz, dz, phi = 8, 2.5, 0.2
    body = sl.salt_body((CX, CY), (R, R))
    L = layer(nz=nz, dz=dz)
    path = to_grdecl(L, tmp_path / "salt.grdecl", structure=st.dome(40.0, 500.0, center=(CX, CY)), salt=body)
    blocks = read_grdecl(path)
    zc = zcorn_cube(blocks, NX, NY, nz)
    thick = (zc[:, :, 1::2] - zc[:, :, 0::2]).reshape(NX, 2, NY, 2, nz).mean(axis=(1, 3))
    pv = prop_cube(blocks, "PORO", NX, NY, nz) * DX * DX * thick               # what the cells would hold if all were active
    act = prop_cube(blocks, "ACTNUM", NX, NY, nz)
    xc, yc = centres()
    salt = ((xc - CX) ** 2 + (yc - CY) ** 2 < R ** 2)[:, :, None] * np.ones(nz, dtype=bool)
    held = act * pv                                                            # the simulator's PORV
    assert (held[salt] == 0.0).all() and pv[salt].min() > 0.0                  # the file keeps their PORO: ACTNUM is what zeroes it
    assert held.sum() == pytest.approx(pv.sum() - pv[salt].sum(), rel=1e-12)
    assert thick.sum(axis=2).mean() == pytest.approx(nz * dz, rel=1e-9)        # the dome shifts, it does not stretch
    assert pv.sum() - held.sum() == pytest.approx(phi * math.pi * R ** 2 * nz * dz, abs=phi * DX * 2.0 * math.pi * R * nz * dz)


@pytest.mark.parametrize("dip", [30.0, 60.0, 85.0])
def test_a_steep_upturn_shears_cells_but_neighbouring_columns_share_their_corners(dip):
    """An upturn is a smooth shift of the whole stack at each pillar, so the two columns that meet at a pillar take the same
    corner depths there: no step between them, however steep the dip, and so no non-neighbour connection when Flow reads the
    grid (the OPM bench of step 5 counts 0 NNCs at 30 to 85 degrees; only faults make them). The cells shear instead: the top
    steps by more than a cell's thickness between neighbouring pillars, and by no more than the upturn's relief."""
    L = layer(nz=12, dz=5.0)
    body = sl.salt_body((CX, CY), (R, R), z_ref=TOP + 30.0)
    up = sl.salt_upturn(body, dip, 400.0, power=max(1.0, 400.0 * math.tan(math.radians(dip)) / 250.0))
    _, _, zc, act = _build_geometry([L], structure=up, salt=body)
    assert np.array_equal(zc[1:-1:2], zc[2::2]) and np.array_equal(zc[:, 1:-1:2], zc[:, 2::2])     # shared pillars, in x and in y
    drop = np.abs(np.diff(zc[::2, ::2, 0], axis=0)).max()                       # the steepest step of the top between pillars
    assert 5.0 < drop <= 250.0 + 1e-6                                           # more than a cell's thickness (they shear), within the relief


def test_no_salt_changes_nothing(tmp_path):
    a = to_grdecl(layer(), tmp_path / "a.grdecl", structure=st.dome(30.0, 400.0))
    b = to_grdecl(layer(), tmp_path / "b.grdecl", structure=st.dome(30.0, 400.0), salt=None)
    assert a.read_bytes().split(b"\n", 1)[1] == b.read_bytes().split(b"\n", 1)[1]


def test_an_overhang_leaves_active_cells_beneath_salt_and_a_vertical_wall_does_not():
    nz, dz = 40, 5.0                                                     # 200 m, 2000-2200
    L = layer(nz=nz, dz=dz)
    z_mid = TOP + 100.0
    for flare, overhang in ((-0.3, True), (0.0, False)):
        body = sl.salt_body((CX, CY), (R, R), z_ref=z_mid, flare=flare)
        _, _, _, act = _build_geometry([L], salt=body)
        under = act.any(axis=2) & (act[:, :, 0] == 0)                      # an active cell beneath an inactive top
        assert bool(under.any()) == overhang
    xc, yc = centres()
    rho = np.hypot(xc - CX, yc - CY)
    col = np.unravel_index(np.argmin(np.abs(rho - (R + 15.0))), rho.shape)    # 15 m outside the wall at mid-depth
    shallow = TOP + (np.arange(nz) + 0.5) * dz
    expect = ~(rho[col] < R + 0.3 * (z_mid - shallow))
    _, _, _, act = _build_geometry([L], salt=sl.salt_body((CX, CY), (R, R), z_ref=z_mid, flare=-0.3))
    assert (act[col] == expect).all() and not expect.all() and expect.any()  # salt on top, sediment beneath


@pytest.mark.parametrize("lateral,height", [(400.0, 100.0), (300.0, 300.0)])
def test_the_part_of_the_trap_beneath_an_overhang_is_as_wide_as_the_underside_reaches_across_the_reservoir(lateral, height):
    """The grid holds the reservoir interval only, so what lies beneath an overhang (active cells with salt cells above them
    in their column) is where the underside, dipping atan(H / L) from horizontal, cuts through the interval. A reservoir
    h0 thick whose top rises toward the wall at slope s (z = 2000 + s e, e from the neck line) meets an underside
    z = z_ref - (H / L) e at e_hi = (z_ref - 2000) / (s + H / L) and leaves it at e_lo = (z_ref - 2000 - h0) / (s + H / L): the
    band is h0 / (s + H / L) wide, and columns nearer than e_lo are dead. Hand-computed, h0 = 60 m, s = 0.2, neck at 2090 m:
    133 m wide at 14 degrees (400 / 100), 50 m at 45 (300 / 300). A steep underside is why the old overhang trap was a cell."""
    dx, nz, dz, s, y_c, z_ref = 5.0, 30, 2.0, 0.2, 20.0, 2090.0
    nx, ny = 6, 160
    L = layer(nz=nz, dz=dz, nx=nx, ny=ny, dx=dx)
    body = sl.salt_body((0.5 * nx * dx, y_c - 5000.0), (20000.0, 5000.0), z_ref=z_ref, overhang=(lateral, height))
    ramp = st.Structure(lambda x, y: s * (np.asarray(y, dtype=float) - y_c))
    Xc, Yc, zc, act = _build_geometry([L], structure=ramp, salt=body)
    salt = sl.salt_cells(body, Xc, Yc, zc)
    under = (act > 0) & np.logical_or.accumulate(salt, axis=2)
    e = (np.arange(ny) + 0.5) * dx - y_c
    cols = np.flatnonzero(under[0].any(axis=1))
    slope = s + height / lateral
    e_hi, e_lo = (z_ref - 2000.0) / slope, (z_ref - 2000.0 - nz * dz) / slope
    assert (cols.max() - cols.min() + 1) * dx == pytest.approx(nz * dz / slope, abs=3 * dx)
    assert e[cols.min()] == pytest.approx(e_lo, abs=3 * dx) and e[cols.max()] == pytest.approx(e_hi, abs=3 * dx)
    live = np.flatnonzero(act[0].any(axis=1))
    assert e[live.min()] == pytest.approx(e_lo, abs=3 * dx)                       # nearer columns are dead


def test_a_leaning_wall_is_inactive_where_its_depth_shifted_outline_says():
    nz, dz = 40, 5.0
    cot = 1.0 / math.tan(math.radians(60.0))
    body = sl.salt_body((CX, CY), (R, R), z_ref=TOP + 100.0, lean=(cot, 0.0))
    _, _, _, act = _build_geometry([layer(nz=nz, dz=dz)], salt=body)
    xc, yc = centres()
    for k in (0, 20, 39):
        z = TOP + (k + 0.5) * dz
        shift = cot * (z - TOP - 100.0)
        assert (act[:, :, k] == ~((xc - CX - shift) ** 2 + (yc - CY) ** 2 < R ** 2)).all()


def test_faults_and_erosion_still_apply_with_salt():
    L = layer(nz=8, dz=2.5)
    fold = st.dome(25.0, 500.0, center=(CX, CY))
    fault = Fault(center=(CX + 400.0, CY), strike=90.0, length=2000.0, throw=6.0, dip=60.0, name="F1")
    erode = st.Structure(lambda x, y: np.zeros_like(np.asarray(x, dtype=float)) + TOP - 15.0)
    shape = dict(structure=fold, faults=[fault], erode_above=erode)
    body = sl.salt_body((CX - 200.0, CY), (250.0, 250.0))
    _, _, zc, without = _build_geometry([L], **shape)
    _, _, _, with_salt = _build_geometry([L], salt=body, **shape)
    cells = sl.salt_cells(body, *_build_geometry([L], **shape)[:3])
    assert cells.any() and (without == 0).any()
    assert (with_salt == without * (~cells)).all()


def test_the_faults_file_lists_only_faces_of_active_cells_beside_salt(tmp_path):
    """A salt cell has no connection for the fault multiplier to act on, so FAULTS lists only cells that are live: of the
    faces a fault would have in this model, those with a salt cell on either side (about half in the review's models, 175 of
    331) are not written. Read back: every listed cell is active, the fault still has faces on both sides of the stock, and
    without salt the same fault lists exactly the faces the salt cells took away and the live ones."""
    L = layer(nz=6, dz=3.0)
    fault = Fault(center=(CX, CY), strike=90.0, length=2400.0, throw=8.0, dip=60.0, name="F1")
    body = sl.salt_body((CX, CY), (R, R))

    def rows(**kw):
        text = to_grdecl(L, tmp_path / "m.grdecl", faults=[fault], **kw).read_text()
        return [r.split() for r in text.split("\nFAULTS\n")[1].split("\n/\n")[0].splitlines()], read_grdecl(tmp_path / "m.grdecl")

    listed, blocks = rows(salt=body)
    act = prop_cube(blocks, "ACTNUM", NX, NY, 6)
    cells = lambda r: [(int(r[1]) - 1, int(r[3]) - 1, k - 1) for k in range(int(r[5]), int(r[6]) + 1)]
    assert listed and all(act[c] > 0 for r in listed for c in cells(r))
    bare, _ = rows()
    in_salt = [r for r in bare if r[7] in ("'X'", "'Y'") and any(act[c] == 0 for c in cells(r))]
    assert in_salt and len(bare) > len(listed)                                         # the dead faces were listed before
    assert {" ".join(r) for r in listed} <= {" ".join(r) for r in bare}                # nothing else changes
    js = {int(r[3]) for r in listed if r[7] == "'X'"}                                  # the trace runs along y through the stock
    assert min(js) < NY // 2 - 5 and max(js) > NY // 2 + 5 and not {NY // 2} & js      # faces remain on both sides, none inside


def test_the_grid_must_resolve_the_upturn():
    body = sl.salt_body((CX, CY), (R, R))
    sl.salt_upturn(body, 30.0, 300.0)                                    # folding zone 300 m: cells up to 150 m
    _build_geometry([layer(nx=20, ny=16, dx=150.0)], salt=body)           # exactly half the zone passes
    with pytest.raises(ValueError, match="150"):
        _build_geometry([layer(nx=15, ny=12, dx=200.0)], salt=body)
    with pytest.raises(ValueError, match="wider than half"):
        to_grdecl(layer(nx=15, ny=12, dx=200.0), "/nonexistent/x.grdecl", salt=body)


def test_a_zone_exactly_two_cells_wide_is_resolved_whatever_the_float_noise_in_the_cell_size():
    """The narrowest zone a grid may have is two cells: ``max_cell`` = dx exactly. A layer's cell size is ``x_len / nx``, which for
    a tenth of the sizes between 40 and 60 m is a unit in the last place above the ``dx`` that made ``x_len`` (7 of these 57
    raised on that alone, with 6 cells), so the comparison has a tolerance of 1e-9; a cell a thousandth wider is still refused."""
    for dx in np.linspace(40.0, 60.0, 57):
        body = sl.salt_body((CX, CY), (R, R))
        sl.salt_upturn(body, 30.0, 2.0 * dx)
        _build_geometry([layer(nx=6, ny=6, dx=float(dx))], salt=body)
    body = sl.salt_body((CX, CY), (R, R))
    sl.salt_upturn(body, 30.0, 100.0)
    with pytest.raises(ValueError, match="wider than half"):
        _build_geometry([layer(nx=6, ny=6, dx=50.05)], salt=body)


def test_plot_section_and_to_pyvista_take_salt():
    body = sl.salt_body((CX, CY), (R, R))
    L = layer()
    act = _build_geometry([L], salt=body)[3]
    fig, ax = plt.subplots()
    plot_section(L, "perm_mat", axis="y", index=NY // 2, ax=ax, salt=body)
    assert len(ax.collections[0].get_paths()) == act[:, NY // 2, :].sum() < NX * 4      # the salt cells are not drawn
    plt.close(fig)
    pytest.importorskip("pyvista")
    assert to_pyvista(L, salt=body).cell_data["ACTNUM"].sum() == act.sum()
