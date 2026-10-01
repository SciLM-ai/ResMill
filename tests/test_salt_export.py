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
from tests.test_export import prop_cube, read_grdecl

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


def test_the_grid_must_resolve_the_upturn():
    body = sl.salt_body((CX, CY), (R, R))
    sl.salt_upturn(body, 30.0, 300.0)                                    # folding zone 300 m: cells up to 150 m
    _build_geometry([layer(nx=20, ny=16, dx=150.0)], salt=body)           # exactly half the zone passes
    with pytest.raises(ValueError, match="150"):
        _build_geometry([layer(nx=15, ny=12, dx=200.0)], salt=body)
    with pytest.raises(ValueError, match="wider than half"):
        to_grdecl(layer(nx=15, ny=12, dx=200.0), "/nonexistent/x.grdecl", salt=body)


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
