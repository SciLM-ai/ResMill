import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from resmill import structure as st
from resmill.export import to_grdecl
from resmill.layers.base import Layer
from resmill.layers.gaussian import GaussianLayer
from resmill.plotting import plot_section
from resmill.reservoir import Reservoir


# ---------------------------------------------------------------------------
# Minimal dependency-free GRDECL reader used to verify what was written.
# ---------------------------------------------------------------------------

def _num(tok):
    try:
        return float(tok)
    except ValueError:
        return tok


def read_grdecl(path):
    text = re.sub(r"--[^\n]*", "", Path(path).read_text())
    blocks, kw, vals = {}, None, None
    for tok in text.split():
        if kw is None:
            kw, vals = tok, []
        elif tok == "/":
            blocks[kw], kw = vals, None
        elif "*" in tok:
            n, v = tok.split("*")
            vals.extend([_num(v)] * int(n))
        else:
            vals.append(_num(tok))
    return blocks


def zcorn_cube(blocks, nx, ny, nz):
    return np.asarray(blocks["ZCORN"], dtype=float).reshape(
        (2 * nx, 2 * ny, 2 * nz), order="F")


def prop_cube(blocks, kw, nx, ny, nz):
    return np.asarray(blocks[kw], dtype=float).reshape((nx, ny, nz), order="F")


NX, NY, NZ, DX, DZ, TOP = 6, 5, 4, 10.0, 2.0, 1000.0


def make_layer(nx=NX, ny=NY, nz=NZ, dx=DX, dz=DZ, top=TOP, dip=0.0,
               poro=0.2, perm=100.0, kzkx=0.1):
    layer = Layer(nx, ny, nz, nx * dx, ny * dx, nz * dz, top_depth=top,
                  dip=dip, kzkx=kzkx)
    layer.poro_mat = np.full((nx, ny, nz), poro)
    layer.perm_mat = np.full((nx, ny, nz), perm)
    return layer


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_flat_grid_blocks_and_geometry(tmp_path):
    p = make_layer().to_grdecl(tmp_path / "flat.grdecl")
    blocks = read_grdecl(p)
    assert blocks["SPECGRID"][:4] == [NX, NY, NZ, 1] and blocks["SPECGRID"][4] == "F"
    assert blocks["PINCH"] == []
    assert blocks["MAPUNITS"] == ["METRES"] and blocks["GRIDUNIT"] == ["METRES"]
    assert len(blocks["COORD"]) == 6 * (NX + 1) * (NY + 1)
    assert len(blocks["ZCORN"]) == 8 * NX * NY * NZ

    zc = zcorn_cube(blocks, NX, NY, NZ)
    assert np.allclose(zc[:, :, 0], TOP, atol=0.006)
    thick = zc[:, :, 1::2] - zc[:, :, 0::2]
    assert np.allclose(thick, DZ, atol=0.011)
    assert np.all(np.diff(zc, axis=2) >= -1e-9)

    coord = np.asarray(blocks["COORD"], dtype=float).reshape(-1, 6)
    for j in range(NY + 1):
        for i in range(NX + 1):
            x, y, zt, x2, y2, zb = coord[j * (NX + 1) + i]
            assert (x, x2) == (i * DX, i * DX)
            assert (y, y2) == (j * DX, j * DX)
            assert zt < TOP and zb > TOP + NZ * DZ


def test_actnum_is_run_length_encoded(tmp_path):
    p = make_layer().to_grdecl(tmp_path / "rle.grdecl")
    assert f"{NX * NY * NZ}*1" in p.read_text()


def test_dip_reaches_the_grid(tmp_path):
    layer = make_layer(dip=10.0)
    blocks = read_grdecl(layer.to_grdecl(tmp_path / "dip.grdecl"))
    zc = zcorn_cube(blocks, NX, NY, NZ)
    for dj, j_node in [(0, 0), (1, 1)]:
        expected = TOP + j_node * DX * np.tan(np.radians(10.0))
        assert np.allclose(zc[:, dj, 0], expected, atol=0.006)


def test_anticline_matches_analytic_and_shares_corners(tmp_path):
    fold = st.anticline(amplitude=30, wavelength=600, azimuth=0, center=(30, 25))
    blocks = read_grdecl(make_layer().to_grdecl(tmp_path / "fold.grdecl",
                                                structure=fold))
    zc = zcorn_cube(blocks, NX, NY, NZ)
    for i_node, j_node in [(0, 0), (3, 2), (6, 5)]:
        expected = TOP + fold(i_node * DX, j_node * DX)
        di = 2 * i_node - 1 if i_node else 0
        dj = 2 * j_node - 1 if j_node else 0
        assert zc[di, dj, 0] == pytest.approx(expected, abs=0.006)
    # Smooth structure: corners shared between neighbor cells are identical.
    assert np.array_equal(zc[1:-1:2, :, :], zc[2::2, :, :])
    assert np.array_equal(zc[:, 1:-1:2, :], zc[:, 2::2, :])


def test_fault_throw_on_grid_line(tmp_path):
    flt = st.fault(throw=15, x0=3 * DX)
    blocks = read_grdecl(make_layer().to_grdecl(tmp_path / "fault.grdecl",
                                                structure=flt))
    zc = zcorn_cube(blocks, NX, NY, NZ)
    top = zc[:, :, 0]
    assert np.allclose(top[:6], TOP, atol=0.006)          # cells 0-2 unmoved
    assert np.allclose(top[6:], TOP + 15, atol=0.006)     # cells 3-5 dropped
    jumps = np.abs(top[2::2] - top[1:-1:2])
    assert np.allclose(jumps[2], 15, atol=0.012)          # only at the trace
    mask = np.ones(len(jumps), dtype=bool)
    mask[2] = False
    assert np.allclose(jumps[mask], 0, atol=1e-9)


def test_proportional_squeeze_between_surfaces(tmp_path):
    blocks = read_grdecl(make_layer().to_grdecl(tmp_path / "squeeze.grdecl",
                                                top=900.0, base=1100.0))
    zc = zcorn_cube(blocks, NX, NY, NZ)
    assert np.allclose(zc[:, :, 0], 900, atol=0.006)
    assert np.allclose(zc[:, :, -1], 1100, atol=0.006)
    thick = zc[:, :, 1::2] - zc[:, :, 0::2]
    assert np.allclose(thick, 200.0 / NZ, atol=0.011)


def test_drape_onto_top_preserves_thickness(tmp_path):
    blocks = read_grdecl(make_layer().to_grdecl(tmp_path / "drape.grdecl",
                                                top=900.0))
    zc = zcorn_cube(blocks, NX, NY, NZ)
    assert np.allclose(zc[:, :, 0], 900, atol=0.006)
    assert np.allclose(zc[:, :, 1::2] - zc[:, :, 0::2], DZ, atol=0.011)


def test_erosion_clips_and_deactivates(tmp_path):
    blocks = read_grdecl(make_layer().to_grdecl(tmp_path / "erode.grdecl",
                                                erode_above=1003.0))
    zc = zcorn_cube(blocks, NX, NY, NZ)
    actnum = prop_cube(blocks, "ACTNUM", NX, NY, NZ)
    # Interfaces 1000, 1002 clip to 1003; cell k=0 collapses, k=1 is cut.
    assert np.allclose(zc[:, :, 0], 1003, atol=0.006)
    assert np.allclose(zc[:, :, 1], 1003, atol=0.006)
    assert np.allclose(zc[:, :, 3] - zc[:, :, 2], 1.0, atol=0.011)
    assert np.all(actnum[:, :, 0] == 0)
    assert np.all(actnum[:, :, 1:] == 1)


def test_per_layer_structure_builds_unconformity(tmp_path):
    upper = make_layer(nz=2, dz=2.0, top=1000.0)               # 1000-1004
    lower = make_layer(nz=3, dz=3.0, top=1004.0)               # 1004-1013
    lift = st.dome(amplitude=10, radius=25, center=(30, 25))
    res = Reservoir([upper, lower])
    blocks = read_grdecl(to_grdecl(res, tmp_path / "unconf.grdecl",
                                   structure=[None, lift]))
    zc = zcorn_cube(blocks, NX, NY, 5)
    actnum = prop_cube(blocks, "ACTNUM", NX, NY, 5)
    # Contact (base of the unshifted upper layer) stays at 1004.
    assert np.allclose(zc[:, :, 2 * 2 - 1], 1004, atol=0.006)
    # First interface inside the lower layer: lifted, then truncated at
    # the younger base wherever the lift pushed it above 1004.
    node_x = np.repeat(np.arange(NX + 1) * DX, 2)[1:-1]
    node_y = np.repeat(np.arange(NY + 1) * DX, 2)[1:-1]
    Xn, Yn = np.meshgrid(node_x, node_y, indexing="ij")
    expected = np.maximum(1004.0, 1007.0 + lift(Xn, Yn))
    assert np.allclose(zc[:, :, 5], expected, atol=0.006)   # bottom face of cell k=2
    # Crest cell of the lower layer's top cell is eroded away; far corner survives.
    assert actnum[NX // 2, NY // 2, 2] == 0
    assert actnum[0, 0, 2] == 1
    assert np.all(actnum[:, :, :2] == 1)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_property_order_kzkx_and_variable_dz(tmp_path):
    upper = make_layer(nz=2, dz=2.0, top=1000.0, perm=100.0, kzkx=0.5)
    upper.poro_mat = np.full((NX, NY, 2), 0.11)
    upper.poro_mat[:, :, 1] = 0.12          # k-up: index 1 is the layer top
    lower = make_layer(nz=3, dz=3.0, top=1004.0, poro=0.2, perm=200.0, kzkx=0.1)
    res = Reservoir([upper, lower])
    blocks = read_grdecl(to_grdecl(res, tmp_path / "props.grdecl"))

    poro = prop_cube(blocks, "PORO", NX, NY, 5)
    assert np.allclose(poro[:, :, 0], 0.12)  # Eclipse K=1 is the very top cell
    assert np.allclose(poro[:, :, 1], 0.11)
    assert np.allclose(poro[:, :, 2:], 0.2)

    permx = prop_cube(blocks, "PERMX", NX, NY, 5)
    permz = prop_cube(blocks, "PERMZ", NX, NY, 5)
    assert np.array_equal(np.asarray(blocks["PERMY"]), np.asarray(blocks["PERMX"]))
    assert np.allclose(permz[:, :, :2] / permx[:, :, :2], 0.5)
    assert np.allclose(permz[:, :, 2:] / permx[:, :, 2:], 0.1)

    zc = zcorn_cube(blocks, NX, NY, 5)
    cell_thick = (zc[:, :, 1::2] - zc[:, :, 0::2])[0, 0]
    assert np.allclose(cell_thick, [2, 2, 3, 3, 3], atol=0.011)


def test_floors_apply(tmp_path):
    layer = make_layer(poro=0.0, perm=0.0)
    blocks = read_grdecl(layer.to_grdecl(tmp_path / "floors.grdecl",
                                         poro_floor=0.01, perm_floor=0.001))
    assert min(blocks["PORO"]) == pytest.approx(0.01)
    assert min(blocks["PERMX"]) == pytest.approx(0.001)
    assert min(blocks["PERMZ"]) == pytest.approx(0.001)


def test_facies_keyword(tmp_path):
    layer = GaussianLayer(NX, NY, NZ, NX * DX, NY * DX, NZ * DZ, top_depth=TOP)
    layer.create_geology(poro_ave=0.2, perm_ave=1.5, poro_std=0.03,
                         perm_std=0.5, ntg=0.7)
    blocks = read_grdecl(layer.to_grdecl(tmp_path / "facies.grdecl", facies=True))
    fac = np.asarray(blocks["FACIES"])
    assert fac.size == NX * NY * NZ
    assert set(np.unique(fac)) <= {-1.0, 3.0}


# ---------------------------------------------------------------------------
# Validation and plotting
# ---------------------------------------------------------------------------

def test_errors(tmp_path):
    with pytest.raises(ValueError, match="below top"):
        make_layer().to_grdecl(tmp_path / "x.grdecl", top=1100.0, base=900.0)
    with pytest.raises(ValueError, match="non-finite"):
        make_layer().to_grdecl(tmp_path / "x.grdecl",
                               structure=lambda x, y: np.nan * x)
    with pytest.raises(ValueError, match="create_geology"):
        Layer(NX, NY, NZ, 60, 50, 8, top_depth=TOP).to_grdecl(tmp_path / "x.grdecl")
    with pytest.raises(ValueError, match="facies"):
        make_layer().to_grdecl(tmp_path / "x.grdecl", facies=True)
    with pytest.raises(ValueError, match="one entry per layer"):
        make_layer().to_grdecl(tmp_path / "x.grdecl", structure=[None, None])


def test_plot_section_smoke():
    plt.close('all')
    plot_section(make_layer(), structure=st.dome(20, 30, center=(30, 25)),
                 erode_above=995.0, title="smoke")
    assert len(plt.get_fignums()) == 1
    plt.close('all')


# ---------------------------------------------------------------------------
# Optional independent round-trip via xtgeo
# ---------------------------------------------------------------------------

def test_xtgeo_roundtrip(tmp_path):
    xtgeo = pytest.importorskip("xtgeo")
    fold = st.anticline(amplitude=30, wavelength=600, azimuth=0, center=(30, 25))
    p = make_layer().to_grdecl(tmp_path / "xt.grdecl", structure=fold)
    grid = xtgeo.grid_from_file(str(p), fformat="grdecl")
    assert grid.dimensions == (NX, NY, NZ)
    dz = grid.get_dz().values
    assert np.allclose(dz[~dz.mask] if np.ma.isMaskedArray(dz) else dz,
                       DZ, atol=0.02)
