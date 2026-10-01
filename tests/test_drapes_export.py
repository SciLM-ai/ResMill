"""The export of mud drapes: MULTX, MULTY and MULTZ in Eclipse order, times a fault seal's, none without drapes."""
import numpy as np
import pytest

from resmill.export import to_grdecl
from resmill.fault_seal import Seal
from resmill.faults import Fault
from resmill.layers.base import Layer


def _multiplier(path, keyword, shape):
    """One MULT array of a written GRDECL as an (nx, ny, nz) cube, K down (n*value runs expanded). Read by keyword:
    the FAULTS records of a faulted file would confuse a section parser."""
    words = path.read_text().split()
    start = words.index(keyword) + 1
    values = []
    for word in words[start:words.index("/", start)]:
        count, _, value = word.rpartition("*")
        values.extend([float(value)] * (int(count) if count else 1))
    return np.array(values).reshape(shape, order="F")


def _plain_layer(nx=4, ny=3, nz=5, top=1000.0):
    layer = Layer(nx, ny, nz, nx * 10.0, ny * 10.0, nz * 2.0, top_depth=top, kzkx=0.1)
    layer.poro_mat = np.full((nx, ny, nz), 0.2)
    layer.perm_mat = np.full((nx, ny, nz), 100.0)
    return layer


def test_export_writes_each_multiplier_on_the_face_it_belongs_to(tmp_path):
    """MULTX, MULTY and MULTZ are a cell's + faces in Eclipse order (K down): the layer's lower face (k up) of cell k is
    the + z face of Eclipse cell K = nz - 1 - k, i.e. the face towards K + 1. Written arrays are read back."""
    layer = _plain_layer()
    nx, ny, nz = layer.poro_mat.shape
    layer.mult_x, layer.mult_y, layer.mult_z = (np.ones((nx, ny, nz), dtype=np.float32) for _ in range(3))
    layer.mult_z[1, 2, 3] = 0.25            # face between k = 3 and k = 2, i.e. Eclipse K = 1 (0-based) and K = 2
    layer.mult_x[2, 0, 4] = 0.5             # +x face of cell (2, 0), top layer: Eclipse K = 0
    layer.mult_y[0, 1, 0] = 0.125           # +y face of cell (0, 1), layer base: Eclipse K = 4
    to_grdecl(layer, tmp_path / "m.grdecl")
    mz, mx, my = (_multiplier(tmp_path / "m.grdecl", key, (nx, ny, nz)) for key in ("MULTZ", "MULTX", "MULTY"))
    assert mz[1, 2, 1] == 0.25 and (mz != 1.0).sum() == 1
    assert mx[2, 0, 0] == 0.5 and (mx != 1.0).sum() == 1
    assert my[0, 1, 4] == 0.125 and (my != 1.0).sum() == 1


def test_export_without_drapes_writes_no_multipliers(tmp_path):
    """A layer without drapes, or with drapes that cover nothing, writes the file it always did."""
    layer = _plain_layer()
    to_grdecl(layer, tmp_path / "plain.grdecl")
    layer.mult_x = layer.mult_y = layer.mult_z = np.ones(layer.poro_mat.shape, dtype=np.float32)
    to_grdecl(layer, tmp_path / "ones.grdecl")
    assert "MULTZ" not in (tmp_path / "ones.grdecl").read_text()
    strip = lambda p: (tmp_path / p).read_bytes().split(b"\n", 1)[1]
    assert strip("plain.grdecl") == strip("ones.grdecl")


def test_export_multiplies_drapes_and_fault_seal_face_by_face(tmp_path):
    """With faults and a seal, a face's written multiplier is the fault seal's times the drape's."""
    nx, ny, nz = 40, 20, 20
    layer = Layer(nx, ny, nz, nx * 25.0, ny * 25.0, nz * 2.5, top_depth=2000.0, kzkx=0.1)
    sand = (np.arange(nz)[::-1] // 2) % 2 == 0
    layer.poro_mat = np.where(sand, 0.25, 0.05)[None, None, :] * np.ones((nx, ny, nz))
    layer.perm_mat = np.where(sand, 100.0, 0.01)[None, None, :] * np.ones((nx, ny, nz))
    vsh = np.where(sand[::-1], 0.1, 0.9)[None, None, :] * np.ones((nx, ny, nz))
    fault = Fault(center=(500.0, 250.0), strike=90.0, length=20000.0, throw=10.0, dip=60.0, drag=(0.0, 0.0), name="F1")
    to_grdecl(layer, tmp_path / "seal.grdecl", faults=[fault], seal=Seal(vsh=vsh))
    rng = np.random.default_rng(1)
    layer.mult_x, layer.mult_y, layer.mult_z = (np.where(rng.random((nx, ny, nz)) < 0.2, 1e-3, 1.0).astype(np.float32)
                                                for _ in range(3))
    to_grdecl(layer, tmp_path / "both.grdecl", faults=[fault], seal=Seal(vsh=vsh))
    shape = (nx, ny, nz)
    for key, mult in (("MULTX", layer.mult_x), ("MULTY", layer.mult_y), ("MULTZ", layer.mult_z)):
        sealed = _multiplier(tmp_path / "seal.grdecl", key, shape)
        expected = sealed * np.asarray(mult, dtype=float)[:, :, ::-1]
        assert np.allclose(_multiplier(tmp_path / "both.grdecl", key, shape), expected, rtol=2e-5, atol=0.0)
        assert (expected != sealed).any()
    to_grdecl(layer, tmp_path / "faults_only.grdecl", faults=[fault])          # no seal: the drapes alone
    assert np.allclose(_multiplier(tmp_path / "faults_only.grdecl", "MULTZ", shape),
                       np.asarray(layer.mult_z, dtype=float)[:, :, ::-1], rtol=2e-5)


def test_export_stacks_the_multipliers_of_layers_and_one_without(tmp_path):
    """In a stack the layers' multipliers sit at their own K ranges (listed top to bottom); a layer without drapes
    counts as 1."""
    from resmill.reservoir import Reservoir
    top, bottom = _plain_layer(nz=3), _plain_layer(nz=4, top=1006.0)
    bottom.mult_x = bottom.mult_y = None
    bottom.mult_z = np.ones((4, 3, 4), dtype=np.float32)
    bottom.mult_z[0, 0, 2] = 0.1            # bottom layer: face between k = 2 and 1; Eclipse K = 3 + (4 - 1 - 2) = 4
    to_grdecl(Reservoir([top, bottom]), tmp_path / "s.grdecl")
    mz = _multiplier(tmp_path / "s.grdecl", "MULTZ", (4, 3, 7))
    assert mz[0, 0, 4] == pytest.approx(0.1) and (mz != 1.0).sum() == 1
