"""The simulation outline: structure.outline and to_grdecl(outline=) (ResSimMill step 7, G6)."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from resmill.export import to_grdecl
from resmill.fault_seal import Seal
from resmill.fold_traps import fold_trap
from resmill.structure import outline
from tests.test_export import prop_cube, read_grdecl
from tests.test_fold_traps import DX, NX, NY, NZ, THICKNESS, TOP, X_LEN, Y_LEN, args, layer


def actnum_of(path):
    return prop_cube(read_grdecl(path), "ACTNUM", NX, NY, NZ).astype(int)


def without_actnum(path):
    """The file's text without its ACTNUM block."""
    text = path.read_text()
    start = text.index("\nACTNUM\n")
    return text[:start] + text[text.index("\n/\n", start) + 3:]


def test_the_outline_is_the_trap_dilated_by_the_rim_from_centre_to_centre():
    nx, ny, dx, dy = 60, 40, 100.0, 50.0
    x, y = np.meshgrid((np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy, indexing="ij")
    trap = ((x - 3000.0) / 900.0) ** 2 + ((y - 1000.0) / 400.0) ** 2 <= 1.0
    tree = cKDTree(np.column_stack([x[trap], y[trap]]))
    near = tree.query(np.column_stack([x.ravel(), y.ravel()]))[0].reshape(nx, ny)           # an independent distance
    for rim in (0.0, 150.0, 700.0, 2500.0):
        assert (outline(trap, dx, dy, rim) == (near <= rim)).all()
    assert (outline(trap, dx, dy, 0.0) == trap).all() and outline(trap, dx, dy, 1e6).all()
    with pytest.raises(ValueError, match="no column"):
        outline(np.zeros((nx, ny), dtype=bool), dx, dy, 100.0)
    with pytest.raises(ValueError, match="rim"):
        outline(trap, dx, dy, -1.0)


def footprint():
    """An ellipse of columns about the middle of the map, a closure's stand-in."""
    x, y = np.meshgrid((np.arange(NX) + 0.5) * DX, (np.arange(NY) + 0.5) * DX, indexing="ij")
    return ((x - 0.5 * X_LEN) / 1200.0) ** 2 + ((y - 0.5 * Y_LEN) / 700.0) ** 2 <= 1.0


def test_to_grdecl_cuts_the_actnum_block_and_nothing_else(tmp_path):
    """The cut file is the whole map's with its ACTNUM replaced: ZCORN, the seal's multipliers, FAULTS and the report
    are those of the whole map."""
    built = fold_trap("faulted_anticline", X_LEN, Y_LEN, DX, TOP, THICKNESS, **args("faulted_anticline"))
    keep = outline(footprint(), DX, DX, 1000.0)
    assert built["kwargs"]["faults"] and 0 < keep.sum() < keep.size
    seal = Seal(vsh=np.full((NX, NY, NZ), 0.3), seed=1)
    plain, cut, whole, kept = tmp_path / "a.grdecl", tmp_path / "b.grdecl", {}, {}
    to_grdecl(layer(), plain, seal=seal, report=whole, **built["kwargs"])
    to_grdecl(layer(), cut, seal=seal, report=kept, outline=keep, **built["kwargs"])
    assert without_actnum(cut) == without_actnum(plain) and cut.read_bytes() != plain.read_bytes()
    assert (actnum_of(cut) == actnum_of(plain) * keep[:, :, None]).all()     # a cell a fault collapsed stays inactive
    assert kept["fault_names"] == whole["fault_names"] and kept["faults"] == whole["faults"]
    assert np.array_equal(kept["block_inputs"]["depth"], whole["block_inputs"]["depth"], equal_nan=True)
    for key in ("alive", "face_records"):
        assert np.array_equal(kept["block_inputs"][key], whole["block_inputs"][key])
    assert all(np.array_equal(a, b) for a, b in zip(kept["block_inputs"]["split"], whole["block_inputs"]["split"]))


def test_an_outline_that_keeps_every_column_or_none_is_what_it_says(tmp_path):
    built = fold_trap("four_way", X_LEN, Y_LEN, DX, TOP, THICKNESS, **args("four_way"))
    plain, everything, none = tmp_path / "a.grdecl", tmp_path / "b.grdecl", tmp_path / "c.grdecl"
    to_grdecl(layer(), plain, **built["kwargs"])
    to_grdecl(layer(), everything, outline=np.ones((NX, NY), dtype=bool), **built["kwargs"])
    to_grdecl(layer(), none, outline=None, **built["kwargs"])
    assert plain.read_bytes() == everything.read_bytes() == none.read_bytes()
    for bad, message in ((np.ones((NX, NY + 1), dtype=bool), "shape"), (np.zeros((NX, NY), dtype=bool), "keeps no")):
        with pytest.raises(ValueError, match=message):
            to_grdecl(layer(), tmp_path / "d.grdecl", outline=bad, **built["kwargs"])
    assert not (tmp_path / "d.grdecl").exists()                                     # refused before a file is written
