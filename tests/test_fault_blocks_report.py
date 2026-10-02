"""fault_blocks split into block_inputs and blocks_at, and to_grdecl's report (ResSimMill step 7, task 1)."""
import numpy as np
import pytest

from resmill import GaussianLayer, Reservoir, structure as st
from resmill.export import _build_geometry, horizontal_permeability, to_grdecl
from resmill.fault_patterns import fold_faults
from resmill.fault_seal import (Capillary, Seal, block_inputs, block_labels, blocks_at, face_multipliers,
                                fault_blocks)

NX, NY, NZ, DX = 40, 30, 6, 100.0


def _model():
    np.random.seed(11)                                  # GaussianLayer draws from numpy's global state
    layer = GaussianLayer(nx=NX, ny=NY, nz=NZ, x_len=NX * DX, y_len=NY * DX, z_len=NZ * 4.0, top_depth=2000.0)
    layer.create_geology(poro_ave=0.25, perm_ave=2.5, poro_std=0.02, perm_std=0.3, ntg=0.6)
    fold = st.closure(area=3e6, height=90.0, aspect=1.8, azimuth=20.0, center=(2000.0, 1500.0), warp=0.2, seed=3)
    faults = fold_faults("faulted_anticline", fold, NX * DX, NY * DX, DX, 1.5, 2000.0, NZ * 4.0, seed=5)
    return layer, fold, faults


def _inputs(layer, fold, faults):
    vsh, faces = np.full((NX, NY, NZ), 0.3), []
    _, _, zc, act = _build_geometry([layer], fold, None, None, None, None, None, False, faults, faces)
    permx, permy = horizontal_permeability(Reservoir([layer]))
    permz = np.asarray(layer.perm_mat, dtype=float)[:, :, ::-1] * layer.kzkx
    mults = face_multipliers(faces, zc, act, vsh, (permx, permy, permz), DX, DX, Seal(vsh=vsh, seed=7))
    return zc, act, faces, mults


@pytest.mark.parametrize("delta_rho", [100.0, 300.0, 700.0])
def test_blocks_at_is_fault_blocks_for_any_capillary(delta_rho):
    layer, fold, faults = _model()
    assert faults, "the fixture needs faults"
    zc, act, faces, mults = _inputs(layer, fold, faults)
    cap = Capillary(delta_rho=delta_rho)
    old = fault_blocks(zc, act, faces, mults, DX, DX, cap)
    new = blocks_at(block_inputs(zc, act, faces, mults), DX, DX, cap)
    assert len(old) == len(new) and old
    for a, b in zip(old, new):
        assert {k: v for k, v in a.items() if k != "mask"} == {k: v for k, v in b.items() if k != "mask"}
        assert (a["mask"] == b["mask"]).all()


def test_labels_separate_fault_blocks_and_mark_dead_columns():
    layer, fold, faults = _model()
    zc, act, faces, mults = _inputs(layer, fold, faults)
    inputs = block_inputs(zc, act, faces, mults)
    labels = block_labels(inputs)
    assert labels.shape == (NX, NY) and labels.max() >= 1
    assert (labels[~inputs["alive"]] == -1).all() and (labels[inputs["alive"]] >= 0).all()
    for walls, (lo, hi) in zip(inputs["split"], ((np.s_[:-1], np.s_[1:]), (np.s_[:, :-1], np.s_[:, 1:]))):
        free = ~walls & inputs["alive"][lo] & inputs["alive"][hi]
        assert (labels[lo][free] == labels[hi][free]).all()          # an edge no fault splits joins its columns


def test_to_grdecl_report_fills_the_inputs_and_changes_no_byte(tmp_path):
    layer, fold, faults = _model()
    seal = Seal(vsh=np.full((NX, NY, NZ), 0.3), seed=7)
    plain, report = tmp_path / "a.grdecl", {}
    to_grdecl(layer, plain, structure=fold, faults=faults, seal=seal)
    to_grdecl(layer, tmp_path / "b.grdecl", structure=fold, faults=faults, seal=seal, report=report)
    assert plain.read_bytes().split(b"\n", 1)[1] == (tmp_path / "b.grdecl").read_bytes().split(b"\n", 1)[1]
    assert set(report) == {"fault_names", "faults", "block_inputs"}
    assert report["fault_names"] == [fault.name or f"F{n + 1:02d}" for n, fault in enumerate(faults)]
    assert {"name", "mode", "effective"} <= set(report["faults"][0])
    assert report["block_inputs"]["depth"].shape == (NX, NY)


def test_a_model_without_faults_still_reports_one_block(tmp_path):
    layer, fold, _ = _model()
    report = {}
    to_grdecl(layer, tmp_path / "c.grdecl", structure=fold, report=report)
    assert report["faults"] == [] and report["fault_names"] == []
    blocks = blocks_at(report["block_inputs"], DX, DX, Capillary(delta_rho=300.0))
    assert len(blocks) == 1 and blocks[0]["limited_by"] == "spill"
    assert (block_labels(report["block_inputs"]) == 0).all()


def test_every_trap_lies_in_one_labelled_block():
    layer, fold, faults = _model()
    zc, act, faces, mults = _inputs(layer, fold, faults)
    inputs = block_inputs(zc, act, faces, mults)
    labels = block_labels(inputs)
    blocks = blocks_at(inputs, DX, DX, Capillary(delta_rho=300.0))
    assert blocks
    for block in blocks:
        assert len(np.unique(labels[block["mask"]])) == 1 == len(np.unique(labels[block["crest"]]))


def test_without_a_seal_a_fault_that_splits_the_tops_is_a_wall(tmp_path):
    layer, fold, faults = _model()
    report = {}
    to_grdecl(layer, tmp_path / "d.grdecl", structure=fold, faults=faults, report=report)
    assert report["faults"] == [] and len(report["fault_names"]) == len(faults)
    inputs = report["block_inputs"]
    assert len(inputs["face_records"]) == 0 and any(walls.any() for walls in inputs["split"])
    blocks = blocks_at(inputs, DX, DX, Capillary(delta_rho=300.0))
    assert blocks and all(block["limited_by"] != "leak" for block in blocks)         # no face has a level to leak at
