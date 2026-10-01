"""Fault seal per cell face (shale gouge ratio, Manzocchi transmissibility) and fault blocks."""
import math

import numpy as np
import pytest

from resmill import structure as st
from resmill.export import _build_geometry, to_grdecl
from resmill.fault_seal import Seal, fault_blocks, fault_rock_permeability, face_multipliers, knott_seal_probability
from resmill.faults import Fault, face_records
from resmill.layers.base import Layer

NX, NY, NZ, DX, DZ, TOP = 40, 20, 20, 25.0, 2.5, 2000.0


def cake(sand_vsh=0.1, shale_vsh=0.9, beds=2):
    """A flat 50 m stack of alternating 5 m sand and shale beds (ResMill's k runs bottom-up); returns the layer and
    its clay fraction per cell in K-down order."""
    layer = Layer(NX, NY, NZ, NX * DX, NY * DX, NZ * DZ, top_depth=TOP, kzkx=0.1)
    sand = (np.arange(NZ)[::-1] // beds) % 2 == 0                      # ResMill order, bottom-up
    layer.poro_mat = np.where(sand, 0.25, 0.05)[None, None, :] * np.ones((NX, NY, NZ))
    layer.perm_mat = np.where(sand, 100.0, 0.01)[None, None, :] * np.ones((NX, NY, NZ))
    vsh = np.where(sand[::-1], sand_vsh, shale_vsh)[None, None, :] * np.ones((NX, NY, NZ))   # K-down
    return layer, vsh


def test_fault_rock_permeability_follows_manzocchi():
    """log10 k_f = -4 SGR - 1/4 log10(D) (1 - SGR)^5, k_f in mD, D in m (Manzocchi et al. 1999)."""
    assert fault_rock_permeability(0.25, 10.0) == pytest.approx(10.0 ** (-1.0 - 0.25 * 0.75 ** 5), rel=1e-9)
    assert fault_rock_permeability(0.6, 50.0) < fault_rock_permeability(0.2, 50.0) < fault_rock_permeability(0.2, 5.0)


@pytest.mark.parametrize("throw,expected", [(20.0, 0.5), (2.0, 0.1)])
def test_sgr_is_the_clay_of_the_beds_that_slid_past(throw, expected):
    """A 20 m throw drags two sand and two shale beds past each face (SGR 0.5); a 2 m throw in the middle of a sand
    bed drags only sand past (SGR 0.1)."""
    layer, vsh = cake()
    f = Fault(center=(500.0, 250.0), strike=90.0, length=20000.0, throw=throw, dip=89.0, drag=(0.0, 0.0), name="F1")
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=[f], _faces=faces)
    perm = np.asarray(layer.perm_mat)[:, :, ::-1]
    out = face_multipliers(faces, zc, act, vsh, (perm, perm, perm), DX, DX, Seal())
    sgr = np.array([s for s in out["faults"][0]["sgr"] if np.isfinite(s)])
    if throw > 5.0:
        assert np.median(sgr) == pytest.approx(expected, abs=0.08)
    else:
        assert np.percentile(sgr, 25) == pytest.approx(expected, abs=0.05)


def test_the_face_multiplier_follows_manzocchis_transmissibility():
    """T = [1 + t_f (2/k_f - 1/k_i - 1/k_j) / (L_i/k_i + L_j/k_j)]^-1 with t_f = D/66: for SGR 0.5, 10 m of displacement
    between 100 mD cells 25 m wide, and no fault (multipliers 1) away from the fault."""
    layer, vsh = cake(sand_vsh=0.5, shale_vsh=0.5)
    layer.perm_mat[:] = 100.0
    f = Fault(center=(510.0, 250.0), strike=90.0, length=20000.0, throw=10.0, dip=90.0, drag=(0.0, 0.0), name="F1")
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=[f], _faces=faces)
    perm = np.asarray(layer.perm_mat)[:, :, ::-1]
    out = face_multipliers(faces, zc, act, vsh, (perm, perm, perm), DX, DX, Seal())
    kf = fault_rock_permeability(0.5, 10.0)
    expected = 1.0 / (1.0 + 10.0 / 66.0 * (2.0 / kf - 2.0 / 100.0) / (2.0 * 12.5 / 100.0))
    mx = out["MULTX"]
    listed = mx < 1.0
    assert listed.any() and np.median(mx[listed]) == pytest.approx(expected, rel=0.15)
    assert np.all(mx[:5] == 1.0) and np.all(mx[-5:] == 1.0)


def test_knott_seal_probability_rises_with_throw_over_thickness():
    """Knott 1993 (North Sea): faults throwing more than the reservoir's thickness seal over 90 % of the time; within
    the reservoir sealing rises with Dn."""
    p = [knott_seal_probability(dn) for dn in (0.1, 0.3, 0.7, 1.5)]
    assert p == sorted(p) and p[-1] >= 0.9 and 0.1 <= p[0] <= 0.35


def test_faces_lie_between_open_and_sealed_by_the_clay_that_slid_past():
    """Not a switch: between sands of one permeability each face's multiplier falls continuously as more clay slid past
    it, from about 0.3 (clean sand) to about 0.01 (half clay), where most of Norne's history-matched faults sit."""
    layer, vsh = cake(shale_vsh=0.5, beds=4)
    layer.perm_mat[:] = 100.0
    f = Fault(center=(510.0, 250.0), strike=90.0, length=20000.0, throw=6.0, dip=70.0, name="F1")
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=[f], _faces=faces)
    perm = np.asarray(layer.perm_mat)[:, :, ::-1]
    info = face_multipliers(faces, zc, act, vsh, (perm, perm, perm), DX, DX, Seal())["faults"][0]
    mult, sgr = np.array(info["mult"]), np.array(info["sgr"])
    assert 0.001 < mult.min() < 0.05 and 0.2 < mult.max() < 0.5
    assert np.unique(mult.round(3)).size > 10
    assert np.corrcoef(np.argsort(np.argsort(sgr)), np.argsort(np.argsort(mult)))[0, 1] < -0.9


def test_a_faults_effective_multiplier_follows_the_faces_that_carry_flow():
    """Faces between tight shales carry almost no flow, so the fault's one equivalent multiplier (a MULTFLT passing the
    same flow) follows its sand faces (0.08-0.3 here), not the plain mean of all faces, which the shale faces (near 1
    against the shale's own permeability) pull up."""
    layer, vsh = cake(beds=4)
    f = Fault(center=(510.0, 250.0), strike=90.0, length=20000.0, throw=6.0, dip=70.0, name="F1")
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=[f], _faces=faces)
    perm = np.asarray(layer.perm_mat)[:, :, ::-1]
    info = face_multipliers(faces, zc, act, vsh, (perm, perm, perm), DX, DX, Seal())["faults"][0]
    mult = np.array(info["mult"])
    assert 0.08 < info["effective"] < 0.3
    assert 10.0 ** np.mean(np.log10(mult)) > 2.0 * info["effective"]


def test_to_grdecl_writes_the_face_multipliers(tmp_path):
    layer, vsh = cake()
    f = Fault(center=(510.0, 250.0), strike=90.0, length=20000.0, throw=12.0, dip=60.0, name="F1")
    to_grdecl(layer, tmp_path / "m.grdecl", faults=[f], seal=Seal(vsh=vsh))
    text = (tmp_path / "m.grdecl").read_text()
    assert "\nMULTX\n" in text and "\nMULTY\n" in text and "\nMULTZ\n" in text
    plain = tmp_path / "plain.grdecl"
    to_grdecl(layer, plain, faults=[f])
    assert "MULTX" not in plain.read_text()


def dome_model(nx=80, ny=60, dx=50.0):
    layer = Layer(nx, ny, 10, nx * dx, ny * dx, 30.0, top_depth=TOP, kzkx=0.1)
    layer.poro_mat = np.full((nx, ny, 10), 0.2)
    layer.perm_mat = np.full((nx, ny, 10), 100.0)
    fold = st.closure(area=4e6, height=80.0, aspect=1.5, center=(0.5 * nx * dx, 0.5 * ny * dx))
    return layer, fold


def test_a_sealing_fault_through_the_crest_splits_the_trap_into_two_blocks():
    """A fault that seals, through the crest, makes two blocks with their own crests and spill points; an open one
    leaves one trap, the same as without the fault."""
    layer, fold = dome_model()
    f = Fault(center=(2000.0, 1500.0), strike=90.0, length=20000.0, throw=8.0, dip=89.0, drag=(0.0, 0.0), name="F1")
    for sealing, n in ((True, 2), (False, 1)):
        faces = []
        _, _, zc, act = _build_geometry([layer], structure=fold, faults=[f], _faces=faces)
        perm = np.full(act.shape, 100.0)
        vsh = np.full(act.shape, 0.9 if sealing else 0.0)
        mults = face_multipliers(faces, zc, act, vsh, (perm, perm, perm), 50.0, 50.0, Seal())
        blocks = fault_blocks(zc, act, faces, mults, 50.0, 50.0)
        traps = [b for b in blocks if b["area"] > 0.2e6]
        assert len(traps) == n
        for b in traps:
            assert b["height"] > 0.0 and b["spill_depth"] > b["crest_depth"]
