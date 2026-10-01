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


def tiny_fault():
    """One fault through a 10 x 4 x 8 stack of 2.5 m beds, clay-rich and clean in pairs, all of 100 mD: the faults
    ``faces`` and the arguments of ``face_multipliers`` that follow them, up to the seal."""
    layer = Layer(10, 4, 8, 250.0, 100.0, 20.0, top_depth=TOP, kzkx=0.1)
    layer.perm_mat = np.full((10, 4, 8), 100.0)
    vsh = np.where((np.arange(8) // 2) % 2 == 0, 0.1, 0.9)[None, None, :] * np.ones((10, 4, 8))     # K-down
    f = Fault(center=(130.0, 50.0), strike=90.0, length=20000.0, throw=3.0, dip=70.0, name="F1")
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=[f], _faces=faces)
    perm = np.asarray(layer.perm_mat)[:, :, ::-1]
    return faces, (zc, act, vsh, (perm, perm, perm), 25.0, 25.0)


def many_faults(seal, n):
    """The per-fault results of ``n`` copies of the tiny fault, each with its own draws."""
    faces, rest = tiny_fault()
    return face_multipliers(faces * n, *rest, seal)["faults"]


def test_the_spread_options_off_reproduce_the_physics_and_draw_nothing(monkeypatch):
    """With the defaults every face keeps its physics multiplier, as before the spread existed (the sum is that of the
    code at 9b23681), and a fixed offset shifts the faults without drawing a random number."""
    faces, rest = tiny_fault()
    base = face_multipliers(faces, *rest, Seal())["faults"][0]
    assert base["mode"] == "seal" and len(base["mult"]) == 28
    assert sum(base["mult"]) == pytest.approx(2.789068201551789, rel=1e-12)

    class Silent:
        def __getattr__(self, name):
            raise AssertionError(f"drew a random number ({name})")

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: Silent())
    off = face_multipliers(faces, *rest, Seal(offset=0.0, p_open=0.0, p_enhance=0.0, scatter=0.0))["faults"][0]
    assert off["mult"] == base["mult"]
    shifted = face_multipliers(faces, *rest, Seal(offset=-0.6))["faults"][0]["mult"]
    assert np.allclose(np.array(shifted) / np.array(base["mult"]), 10.0 ** -0.6, rtol=1e-12)


def test_each_fault_has_one_factor_with_the_asked_median_and_spread():
    """All faces of a fault share one factor 10^(offset + N(0, scatter)); over many faults its log10 has the asked
    median and spread (the faces stay below 1 here, so the cap does not touch the ratio)."""
    base = np.array(many_faults(Seal(), 1)[0]["mult"])
    logs = []
    for info in many_faults(Seal(offset=-0.8, scatter=0.3, seed=4), 300):
        ratio = np.log10(np.array(info["mult"]) / base)
        assert np.ptp(ratio) < 1e-9
        logs.append(ratio[0])
    assert np.median(logs) == pytest.approx(-0.8, abs=0.07)
    assert np.std(logs) == pytest.approx(0.3, abs=0.04)


def test_scatter_alone_draws_one_normal_per_fault_as_it_always_did():
    """With no open or enhancing share the stream is what it was: one normal per fault, in the order of the faults."""
    base = np.array(many_faults(Seal(), 1)[0]["mult"])
    rng = np.random.default_rng(9)
    for info in many_faults(Seal(scatter=0.3, seed=9), 5):
        assert np.array(info["mult"])[0] / base[0] == pytest.approx(10.0 ** rng.normal(0.0, 0.3), rel=1e-12)


def test_open_and_enhancing_faults_follow_their_shares():
    """A uniform draw per fault: below p_open the fault is open (every face 1), up to p_open + p_enhance it enhances
    flow (every face one log-uniform value of ``enhance``, above 1), else the faces keep their physics. Counts lie
    within four binomial standard deviations of the shares."""
    n, p_open, p_enhance = 500, 0.2, 0.3
    faults = many_faults(Seal(p_open=p_open, p_enhance=p_enhance, seed=1), n)
    modes = [f["mode"] for f in faults]
    for mode, p in (("open", p_open), ("enhancing", p_enhance), ("seal", 1.0 - p_open - p_enhance)):
        assert abs(modes.count(mode) - n * p) <= 4.0 * math.sqrt(n * p * (1.0 - p))
    boost = []
    for f in faults:
        mult = np.array(f["mult"])
        if f["mode"] == "open":
            assert np.all(mult == 1.0) and f["effective"] == pytest.approx(1.0, abs=1e-12)
        elif f["mode"] == "enhancing":
            assert np.ptp(mult) == 0.0 and 1.0 <= mult[0] <= 20.0
            boost.append(np.log10(mult[0]))
        else:
            assert np.all(mult < 1.0)
    assert max(boost) > 1.0 and min(boost) < 0.3                       # beyond the cap of 1, over the whole range
    assert np.mean(boost) == pytest.approx(0.5 * math.log10(20.0), abs=0.13)    # log-uniform: the middle of the range


def test_the_draws_per_fault_are_the_share_then_one_value():
    """The order is fixed, so a seed reproduces a tree: a uniform per fault, then (enhancing) one uniform in log10 of
    ``enhance`` or (seal, with a scatter) one normal."""
    seal = Seal(p_open=0.3, p_enhance=0.2, scatter=0.25, enhance=(2.0, 8.0), seed=5)
    base = np.array(many_faults(Seal(), 1)[0]["mult"])
    rng = np.random.default_rng(5)
    for info in many_faults(seal, 40):
        u = rng.random()
        mult = np.array(info["mult"])
        if u < 0.3:
            assert info["mode"] == "open" and np.all(mult == 1.0)
        elif u < 0.5:
            assert info["mode"] == "enhancing"
            assert mult[0] == pytest.approx(10.0 ** rng.uniform(math.log10(2.0), math.log10(8.0)), rel=1e-12)
        else:
            assert info["mode"] == "seal"
            assert mult[0] / base[0] == pytest.approx(min(10.0 ** rng.normal(0.0, 0.25), 1.0 / base[0]), rel=1e-12)


@pytest.mark.parametrize("kw", [dict(p_open=-0.1), dict(p_enhance=1.2), dict(p_open=0.7, p_enhance=0.5),
                                dict(enhance=(5.0, 2.0)), dict(enhance=(0.0, 4.0))])
def test_shares_and_ranges_that_make_no_sense_are_refused(kw):
    with pytest.raises(ValueError):
        Seal(**kw)


def test_to_grdecl_writes_the_face_multipliers(tmp_path):
    layer, vsh = cake()
    f = Fault(center=(510.0, 250.0), strike=90.0, length=20000.0, throw=12.0, dip=60.0, name="F1")
    to_grdecl(layer, tmp_path / "m.grdecl", faults=[f], seal=Seal(vsh=vsh))
    text = (tmp_path / "m.grdecl").read_text()
    assert "\nMULTX\n" in text and "\nMULTY\n" in text and "\nMULTZ\n" in text
    plain = tmp_path / "plain.grdecl"
    to_grdecl(layer, plain, faults=[f])
    assert "MULTX" not in plain.read_text()


def test_to_grdecl_writes_the_spread_of_the_faults(tmp_path):
    """The spread options reach the file: an enhancing fault's faces are written above 1, an open one's at 1."""
    layer, vsh = cake()
    f = Fault(center=(510.0, 250.0), strike=90.0, length=20000.0, throw=12.0, dip=60.0, name="F1")

    def multx(seal):
        to_grdecl(layer, tmp_path / "m.grdecl", faults=[f], seal=seal)
        block = (tmp_path / "m.grdecl").read_text().split("\nMULTX\n")[1].split("\n/")[0]
        return np.array([float(tok.split("*")[-1]) for tok in block.split()])

    assert multx(Seal(vsh=vsh)).min() < 1.0
    enhanced = multx(Seal(vsh=vsh, p_enhance=1.0, enhance=(5.0, 5.0), seed=1))
    assert enhanced.max() == pytest.approx(5.0) and enhanced.min() == 1.0
    assert np.all(multx(Seal(vsh=vsh, p_open=1.0, seed=1)) == 1.0)


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
