"""Fault seal per cell face (shale gouge ratio, Manzocchi transmissibility), its spread over faults, and fault blocks
by capillary seal capacity."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from resmill import structure as st
from resmill.export import _build_geometry, to_grdecl
from resmill.fault_seal import (Capillary, Seal, bretan_pressure, fault_blocks, fault_rock_permeability,
                                face_multipliers, knott_seal_probability, seal_capacity)
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
    """On a fault of the real geometry: for SGR 0.5, 10 m of displacement between 100 mD cells 25 m wide the multiplier
    is 0.01595 (k_f 0.00982 mD, t_f 0.1515 m, resistance 15.68 against 0.25: the hand calculation of
    ``test_fault_seal_references``), and 1 (no fault) away from the fault."""
    layer, vsh = cake(sand_vsh=0.5, shale_vsh=0.5)
    layer.perm_mat[:] = 100.0
    f = Fault(center=(510.0, 250.0), strike=90.0, length=20000.0, throw=10.0, dip=90.0, drag=(0.0, 0.0), name="F1")
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=[f], _faces=faces)
    perm = np.asarray(layer.perm_mat)[:, :, ::-1]
    out = face_multipliers(faces, zc, act, vsh, (perm, perm, perm), DX, DX, Seal())
    mx = out["MULTX"]
    listed = mx < 1.0
    assert listed.any() and np.median(mx[listed]) == pytest.approx(0.01595, rel=0.01)
    assert np.all(mx[:5] == 1.0) and np.all(mx[-5:] == 1.0)


def test_knott_seal_probability_rises_with_throw_over_thickness():
    """Knott 1993 (North Sea): faults throwing more than the reservoir's thickness seal over 90 % of the time; within
    the reservoir sealing rises with Dn."""
    p = [knott_seal_probability(dn) for dn in (0.1, 0.3, 0.7, 1.5)]
    assert p == sorted(p) and p[-1] >= 0.9 and 0.1 <= p[0] <= 0.35


def test_faces_lie_between_open_and_sealed_by_the_clay_that_slid_past():
    """Not a switch: between sands of one permeability each face's multiplier falls continuously as more clay slid past
    it, from 0.46 (clean sand, SGR 0.1) to 0.03 (half clay, SGR 0.5) on a vertical fault with 6 m of throw, where most
    of Norne's history-matched faults sit (10 m beds of 0.1 and 0.5 clay: the window takes six SGR levels)."""
    layer, vsh = cake(shale_vsh=0.5, beds=4)
    layer.perm_mat[:] = 100.0
    f = Fault(center=(510.0, 250.0), strike=90.0, length=20000.0, throw=6.0, dip=90.0, drag=(0.0, 0.0), name="F1")
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=[f], _faces=faces)
    perm = np.asarray(layer.perm_mat)[:, :, ::-1]
    info = face_multipliers(faces, zc, act, vsh, (perm, perm, perm), DX, DX, Seal())["faults"][0]
    mult, sgr = np.array(info["mult"]), np.array(info["sgr"])
    levels = np.unique(sgr.round(3))
    assert len(levels) == 6 and levels.min() == pytest.approx(0.1) and levels.max() == pytest.approx(0.5)
    by_level = np.array([mult[sgr.round(3) == s].mean() for s in levels])
    assert np.all(np.diff(by_level) < -0.02)
    assert by_level[0] == pytest.approx(0.457, abs=0.005) and by_level[-1] == pytest.approx(0.0264, abs=0.001)


def test_a_faults_effective_multiplier_follows_the_faces_that_carry_flow():
    """Faces between tight shales carry almost no flow, so the fault's one equivalent multiplier (a MULTFLT passing the
    same flow) follows its sand faces (SGR 0.1, 100 mD: 0.44 at 70 degrees dip), not the plain mean of all faces, which
    the shale faces (near 1 against the shale's own permeability) pull up to 0.76."""
    layer, vsh = cake(beds=4)
    f = Fault(center=(510.0, 250.0), strike=90.0, length=20000.0, throw=6.0, dip=70.0, name="F1")
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=[f], _faces=faces)
    perm = np.asarray(layer.perm_mat)[:, :, ::-1]
    info = face_multipliers(faces, zc, act, vsh, (perm, perm, perm), DX, DX, Seal())["faults"][0]
    mult = np.array(info["mult"])
    assert 0.40 < info["effective"] < 0.47
    assert 10.0 ** np.mean(np.log10(mult)) > 1.5 * info["effective"]


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
    """With the defaults every face keeps its physics multiplier, as before the spread existed, and a fixed offset
    shifts the faults without drawing a random number."""
    faces, rest = tiny_fault()
    base = face_multipliers(faces, *rest, Seal())["faults"][0]
    assert base["mode"] == "seal" and len(base["mult"]) == 28

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
    """With no open or enhancing share the stream is what it was: one normal per fault, in the order of the faults
    (read on the face with the lowest multiplier, which the cap at 1 does not touch)."""
    base = np.array(many_faults(Seal(), 1)[0]["mult"])
    low = int(np.argmin(base))
    rng = np.random.default_rng(9)
    for info in many_faults(Seal(scatter=0.3, seed=9), 5):
        assert np.array(info["mult"])[low] / base[low] == pytest.approx(10.0 ** rng.normal(0.0, 0.3), rel=1e-12)


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


def dome_model(nx=80, ny=60, dx=50.0, top=TOP):
    layer = Layer(nx, ny, 10, nx * dx, ny * dx, 30.0, top_depth=top, kzkx=0.1)
    layer.poro_mat = np.full((nx, ny, 10), 0.2)
    layer.perm_mat = np.full((nx, ny, 10), 100.0)
    fold = st.closure(area=4e6, height=80.0, aspect=1.5, center=(0.5 * nx * dx, 0.5 * ny * dx))
    return layer, fold


CREST_FAULT = dict(center=(2000.0, 1500.0), strike=90.0, length=20000.0, dip=89.0, drag=(0.0, 0.0))


def dome_blocks(vsh=0.9, perm=100.0, throw=8.0, faults=None, top=TOP, **capillary):
    """``fault_blocks`` of the 30 m thick dome (``dome_model``) cut through its crest by a fault of ``throw`` m, or by
    ``faults``, with clay fraction ``vsh`` and permeability ``perm`` (a number, or one per layer from the top) under
    ``Capillary(delta_rho=300, **capillary)``: the ``blocks``, the multipliers ``mults``, ``cap`` and the top
    ``depth``."""
    layer, fold = dome_model(top=top)
    faults = faults or [Fault(throw=throw, name="F1", **CREST_FAULT)]
    faces = []
    _, _, zc, act = _build_geometry([layer], structure=fold, faults=faults, _faces=faces)
    full = lambda v: np.broadcast_to(np.asarray(v, dtype=float), act.shape)
    mults = face_multipliers(faces, zc, act, full(vsh), (full(perm),) * 3, 50.0, 50.0, Seal())
    cap = Capillary(**{"delta_rho": 300.0, **capillary})
    depth = 0.25 * (zc[0::2, 0::2] + zc[1::2, 0::2] + zc[0::2, 1::2] + zc[1::2, 1::2])[..., 0]
    blocks = fault_blocks(zc, act, faces, mults, 50.0, 50.0, cap)
    return SimpleNamespace(blocks=blocks, mults=mults, cap=cap, depth=depth)


def two_blocks(res):
    """The two blocks of a fault through the crest, the upthrown (shallower crest) first."""
    big = [b for b in res.blocks if b["area"] > 0.2e6]
    assert len(big) == 2
    return big


def column(res):
    """The oil column (m) each net face of the fault holds, from its capacity: H = 1e5 P / (g delta_rho)."""
    rec = res.mults["face_records"]
    rec = rec[rec["perm"] >= res.cap.net_perm]
    return 1e5 * seal_capacity(rec["sgr"], rec["depth"] - res.cap.mudline, res.cap) / (9.81 * res.cap.delta_rho)


def test_seal_capacity_follows_bretans_envelope():
    """Bretan, Yielding & Jones (2003) eq. 1, 10^(100 SGR/27 - C) bar with C = 0.5 below 3 km burial and 0.25 for
    3-3.5 km (the research's table: 1.74 bar at SGR 20 % and 4.08 at 30 %, then 3.10 at 20 % and C 0.25): at most the
    plateau, the floor below the onset or without the membrane, and the plateau above 3.5 km, where oil is
    uncalibrated."""
    wide = Capillary(delta_rho=300.0, plateau=100.0)
    plain = Capillary(delta_rho=300.0)
    assert seal_capacity(0.20, 2000.0, wide) == pytest.approx(1.74, abs=0.005)
    assert seal_capacity(0.30, 2000.0, wide) == pytest.approx(4.08, abs=0.005)
    assert seal_capacity(0.20, 3200.0, wide) == pytest.approx(3.10, abs=0.005)
    assert seal_capacity(0.30, 2000.0, plain) == plain.plateau
    assert seal_capacity(0.19, 2000.0, wide) == wide.floor
    assert seal_capacity(0.90, 2000.0, Capillary(delta_rho=300.0, membrane=False)) == plain.floor
    assert seal_capacity(0.25, 4000.0, plain) == plain.plateau
    sgr, burial = np.array([0.1, 0.2, 0.3]), np.array([2000.0, 3200.0, 4000.0])
    assert seal_capacity(sgr, burial, wide) == pytest.approx([0.5, 3.10, 100.0], abs=0.005)


def test_bretans_envelope_steps_up_by_burial_class():
    """Eq. 1 at SGR 0 gives 0.32, 0.56 and 1.00 bar for burial below 3 km, at 3-3.5 km and above 3.5 km (the research's
    table); the classes' edges belong to the shallower class."""
    steps = bretan_pressure(0.0, [2999.0, 3000.0, 3500.0, 3501.0])
    assert steps == pytest.approx([0.316, 0.562, 0.562, 1.0], abs=0.001)
    assert bretan_pressure(0.0, 2000.0) == pytest.approx(0.316, abs=0.001) and bretan_pressure(0.0, 4000.0) == 1.0


@pytest.mark.parametrize("kw", [dict(delta_rho=0.0), dict(delta_rho=-50.0), dict(onset=1.2), dict(floor=-0.1),
                                dict(plateau=0.2)])
def test_capillary_values_that_make_no_sense_are_refused(kw):
    with pytest.raises(ValueError):
        Capillary(**{"delta_rho": 300.0, **kw})


def test_every_face_with_a_multiplier_leaves_a_record_on_a_lateral_edge():
    """fault_blocks needs each face's depth, SGR and permeability: one record per face that got a multiplier, a Z face
    (a tread inside one column) on the nearest lateral (X or Y) edge of its fault, as for its throw."""
    layer, vsh = cake(beds=4)
    f = Fault(center=(510.0, 250.0), strike=90.0, length=20000.0, throw=6.0, dip=70.0, name="F1")
    faces = []
    _, _, zc, act = _build_geometry([layer], faults=[f], _faces=faces)
    perm = np.asarray(layer.perm_mat)[:, :, ::-1]
    out = face_multipliers(faces, zc, act, vsh, (perm, perm, perm), DX, DX, Seal())
    rec, info = out["face_records"], out["faults"][0]
    raw = face_records("F1", faces[0][1], zc, act)
    lateral = list(dict.fromkeys((face, i, j) for _, i, _, j, _, _, _, face in raw if face != "Z"))
    expected = []                                     # the edge of every face: its own, or for a Z face the nearest
    for _, i, _, j, _, k1, k2, face in raw:
        edge = min(lateral, key=lambda e: abs(e[1] - i) + abs(e[2] - j)) if face == "Z" else (face, i, j)
        expected += [("XY".index(edge[0]), edge[1] - 1, edge[2] - 1)] * (k2 - k1 + 1)
    assert any(face == "Z" for *_, face in raw) and len(rec) == len(info["sgr"]) == len(expected)
    assert sorted(zip(rec["axis"], rec["i"], rec["j"])) == sorted(expected)
    assert np.array_equal(rec["sgr"], info["sgr"]) and np.all((rec["depth"] > TOP) & (rec["depth"] < TOP + 50.0))
    assert set(np.unique(rec["perm"])) == {0.01, 100.0} and np.all(rec["fault"] == 0)


def test_a_fault_that_holds_gives_its_blocks_their_own_contacts():
    """Shale-rich gouge (SGR 0.9, the plateau of 4 bar: 136 m of oil at delta_rho 300, more than the trap's 80 m) holds
    the contact difference of a fault through the crest: each block fills to its own spill point, so the contacts differ
    by the 8 m throw, more than 0 and at most the column the weakest face holds, and the blocks are two
    accumulations."""
    res = dome_blocks(vsh=0.9)
    up, down = two_blocks(res)
    gap = down["contact_depth"] - up["contact_depth"]
    assert column(res).min() == pytest.approx(135.9, abs=0.1)
    assert 0.0 < gap <= column(res).min() and gap == pytest.approx(8.0, abs=0.5)
    assert up["group"] != down["group"] and up["limited_by"] == down["limited_by"] == "spill"
    for blk in (up, down):
        assert blk["height"] == pytest.approx(blk["contact_depth"] - blk["crest_depth"]) and blk["height"] > 0.0


def test_a_fault_that_does_not_hold_leaves_one_accumulation():
    """Clean sand on sand (SGR below the onset) holds the floor, 0.5 bar = 17 m of oil at delta_rho 300: the contacts
    differ by at most that (here not at all, the weakest window lies far above the spill point), and the blocks share
    one contact: the upthrown block spills, the downthrown one leaks into it across the fault."""
    res = dome_blocks(vsh=0.0)
    up, down = two_blocks(res)
    assert column(res).max() == pytest.approx(17.0, abs=0.1)
    assert abs(down["contact_depth"] - up["contact_depth"]) <= column(res).max()
    assert up["group"] == down["group"]
    assert (up["limited_by"], down["limited_by"]) == ("spill", "leak")
    assert up["mask"][down["point"]]                                  # it leaks into the other block's trap


def test_a_window_below_the_other_blocks_spill_sets_the_contact():
    """The weakest window 4 m below the upthrown block's spill point: the downthrown block, whose own spill point lies
    8 m deeper, holds oil down to that window and no further (limited by the leak), so the fault holds the 4 m."""
    res = dome_blocks(vsh=0.0, membrane=False)
    spill = two_blocks(res)[0]["contact_depth"]
    window = res.mults["face_records"]["depth"].min()                 # the shallowest face of the fault
    held = spill + 4.0 - window                                       # the column that puts its leak level there
    res = dome_blocks(vsh=0.0, membrane=False, delta_rho=1e5 * res.cap.floor / (9.81 * held))
    up, down = two_blocks(res)
    assert up["contact_depth"] == spill and down["contact_depth"] == pytest.approx(spill + 4.0, abs=1e-6)
    assert (up["limited_by"], down["limited_by"]) == ("spill", "leak") and up["group"] != down["group"]
    assert up["mask"][down["point"]]


def test_a_throw_beyond_the_reservoir_is_a_wall():
    """No bed meets a bed across a throw of 40 m in a 30 m reservoir, so the fault has no window: a wall. Each block
    fills to its own spill point, a throw apart."""
    res = dome_blocks(vsh=0.0, throw=40.0)
    up, down = two_blocks(res)
    assert len(res.mults["face_records"]) == 0
    assert down["contact_depth"] - up["contact_depth"] == pytest.approx(40.0, rel=0.1)
    assert up["limited_by"] == down["limited_by"] == "spill" and up["group"] != down["group"]


def test_sand_against_shale_seals_whatever_the_gouge():
    """A net bed against a non-net one never leaks. The top 5 layers are sand (100 mD) over 5 of shale (0.01 mD) and
    the 20 m throw is more than the sand's 15 m, so every face is sand against shale, clean gouge (SGR 0) and all. Make
    the shale net and the same clean faces leak."""
    res = dome_blocks(vsh=0.0, perm=np.where(np.arange(10) < 5, 100.0, 0.01), throw=20.0)
    rec = res.mults["face_records"]
    up, down = two_blocks(res)
    assert len(rec) > 0 and np.all(rec["perm"] < res.cap.net_perm)
    assert up["group"] != down["group"] and down["contact_depth"] - up["contact_depth"] == pytest.approx(20.0, rel=0.1)
    up, down = two_blocks(dome_blocks(vsh=0.0, throw=20.0))
    assert up["group"] == down["group"]


def test_without_the_membrane_every_net_face_holds_the_floor():
    """With the membrane off the clay in the gouge gives no capillary seal: every net-on-net face holds the floor and
    shale-rich gouge leaks as clean sand does (with it, the same fault holds its two contacts apart)."""
    res = dome_blocks(vsh=0.9, membrane=False)
    up, down = two_blocks(res)
    assert len(column(res)) > 0 and np.allclose(column(res), 17.0, atol=0.1)
    assert up["group"] == down["group"] and up["contact_depth"] == down["contact_depth"]
    held_up, held_down = two_blocks(dome_blocks(vsh=0.9))
    assert held_up["group"] != held_down["group"]


def test_burial_is_counted_below_the_mudline():
    """The same faces at 3.3 km depth hold more at 3.3 km burial (C 0.25, 4.7 bar at SGR 25 %) than under 1 km of
    water (burial 2.3 km, C 0.5, 2.7 bar): at delta_rho 430 the first holds the trap (112 m), the second leaks
    (63 m)."""
    kw = dict(vsh=0.25, top=3300.0, plateau=100.0, delta_rho=430.0)
    up, down = two_blocks(dome_blocks(**kw))
    assert up["group"] != down["group"]
    up, down = two_blocks(dome_blocks(mudline=1000.0, **kw))
    assert up["group"] == down["group"]


def test_a_block_closed_all_round_is_sealed_and_fills_to_its_deepest_point():
    """Four faults with walls (throws beyond the reservoir) round a horst: nothing reaches the block, so it fills to its
    deepest column, with no spill or leak point."""
    sides = (((1500.0, 1500.0), 90.0, -1, "W"), ((2500.0, 1500.0), 90.0, 1, "E"),
             ((2000.0, 1000.0), 0.0, -1, "S"), ((2000.0, 2000.0), 0.0, 1, "N"))
    box = [Fault(center=c, strike=strike, length=20000.0, throw=40.0, dip=89.0, drag=(0.0, 0.0), hanging_wall=side,
                 name=name) for c, strike, side, name in sides]
    res = dome_blocks(vsh=0.0, faults=box)
    horst = res.blocks[0]
    assert horst["limited_by"] == "sealed" and horst["point"] is None
    assert horst["contact_depth"] == res.depth[30:50, 20:40].max()
    assert horst["mask"].sum() >= 390 and not horst["mask"][:30].any() and not horst["mask"][50:].any()
    assert all(blk["limited_by"] != "sealed" for blk in res.blocks[1:])


def test_without_faults_the_block_is_the_structures_closure():
    """No fault, no window: one block, the closure ``closure_stats`` measures on the same top surface."""
    layer, fold = dome_model()
    faces = []
    _, _, zc, act = _build_geometry([layer], structure=fold, _faces=faces)
    full = np.ones(act.shape)
    mults = face_multipliers(faces, zc, act, 0.0 * full, (100.0 * full,) * 3, 50.0, 50.0, Seal())
    (blk,) = fault_blocks(zc, act, faces, mults, 50.0, 50.0, Capillary(delta_rho=300.0))
    depth = 0.25 * (zc[0::2, 0::2] + zc[1::2, 0::2] + zc[0::2, 1::2] + zc[1::2, 1::2])[..., 0]
    stats = st.closure_stats(depth, 50.0, 50.0)
    assert blk["area"] == stats["area"] and blk["height"] == stats["height"] and blk["crest"] == stats["crest"]
    assert blk["contact_depth"] == stats["spill_depth"] and np.array_equal(blk["mask"], stats["mask"])
    assert blk["limited_by"] == "spill" and blk["group"] == 0 and depth[blk["point"]] == stats["spill_depth"]
    assert not blk["mask"][blk["point"]]


def test_dead_columns_leave_the_blocks_as_they_were():
    """Columns without an active cell (eroded or pinched out) belong to no block and oil does not cross them: a dead
    strip along the model's edge changes neither the blocks nor the contacts, and the flood ends."""
    layer, fold = dome_model()
    f = Fault(throw=8.0, name="F1", **CREST_FAULT)
    faces = []
    _, _, zc, act = _build_geometry([layer], structure=fold, faults=[f], _faces=faces)
    full = np.ones(act.shape)
    mults = face_multipliers(faces, zc, act, 0.9 * full, (100.0 * full,) * 3, 50.0, 50.0, Seal())
    cap = Capillary(delta_rho=300.0)
    before = fault_blocks(zc, act, faces, mults, 50.0, 50.0, cap)
    act = act.copy()
    act[:, :4] = False
    after = fault_blocks(zc, act, faces, mults, 50.0, 50.0, cap)
    assert [(b["crest"], b["contact_depth"], b["area"], b["group"]) for b in after] == \
        [(b["crest"], b["contact_depth"], b["area"], b["group"]) for b in before]
