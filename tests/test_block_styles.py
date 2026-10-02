"""Block styles (structure step 4): tilted fault blocks and rollovers on listric growth faults.

The numbers they are held to are independent of the code: the research's own (``step4_blocks_rollovers.md``, rows T1-T28),
the closed forms of Fossen & Hesthammer's rigid domino (1998), and measurements on the finished grid.
"""
import dataclasses
import math

import numpy as np
import pytest

from resmill.block_styles import BlockModel, tilted_blocks
from resmill.export import _build_geometry, to_grdecl
from resmill.faults import Fault
from resmill.layers.base import Layer

from .test_faults import lateral_contacts

TOP, THICK = 2500.0, 40.0


# ----- tilted blocks -----

def sawtooth(model, x_len, y_len, dx, kind="block", top=TOP, thickness=THICK):
    """Depth of the top surface along the middle row, across faults striking along y (azimuth 90): x of the cell centres and
    the depth of each cell's top on the finished grid, with only the faults of ``kind`` and their traces straightened (a
    trace that wanders changes the spacing at this one row; the sawtooth closes for the spacing of the fault centre lines)."""
    nx, ny = int(round(x_len / dx)), int(round(y_len / dx))
    layer = Layer(nx, ny, 1, x_len, y_len, thickness, top_depth=top, kzkx=0.1)
    faults = [dataclasses.replace(f, bends=0.0, seed=None, radius=math.inf) for f in model.faults if f.kind == kind]
    _, _, zc, _ = _build_geometry([layer], structure=model.structure, faults=faults, isochore=model.isochore)
    j = ny // 2
    z = 0.25 * (zc[0::2, 2 * j, 0] + zc[1::2, 2 * j, 0] + zc[0::2, 2 * j + 1, 0] + zc[1::2, 2 * j + 1, 0])
    return (np.arange(nx) + 0.5) * dx, z


def teeth(x, z):
    """The sawtooth's blocks: for each, x of its crest (shallowest, the footwall cutoff) and of its deepest point (the
    hanging-wall cutoff of the next fault), and the depths there. The beds deepen with x, so a tooth runs from a crest down
    to the next deepest point; the rise from there to the next crest is the fault's tread."""
    d = np.sign(np.diff(z))
    crest = np.flatnonzero((d[:-1] < 0) & (d[1:] > 0)) + 1
    deep = np.flatnonzero((d[:-1] > 0) & (d[1:] < 0)) + 1
    out = []
    for c in crest:
        later = deep[deep > c]
        if later.size:
            out.append((x[c], z[c], x[later[0]], z[later[0]]))
    return np.array(out)                                    # columns: x crest, z crest, x deepest, z deepest


GULLFAKS = dict(kind="domino", azimuth=90.0, tilt=15.0, dip=30.0, width=2000.0, scatter=0.0, density=0.0)
TAN15, TAN30 = math.tan(math.radians(15.0)), math.tan(math.radians(30.0))


@pytest.fixture(scope="module")
def gullfaks():
    """Gullfaks-like inputs (Fossen & Hesthammer 1998): beds tilted 15 degrees, faults dipping 30, blocks 2 km wide, on a
    16 x 1 km strip of 20 m cells, no faults inside the blocks."""
    m = tilted_blocks(16000.0, 1000.0, 20.0, TOP, THICK, seed=3, **GULLFAKS)
    x, z = sawtooth(m, 16000.0, 1000.0, 20.0)
    return m, x, z, teeth(x, z)


def test_the_domino_blocks_are_tilted_by_the_tilt_and_have_no_net_dip(gullfaks):
    """T6, T9: each block's top dips 15 degrees (within 0.1, fitted clear of its two ends), and the faults take up what
    the tilt gives, so the regional dip over the whole strip is near zero (under 0.5 degrees, against the blocks' 15)."""
    _, x, z, t = gullfaks
    assert len(t) >= 4
    for crest, _, deep, _ in t:
        inside = (x > crest + 300.0) & (x < deep - 300.0)
        assert math.degrees(math.atan(np.polyfit(x[inside], z[inside], 1)[0])) == pytest.approx(15.0, abs=0.1)
    span = (x > t[0, 0]) & (x < t[-1, 0])                    # from the first crest to the last
    assert abs(math.degrees(math.atan(np.polyfit(x[span], z[span], 1)[0]))) < 0.5


def test_the_domino_throw_is_the_block_width_times_the_tangent_of_the_tilt(gullfaks):
    """T9: with no net dip the throw (cutoff to cutoff) across a block's fault is the block width at the horizon times
    tan(tilt): 2 km and 15 degrees give 536 m, and a heave of 928 m at a 30 degree dip (the research's table: 536 m and 1.03 km
    at 27.5 degrees; Gullfaks' faults throw 50-500 m, T8). Block by block it holds to a twelfth (the spacing of the faults varies
    by a fifth, and a fault's cutoffs sit half its heave either side of its centre line); over the blocks to 2 %."""
    _, _, _, t = gullfaks
    width = t[:-1, 2] - t[:-1, 0]                           # a block's top, crest to the next fault's hanging-wall cutoff
    throw = t[:-1, 3] - t[1:, 1]                            # the fault there: that deepest point down to the next crest
    heave = t[1:, 0] - t[:-1, 2]
    assert width.mean() == pytest.approx(2000.0, rel=0.15)
    assert throw == pytest.approx(width * TAN15, rel=0.08)
    assert throw.sum() == pytest.approx(width.sum() * TAN15, rel=0.02)
    assert (throw / width).mean() == pytest.approx(TAN15, rel=0.03)
    assert throw.mean() == pytest.approx(2000.0 * TAN15, rel=0.15)
    assert heave.sum() == pytest.approx(throw.sum() / TAN30, rel=0.02)


def test_the_domino_extension_is_the_rigid_dominos(gullfaks):
    """T6: a rigid domino of beds dipping alpha between faults dipping theta extends by beta = sin(alpha + theta) / sin(theta):
    1.41 at Gullfaks' 15 and 30 degrees (Fossen & Hesthammer 1998, eq. 9). The grid's: a block of horizontal width W and its
    fault's heave H extend a block whose beds are W / cos(alpha) long by (W + H) / (W / cos(alpha))."""
    _, _, _, t = gullfaks
    width, heave = t[:-1, 2] - t[:-1, 0], t[1:, 0] - t[:-1, 2]
    beta = math.cos(math.radians(15.0)) * (width + heave).sum() / width.sum()
    assert beta == pytest.approx(math.sin(math.radians(45.0)) / math.sin(math.radians(30.0)), rel=0.02)
    assert beta == pytest.approx(1.41, abs=0.03)


def test_the_labels_say_what_was_built(gullfaks):
    """The record carries the pinned values, the throws the blocks have, and a count of faults by kind."""
    m, _, _, t = gullfaks
    L = m.labels
    assert (L["kind"], L["tilt_deg"], L["dip_deg"], L["width_m"], L["azimuth_deg"]) == ("domino", 15.0, 30.0, 2000.0, 90.0)
    assert L["beta"] == pytest.approx(1.0 + TAN15 / TAN30)
    assert L["n_main"] == len([f for f in m.faults if f.kind == "block"]) == len(L["throws_m"])
    assert L["fault_kinds"]["block"] == L["n_main"] and L["n_faults"] == len(m.faults)
    assert np.sum(L["throws_m"][1:-1]) == pytest.approx(np.sum(t[:-1, 3] - t[1:, 1]), rel=0.1)
    assert isinstance(m, BlockModel) and m.isochore is None


# ----- the trap a model is measured to have -----

def test_a_long_normal_fault_across_a_ramp_makes_no_trap_of_its_cutoffs_or_its_cut_cells():
    """A ramp deepening toward a fault's hanging wall, the fault longer than the map: no closed contour anywhere, the
    hanging wall's cutoff included (the cells the plane cuts have a corner that lost its thickness and top out on the
    plane, not the horizon: counted as rock they made a sliver 'trap' along every fault, up to its throw high)."""
    from resmill import structure as st
    from resmill.block_styles import _measure
    f = Fault(center=(4000.0, 3000.0), strike=0.0, length=40000.0, throw=120.0, dip=40.0, hanging_wall=1, hw_share=1.0,
              drag=(0.0, 0.0), z_center=TOP + 2.0, name="F1")
    assert _measure(8000.0, 6000.0, 100.0, TOP, st.ramp(2.0, azimuth=0.0, center=(4000.0, 3000.0)), [f]) is None


def test_the_measure_finds_the_closure_a_structure_was_built_with():
    """``closure`` builds a fold whose trap has the area and relief asked for (its own flood measures them): the planning
    map finds them again to a few percent (the rim's cells count whole: 100 m cells overshoot the area by 9 %, 50 m cells by 4 %),
    with no fault in the way."""
    from resmill import structure as st
    from resmill.block_styles import _measure
    fold = st.closure(area=6e6, height=80.0, aspect=2.0, azimuth=30.0, center=(4000.0, 3000.0))
    trap = _measure(8000.0, 6000.0, 50.0, TOP, fold, [])
    assert trap["area"] == pytest.approx(6e6, rel=0.06) and trap["height"] == pytest.approx(80.0, rel=0.03)
    assert trap["crest_xy"] == pytest.approx((4000.0, 3000.0), abs=300.0)
    assert trap["crest_depth"] == pytest.approx(TOP - 80.0, abs=3.0) and trap["spill_depth"] == pytest.approx(TOP, abs=3.0)


# ----- tilted blocks: horst-graben, transfer faults, explicit mode, the style's draws -----

SMALL = (8000.0, 6000.0, 200.0, TOP, THICK)


def blocks_of(m, kind="block"):
    return [f for f in m.faults if f.kind == kind]


def test_a_horst_graben_alternates_its_hanging_walls_with_steep_faults_and_little_tilt():
    """T11: Gullfaks' horst complex has faults of 60-70 degrees and layers near horizontal; the hanging walls alternate (a
    horst, a graben, ...), the blocks extend by nothing (beta 1) and throw 50-500 m each (T8)."""
    for seed in range(6):
        m = tilted_blocks(*SMALL, seed, kind="horst_graben", scatter=0.0, density=0.0)
        faults = blocks_of(m)
        assert [f.hanging_wall for f in faults] == [-1 if k % 2 == 0 else 1 for k in range(len(faults))]
        assert all(55.0 <= f.dip <= 70.0 and 50.0 <= f.throw <= 500.0 for f in faults)
        assert 0.5 <= m.labels["tilt_deg"] <= 5.0 and m.labels["beta"] == 1.0


def test_transfer_faults_cross_the_blocks_with_a_tenth_to_a_third_of_their_throw():
    """T12: up to two steep faults striking across the blocks, throwing 0.1-0.3 of the block faults' median."""
    seen = 0
    for seed in range(12):
        m = tilted_blocks(*SMALL, seed, density=0.0)
        transfers, mains = blocks_of(m, "transfer"), blocks_of(m)
        assert len(transfers) <= 2
        for f in transfers:
            seen += 1
            assert 0.1 <= f.throw / np.median([b.throw for b in mains]) <= 0.3
            assert abs((f.strike - m.labels["azimuth_deg"] - 90.0 + 180.0) % 360.0 - 180.0) < 35.0 and f.dip >= 80.0
    assert seen >= 4


def test_the_same_seed_gives_the_same_model_and_a_pinned_value_changes_only_itself():
    """Explicit mode: what is given is used as given and what is not is drawn as it would have been (the draws are made
    up front, so pinning the tilt leaves the dip, width, azimuth and kind where the seed put them)."""
    a = tilted_blocks(*SMALL, 3, density=0.0)
    b = tilted_blocks(*SMALL, 3, density=0.0)
    c = tilted_blocks(*SMALL, 3, tilt=12.0, density=0.0)
    assert a.labels == b.labels and repr(a.faults) == repr(b.faults)
    assert c.labels["tilt_deg"] == 12.0 and a.labels["tilt_deg"] != 12.0
    for key in ("kind", "dip_deg", "width_m", "azimuth_deg"):
        assert c.labels[key] == a.labels[key]
    with pytest.raises(ValueError, match="kind"):
        tilted_blocks(*SMALL, 3, kind="graben")


@pytest.fixture(scope="module")
def drawn_blocks():
    """Twenty-four models of the style as it draws them, on an 8 x 6 km map of 200 m cells (about a second each)."""
    return [tilted_blocks(*SMALL, seed) for seed in range(24)]


def test_the_draws_lie_in_the_research_ranges_and_every_model_holds_a_trap(drawn_blocks):
    """T1-T5, T11, T28 (the draws) and S14's acceptance: tilts 3-20 degrees (a tenth of the dominos to 30), domino dips 25-35,
    horst-graben 55-70, widths 1.5-5 km, 1-6 block faults, 70 % dominos; and a trap of positive area and relief in at least
    90 % of the draws, 100-300 m high in the median block (T26-T29)."""
    labels = [m.labels for m in drawn_blocks]
    domino = [L for L in labels if L["kind"] == "domino"]
    assert 0.4 <= len(domino) / len(labels) <= 0.9
    assert all(3.0 <= L["tilt_deg"] <= 30.0 for L in domino) and np.mean([L["tilt_deg"] <= 20.0 for L in domino]) >= 0.7
    assert all(25.0 <= L["dip_deg"] <= 35.0 for L in domino)
    assert all(55.0 <= L["dip_deg"] <= 70.0 and 0.5 <= L["tilt_deg"] <= 5.0 for L in labels if L["kind"] == "horst_graben")
    assert all(1500.0 <= L["width_m"] <= 5000.0 and 1 <= L["n_main"] <= 6 for L in labels)
    density = np.array([L["density_per_km2"] for L in labels])                  # log-normal about 1.7 per km2 (T28: 1-3)
    assert density.min() > 0.1 and density.max() < 30.0 and 1.0 <= np.median(density) <= 3.0
    assert np.mean((density >= 1.0) & (density <= 3.0)) >= 0.4
    traps = [L["trap"] for L in labels]
    assert np.mean([t is not None and t["area_km2"] > 0.0 and t["height_m"] > 0.0 for t in traps]) >= 0.9
    assert 60.0 <= np.median([t["height_m"] for t in traps if t]) <= 300.0


def test_the_population_dips_with_the_blocks_for_four_in_five_faults(monkeypatch):
    """The faults inside the trap come from fold_faults on the model's own trap (style "tilted_blocks"): regionally oriented
    across the blocks (R 1.5-3, so 60-75 % of the population), their hanging walls on the main faults' side 80-90 % of the
    time (antithetic 10-20 %, T8)."""
    import resmill.block_styles as bs
    calls = []
    real = bs.fold_faults
    monkeypatch.setattr(bs, "fold_faults", lambda *a, **kw: calls.append((a, kw)) or real(*a, **kw))
    for kind, seed in (("domino", 3), ("horst_graben", 4)):
        calls.clear()
        m = tilted_blocks(*SMALL, seed, kind=kind)
        (args, kw), = calls
        assert args[0] == "tilted_blocks"
        ratio, azimuth = kw["regional"]
        assert 1.5 <= ratio <= 3.0 and 0.8 <= kw["basinward"] <= 0.9
        n = np.array([math.sin(math.radians(m.labels["azimuth_deg"])), math.cos(math.radians(m.labels["azimuth_deg"]))])
        side = -n if kind == "domino" else n                                   # the way the block faults throw down
        assert (math.cos(math.radians(azimuth)), -math.sin(math.radians(azimuth))) == pytest.approx(tuple(side), abs=1e-9)


@pytest.mark.parametrize("style, kw", [("tilted_blocks", dict(kind="domino", density=0.5, azimuth=60.0))])
def test_faults_lists_every_contact_and_tread_of_a_block_model(tmp_path, style, kw):
    """The research's step 8: FAULTS has every face on which a cell meets a cell of the fault's other side and every tread,
    for every fault of a model, found by the brute-force check of the fault tests (each shared face sampled between its
    pillars)."""
    make = tilted_blocks
    x_len, y_len, dx, nz = 12000.0, 9000.0, 300.0, 4
    m = make(x_len, y_len, dx, TOP, 40.0, 7, **kw)
    nx, ny = int(x_len / dx), int(y_len / dx)
    layer = Layer(nx, ny, nz, x_len, y_len, 40.0, top_depth=TOP, kzkx=0.1)
    layer.poro_mat, layer.perm_mat = np.full((nx, ny, nz), 0.2), np.full((nx, ny, nz), 100.0)
    faces = []
    _, _, zc, act = _build_geometry([layer], structure=m.structure, faults=m.faults, isochore=m.isochore, _faces=faces)
    to_grdecl(layer, tmp_path / "m.grdecl", structure=m.structure, faults=m.faults, isochore=m.isochore)
    text = (tmp_path / "m.grdecl").read_text().split("\nFAULTS\n")[1].split("\n/\n")[0]
    listed = {}
    for r in (line.split() for line in text.splitlines() if line.strip()):
        listed.setdefault(r[0].strip("'"), set()).update((r[7].strip("'"), int(r[1]) - 1, int(r[3]) - 1, k)
                                                         for k in range(int(r[5]) - 1, int(r[6])))
    assert len(faces) == len(m.faults) > 5 and set(listed) <= {f.name for f in m.faults}
    treads = 0
    for fault, side in faces:
        assert lateral_contacts(side, zc) <= listed.get(fault.name, set())
        for i, j in zip(*np.nonzero((side > 0).any(axis=2) & (side < 0).any(axis=2))):
            live = [k for k in range(nz) if act[i, j, k]]
            for a, b in zip(live, live[1:]):
                if side[i, j, a] * side[i, j, b] == -1:
                    treads += 1
                    assert ("Z", i, j, a) in listed[fault.name]
    assert treads > 0
