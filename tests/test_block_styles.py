"""Block styles (structure step 4): tilted fault blocks and rollovers on listric growth faults.

The numbers they are held to are independent of the code: the research's own (``step4_blocks_rollovers.md``, rows T1-T28),
the closed forms of Fossen & Hesthammer's rigid domino (1998), and measurements on the finished grid.
"""
import dataclasses
import json
import math

import numpy as np
import pytest

from resmill.block_styles import BlockModel, rollover, tilted_blocks
from resmill.export import _build_geometry, to_grdecl
from resmill.faults import Fault
from resmill.layers.base import Layer

from .fault_helpers import lateral_contacts, log_plane, research_rollover

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
    """The record carries the pinned values, the throws the blocks have, the extension beta of Fossen & Hesthammer's eq. 9 (1.41 at 15 and 30
    degrees; the pitch over the horizontal width, 1.46, is its own label), and a count of faults by kind."""
    m, _, _, t = gullfaks
    L = m.labels
    assert (L["kind"], L["tilt_deg"], L["dip_deg"], L["width_m"], L["azimuth_deg"]) == ("domino", 15.0, 30.0, 2000.0, 90.0)
    assert L["beta"] == pytest.approx(1.41, abs=0.005)                         # Fossen & Hesthammer's eq. 9 at 15 and 30 degrees
    assert L["pitch_over_width"] == pytest.approx(1.464, abs=0.001)            # the fault's pitch over its block's horizontal width: 1 + 1 / 2
    assert L["n_main"] == len([f for f in m.faults if f.kind == "block"]) == len(L["throws_m"])
    assert L["fault_kinds"]["block"] == L["n_main"] and L["n_faults"] == len(m.faults)
    assert np.sum(L["throws_m"][1:-1]) == pytest.approx(np.sum(t[:-1, 3] - t[1:, 1]), rel=0.1)
    assert isinstance(m, BlockModel) and m.isochore is None
    assert json.loads(json.dumps(L)) == L                                     # plain numbers, for the episode record


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


def test_a_trap_is_not_joined_across_a_sealing_fault_inside_it():
    """Two culminations 1.6 km apart share one closing contour (the saddle between them is above their flat level), so with no fault the
    trap is one, 11.6 km2 over both. A fault through the saddle (40 km long, a metre of throw, which cuts no cell's thickness: every corner
    keeps its 4 m) is a wall: the two sides are two traps, and the one measured is the larger, exactly the free trap's western part (the
    cells west of the fault at x = 4,000 m), none of it east of the wall."""
    from resmill import structure as st
    from resmill.block_styles import _measure
    fold = st.closure(area=8e6, height=60.0, aspect=1.0, center=(3200.0, 3000.0)) + \
        st.closure(area=5e6, height=60.0, aspect=1.0, center=(4800.0, 3000.0))
    wall = Fault(center=(4000.0, 3000.0), strike=90.0, length=40000.0, throw=1.0, dip=60.0, hanging_wall=1, hw_share=1.0, drag=(0.0, 0.0),
                 z_center=TOP + 2.0, name="F1")
    free, walled = (_measure(8000.0, 6000.0, 100.0, TOP, fold, faults) for faults in ([], [wall]))
    west = free["mask"][:40]                                                 # cells whose centres lie west of x = 4,000 m
    assert free["mask"][:40].any() and free["mask"][40:].any() and free["area"] > 11e6
    assert not walled["mask"][40:].any()
    assert np.array_equal(walled["mask"][:40], west) and walled["area"] == pytest.approx(west.sum() * 1e4)
    assert walled["crest_xy"][0] < 4000.0


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
        assert 0.5 <= m.labels["tilt_deg"] <= 5.0 and m.labels["beta"] == m.labels["pitch_over_width"] == 1.0


def test_a_dominos_faults_are_long_enough_for_a_displacement_over_length_of_a_tenth():
    """T10 (Lathrop et al. 2022: 94 % of faults below D / L = 0.1): blocks 5 km wide tilted 20 degrees between faults dipping 30 degrees
    throw 1.5 km (the cap), a displacement of 3 km, so each fault is at least 30 km long, not the 12 km of 1.2 diagonals of this 8 x 6 km map."""
    m = tilted_blocks(*SMALL, 3, kind="domino", azimuth=90.0, tilt=20.0, dip=30.0, width=5000.0, scatter=0.0, density=0.0)
    L = m.labels
    assert max(L["throws_m"]) == pytest.approx(1500.0, rel=1e-6)
    assert max(L["throws_m"]) / math.sin(math.radians(30.0)) / L["length_m"] <= 0.1 + 1e-9
    assert L["length_m"] > 1.2 * math.hypot(8000.0, 6000.0) and all(f.length == L["length_m"] for f in blocks_of(m))


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
    assert 0.15 < np.log10(density).std(ddof=1) < 0.42                         # a log10 sd of 0.3 [J]: the range's ends are one sd out
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


# ----- rollovers on listric growth faults -----

def test_a_frio_masters_rollover_is_the_explicit_ramp_and_flattening_construction():
    """Pinned (ramp 55 degrees to 1,800 m, tan(dip) falling by 1/e every 2.5 km below it, 300 m of throw in a horizon at 2.5
    km, no regional dip), the master is a ramp down to 700 m above the reservoir and a flattening plane below, its tip
    ellipse's centre deeper still: the top of a thin reservoir follows the explicit construction of vertical shear over that
    plane (fault_helpers.research_rollover: z + F(x) - F(x - H)) to 1.5 m, the footwall does not move at all, and the
    hanging-wall cutoff is 300 m down."""
    from resmill.faults import _plane
    x_len, y_len, dx, top = 16000.0, 1000.0, 10.0, 2499.0
    m = rollover(x_len, y_len, dx, top, 2.0, 1, kind="frio", azimuth=90.0, dip=55.0, flatten=2500.0, ramp_base=1800.0, throw=300.0,
                 regional_dip=0.0, density=0.0, length=12000.0, expansion=1.0)
    f, = [f for f in m.faults if f.kind == "master"]
    x, z = sawtooth(m, x_len, y_len, dx, "master", top, 2.0)
    assert f.ramp_base == 1800.0 and m.labels["ramp_base_m"] == 1800.0 and f.z_center > 2500.0
    heave, x_c, depth = research_rollover(f.dip, f.ramp_base, f.flatten, top, f.throw, z_anchor=f.z_center, z_ref=f.z_center)
    cutoff = f.center[0] + float(_plane(f, f.z_center)[1](top))            # the footwall cutoff of the top, hanging wall east
    assert cutoff == pytest.approx(f.center[0] + x_c, abs=1e-6)
    assert np.all(z[x < cutoff - 2 * dx] == top)
    east = x > cutoff + heave + 3 * dx
    assert np.abs(z[east] - depth(x[east] - cutoff)).max() < 1.5
    assert z.max() - top == pytest.approx(300.0, abs=1.5)


@pytest.mark.parametrize("regional", [1.0, 2.0, 3.0])
def test_a_rollovers_crest_and_relief_on_the_grid_are_the_explicit_constructions(regional):
    """Pinned (ramp 60 degrees to 1,500 m, L = 2.5 km, 300 m of throw, a regional dip of 1, 2 and 3 degrees toward the basin) and the
    master straightened, the roll's crest, where its dip falls to the regional dip, is where the explicit construction puts it
    to a cell, and it stands as high above the hanging-wall cutoff within 0.05 m. The construction is z0 + F(h) - F(h - H) on the
    ramped top with ONE heave H for the whole hanging wall (that of the horizon at the tip ellipse's centre) and F the plane
    written out with np.log and bisection: a few km out and a closure of its own, set by the throw, the flattening length and the
    regional dip, not by an arc's end. (A heave taken at each column's own depth left the grid off it by 3, 6 and 10 m at 1, 2 and
    3 degrees and the closure at 3 degrees a third short: review of step 4, F1.)"""
    x_len, y_len, dx, top = 16000.0, 1000.0, 10.0, 2499.0
    m = rollover(x_len, y_len, dx, top, 2.0, 1, kind="frio", azimuth=90.0, dip=60.0, flatten=2500.0, ramp_base=1500.0, throw=300.0,
                 regional_dip=regional, density=0.0, length=12000.0, expansion=1.0)
    f, = [f for f in m.faults if f.kind == "master"]
    x, z = sawtooth(m, x_len, y_len, dx, "master", top, 2.0)
    plane, inverse = log_plane(f.dip, f.ramp_base, f.flatten)
    shift = float(inverse(f.z_center))                                       # the bend's trace to the trace at z_center, where `center` is
    heave = float(inverse(f.z_center + f.throw)) - shift                     # one heave for every column and horizon
    tan_a = math.tan(math.radians(regional))
    z0 = top + tan_a * (x - 0.5 * x_len)                                     # the ramped top at each corner, before the fault
    h = x - f.center[0] + shift                                              # from the bend's trace
    built = z0 + plane(h) - plane(h - heave)
    h_cut = 0.0
    for _ in range(50):                                                      # the footwall cutoff of the top: the plane meets the ramped top
        h_cut = float(inverse(top + tan_a * (f.center[0] + h_cut - shift - 0.5 * x_len)))
    cut = f.center[0] + h_cut - shift
    hanging = x > cut + heave + 3 * dx
    assert np.abs(z[hanging] - built[hanging]).max() < 0.05
    crest_grid, crest_built = np.argmin(np.where(hanging, z, np.inf)), np.argmin(np.where(hanging, built, np.inf))
    assert abs(x[crest_grid] - x[crest_built]) <= 2 * dx
    first = np.argmax(hanging)
    assert z[first] - z[crest_grid] == pytest.approx(built[first] - built[crest_built], abs=0.05)
    assert 1000.0 < x[crest_built] - cut < 8000.0 and z[first] - z[crest_grid] > 20.0     # a closure of its own, a few km out


@pytest.mark.parametrize("expansion", [1.0, 2.0])
def test_the_hanging_wall_zones_next_to_the_master_are_as_thick_as_the_index_says(expansion):
    """T18, Thorsen's index (downthrown over upthrown thickness): one heave moves every column of the hanging wall as a unit, so the
    zones' thickness there is the footwall's times the index asked for and nothing else. Three zones of a 90 m reservoir, a 250 m
    throw and ramp 60 degrees, L = 2.5 km, a regional dip of 1 degree, the master as drawn (its trace wanders: the growth isochore and the
    displacement share one frame): on the grid, 100 m, 1 km and 2.5 km beyond the hanging-wall cutoff the middle zone is 1.000 (no growth
    asked for) or 2.000 times as thick as in the footwall, to 0.1 %. (Taking each horizon's heave at its own depth made 1.0 into 1.15-1.17
    and 2.0 into 2.31-2.35 at 100 m, and 1.08 and 2.16-2.19 at 2.5 km, for the five of seeds 0-7 with a footwall on the map: review of step 4, F1.)"""
    from resmill.faults import _frame, _plane
    x_len, y_len, dx, thick = 24000.0, 1500.0, 25.0, 90.0
    m = rollover(x_len, y_len, dx, TOP, thick, 1, kind="frio", azimuth=90.0, dip=60.0, flatten=2500.0, ramp_base=1500.0, throw=250.0,
                 regional_dip=1.0, density=0.0, length=20000.0, expansion=expansion, zones=3)
    nx, ny = int(x_len / dx), int(y_len / dx)
    layers = [Layer(nx, ny, 1, x_len, y_len, thick / 3.0, top_depth=TOP + k * thick / 3.0, kzkx=0.1) for k in range(3)]
    f, = [f for f in m.faults if f.kind == "master"]
    _, _, zc, _ = _build_geometry(layers, structure=m.structure, faults=[f], isochore=m.isochore)
    j = ny // 2
    z = 0.25 * (zc[0::2, 2 * j] + zc[1::2, 2 * j] + zc[0::2, 2 * j + 1] + zc[1::2, 2 * j + 1])          # (nx, 4 interfaces)
    zone = np.diff(z, axis=1)
    x = (np.arange(nx) + 0.5) * dx
    trace = _plane(f, f.z_center)[1]
    heave = float(trace(f.z_center + f.throw) - trace(f.z_center))
    _, h_centre = _frame(f, np.array([f.center[0]]), np.array([f.center[1]]))       # the wandering trace's offset at the centre, along the dip
    cutoff = f.center[0] + float(trace(TOP + thick + 20.0)) + heave - h_centre[0]     # the base's hanging-wall cutoff, a little deeper in the ramped top
    for beyond in (100.0, 1000.0, 2500.0):
        i = int(np.argmin(np.abs(x - (cutoff + beyond))))
        assert zone[i, 1] / zone[:20, 1].mean() == pytest.approx(expansion, rel=1e-3)


@pytest.mark.parametrize("kind", ["frio", "wilcox"])
def test_the_hanging_wall_zones_are_thicker_by_the_expansion_index(kind):
    """T18: downthrown over upthrown thickness of every zone is the index asked for (2), far from the faults on either
    side, however many faults share it (the Wilcox bundle's two or three each take the root): footwall zones as built, the
    hanging wall's twice as thick (a modest throw: a deep cutoff's larger heave adds a differential drag of its own to the zone's
    thickness)."""
    x_len, y_len, dx, thick = 16000.0, 1000.0, 50.0, 60.0
    m = rollover(x_len, y_len, dx, TOP, thick, 2, kind=kind, azimuth=90.0, expansion=2.0, regional_dip=1.0, density=0.0,
                 throw=40.0, length=10000.0, zones=3)
    nx, ny = int(x_len / dx), int(y_len / dx)
    layers = [Layer(nx, ny, 3, x_len, y_len, thick / 3.0, top_depth=TOP + k * thick / 3.0, kzkx=0.1) for k in range(3)]
    faults = [dataclasses.replace(f, bends=0.0, seed=None, radius=math.inf) for f in m.faults]
    _, _, zc, _ = _build_geometry(layers, structure=m.structure, faults=faults, isochore=m.isochore)
    j = ny // 2
    z = 0.25 * (zc[0::2, 2 * j] + zc[1::2, 2 * j] + zc[0::2, 2 * j + 1] + zc[1::2, 2 * j + 1])       # (nx, 10 interfaces)
    thickness = np.diff(z[:, [0, 3, 6, 9]], axis=1)                                                  # the three zones
    footwall, hanging = thickness[:10].mean(axis=0), thickness[-30:].mean(axis=0)
    assert footwall == pytest.approx([20.0] * 3, rel=1e-3)
    assert hanging / footwall == pytest.approx([2.0] * 3, rel=0.02)
    assert m.labels["expansion"] == 2.0 and m.labels["expansion_per_fault"] == pytest.approx(2.0 ** (1.0 / m.labels["n_masters"]))


def test_a_listric_masters_displacement_at_its_tip_ellipses_centre_is_the_one_the_clastic_law_gives():
    """The law's displacement is the fault's maximum (Lathrop et al. 2022: Dmax = 0.11 L^0.84, 252 m at 10 km; T10, T19), which a fault's tip
    ellipse has at its centre as `fold_faults` draws it: the throw of the horizon at z_center over the sine of the plane's dip there (read off
    the explicit plane) is the drawn displacement. The record's throw in the reservoir is what the grid has (cutoff to cutoff with no regional
    dip, within 1 % and the 4 m that the roll rises over half a 20 m cell beside the cutoff); the plane is steeper there than at z_center, so
    it is the larger."""
    x_len, y_len, dx, top = 16000.0, 1000.0, 20.0, 2499.0
    for seed in range(4):
        m = rollover(x_len, y_len, dx, top, 2.0, seed, kind="frio", azimuth=90.0, regional_dip=0.0, density=0.0)
        L = m.labels["masters"][0]
        f, = [f for f in m.faults if f.kind == "master"]
        plane, inverse = log_plane(f.dip, f.ramp_base, f.flatten)
        h = float(inverse(f.z_center))
        dip_c = math.atan(float(plane(h + 0.5) - plane(h - 0.5)))             # the plane's dip at the tip ellipse's centre
        assert f.throw / math.sin(dip_c) == pytest.approx(L["displacement_m"], rel=1e-4) and L["centre_throw_m"] == f.throw
        _, z = sawtooth(m, x_len, y_len, dx, "master", top, 2.0)
        assert z.max() - top == pytest.approx(L["throw_m"], rel=0.01, abs=4.0)       # (the first cell top beside the cutoff has risen)
        assert L["throw_m"] > f.throw


def test_the_displacements_follow_the_clastic_law_with_the_norne_scatter():
    """T10: log10 of displacement over 0.11 L^0.84 is normal with mean 0 and sd 0.27 (60 draws, lengths 3-25 km), a little
    narrower here because a model needs a throw large enough to turn the regional dip into a trap (a draw without one is
    redrawn), which trims the small-throw tail."""
    draws = [rollover(16000.0, 3000.0, 250.0, TOP, THICK, seed, kind="frio", density=0.0).labels["masters"][0]
             for seed in range(60)]
    lengths = np.array([d["length_m"] for d in draws])
    assert lengths.min() >= 3000.0 and lengths.max() <= 25000.0
    off = np.log10([d["displacement_m"] / (0.11 * d["length_m"] ** 0.84) for d in draws])
    assert abs(off.mean()) < 0.1 and 0.15 < off.std() < 0.34


@pytest.fixture(scope="module")
def drawn_rollovers():
    """Forty-eight rollover models as the style draws them (16 x 3 km of 250 m cells, no faults inside the trap): the
    masters, the regional dip, the flavour and the growth, which is what these tests look at."""
    return [rollover(16000.0, 3000.0, 250.0, TOP, THICK, seed, density=0.0) for seed in range(48)]


def test_the_rollover_draws_lie_in_the_research_ranges(drawn_rollovers):
    """T14, T17, T18, S14 (6): 60 % Frio (one listric master: a ramp dipping 50-75 degrees down to its bend, six published
    faults 46-77, then tan(dip) falling by 1/e every 1.2-5 km of depth, six fits 0.6-3.1 km with a median of 2.4), 40 % Wilcox
    (two or three planar faults of 50-60 degrees, none listric), lengths 3-25 km (Wilcox 0.8-1.2 times that), regional dip
    0.5-3 degrees, expansion 1.1-2.5 (Wilcox 1.3-2.5; 5 % of the draws to 5)."""
    labels = [m.labels for m in drawn_rollovers]
    frio = [L for L in labels if L["kind"] == "frio"]
    wilcox = [L for L in labels if L["kind"] == "wilcox"]
    assert 0.45 <= len(frio) / len(labels) <= 0.75
    assert all(L["n_masters"] == 1 and 1200.0 <= L["flatten_m"] <= 5000.0 for L in frio)
    assert all(L["n_masters"] in (2, 3) and L["flatten_m"] is None for L in wilcox)
    assert all(50.0 <= L["dip_deg"] <= 75.0 for L in frio) and all(50.0 <= L["dip_deg"] <= 60.0 for L in wilcox)
    assert all(0.5 <= L["regional_dip_deg"] <= 3.0 for L in labels)
    assert all(1.1 <= L["expansion"] <= 5.0 for L in frio) and all(1.3 <= L["expansion"] <= 5.0 for L in wilcox)
    assert np.mean([L["expansion"] <= 2.5 for L in labels]) >= 0.85
    assert all(3000.0 <= d["length_m"] <= 25000.0 for L in frio for d in L["masters"])
    assert all(2400.0 <= d["length_m"] <= 30000.0 for L in wilcox for d in L["masters"])
    assert all(1 <= m.labels["tries"] <= 32 for m in drawn_rollovers)
    assert all(json.loads(json.dumps(m.labels)) == m.labels for m in drawn_rollovers)     # plain numbers, for the episode record
    assert all(f.flatten is None for m in drawn_rollovers if m.labels["kind"] == "wilcox" for f in m.faults)
    assert all(f.flatten == m.labels["flatten_m"] and f.dip == m.labels["dip_deg"] and f.ramp_base == m.labels["ramp_base_m"]
               for m in drawn_rollovers if m.labels["kind"] == "frio" for f in m.faults[:1])
    # log-uniform across the range: the median is the geometric mean of its ends, 2.45 km, the published median
    assert 1800.0 <= np.median([L["flatten_m"] for L in frio]) <= 3300.0


def test_the_master_faults_are_placed_by_the_tip_ellipse_below_the_reservoir(drawn_rollovers):
    """S11 for growth faults: a master's tip ellipse is centred 0.25-0.5 half-heights below the reservoir (the half-height
    taken at the ramp's dip), so that a Wilcox fault's throw grows downward (a Frio master's hanging wall moves as one block, and
    its ellipse only anchors the plane and sets the throw its heave comes from); a Frio master's ramp ends 0-2 km above the
    reservoir."""
    for m in drawn_rollovers:
        az, alpha = math.radians(m.labels["azimuth_deg"]), math.radians(m.labels["regional_dip_deg"])
        for f, rec in zip([f for f in m.faults if f.kind == "master"], m.labels["masters"]):
            along = np.dot(np.array(f.center) - (8000.0, 1500.0), (math.sin(az), math.cos(az)))   # the ramp deepens along n
            half = 0.5 * f.length / 2.15 * math.sin(math.radians(f.dip))
            below = (f.z_center - (TOP + 0.5 * THICK + math.tan(alpha) * along)) / half
            assert 0.25 - 1e-9 <= below <= 0.5 + 1e-9
            assert rec["centre_throw_m"] == f.throw and rec["z_center_m"] == f.z_center
            if f.flatten is None:                 # a planar fault: Dmax at the ellipse's centre, and the reservoir's throw is the profile there
                assert f.throw / math.sin(math.radians(f.dip)) == pytest.approx(rec["displacement_m"], rel=1e-9)
                assert rec["throw_m"] == pytest.approx(f.throw * (1.0 - below) ** 1.5 * math.sqrt(1.0 + 3.0 * below), rel=1e-9)
            if f.flatten is not None:                                   # the Frio ramp ends 0-2 km above the reservoir where the fault is
                z_k = TOP + 0.5 * THICK + math.tan(alpha) * along
                assert z_k - 2000.0 - 1e-6 <= f.ramp_base <= z_k + 1e-6 and rec["ramp_base_m"] == f.ramp_base
            else:
                assert f.ramp_base is None and rec["ramp_base_m"] is None


@pytest.mark.parametrize("fixture, size, kinds", [("drawn_blocks", (8000.0, 6000.0), ("block", "transfer")),
                                                   ("mid_rollovers", (12000.0, 9000.0), ("master",))])
def test_the_faults_inside_a_trap_reach_their_five_metres_at_the_real_top(request, fixture, size, kinds):
    """S6 counts the faults of 5 m or more in the reservoir. They are drawn on a smooth fold of the trap, flat around it, while the real top
    has the tilt, the roll and the steps of the model's own faults (hundreds of metres a block away): the tip ellipse of each is carried by the
    difference between the two depths at its centre. On the planning layer built again from the model's own faults, over every cell that holds
    rock, 95 % of the population reach 5 m in a reservoir of 40 m (before: 73 % of a tilted-block model's, 86 % of a rollover's; the rest of
    the drawn sets, a graben's steps, are not counted at 5 m even where they are drawn)."""
    from resmill.block_styles import PLAN_THICKNESS
    reached = total = 0
    for m in request.getfixturevalue(fixture):
        grid, (x_len, y_len) = m.labels["planning_dx_m"], size
        nx, ny = max(int(round(x_len / grid)), 3), max(int(round(y_len / grid)), 3)
        _, _, zc, _ = _build_geometry([Layer(nx, ny, 1, x_len, y_len, PLAN_THICKNESS, top_depth=TOP)], structure=m.structure,
                                      faults=[f for f in m.faults if f.kind in kinds])
        top, thin = zc[:, :, 0], (zc[:, :, 1] - zc[:, :, 0]).reshape(nx, 2, ny, 2).min(axis=(1, 3))
        depth = 0.25 * (top[0::2, 0::2] + top[1::2, 0::2] + top[0::2, 1::2] + top[1::2, 1::2])
        for f in (f for f in m.faults if f.kind not in kinds):
            i, j = min(int(f.center[0] // (x_len / nx)), nx - 1), min(int(f.center[1] // (y_len / ny)), ny - 1)
            if thin[i, j] < 0.8 * PLAN_THICKNESS:                              # a cut-out: its top is a fault plane, not a horizon
                continue
            half = 0.5 * f.length / f.aspect * math.sin(math.radians(f.dip))
            gap = max(abs(f.z_center - (depth[i, j] + 0.5 * THICK)) - 0.5 * THICK, 0.0)       # to the nearest depth of the reservoir
            share = (1.0 - gap / half) ** 1.5 * math.sqrt(1.0 + 3.0 * gap / half) if gap < half else 0.0       # Walsh and Watterson
            total += 1
            reached += f.throw * share >= 5.0
    assert total > 200 and reached / total >= 0.95


def test_the_graben_and_the_antithetic_share_of_a_rollovers_population_are_the_researchs(monkeypatch):
    """T23, T24: the faults inside a rollover's trap come from fold_faults (style "rollover": the keystone graben half the
    time) with the regional extension across the master and 40-70 % of the regionally oriented faults antithetic."""
    import resmill.block_styles as bs
    calls = []
    real = bs.fold_faults
    monkeypatch.setattr(bs, "fold_faults", lambda *a, **kw: calls.append((a, kw)) or real(*a, **kw))
    grabens = []
    for seed in range(24):
        calls.clear()
        m = rollover(12000.0, 9000.0, 300.0, TOP, THICK, seed, density=0.0)
        if not calls:
            continue
        (args, kw), = calls
        n = (math.sin(math.radians(m.labels["azimuth_deg"])), math.cos(math.radians(m.labels["azimuth_deg"])))
        assert args[0] == "rollover" and 0.5 <= kw["regional"][0] <= 2.0 and 0.3 <= kw["basinward"] <= 0.6
        assert (math.cos(math.radians(kw["regional"][1])), -math.sin(math.radians(kw["regional"][1]))) == pytest.approx(n)
        grabens.append("graben" in m.labels["fault_kinds"])
    assert len(grabens) >= 18 and 0.25 <= np.mean(grabens) <= 0.75


def test_a_rollover_seed_gives_the_same_model_and_pinning_changes_only_what_is_pinned():
    """Explicit mode, as for the blocks: the same seed, the same model; a pinned regional dip (a tenth under the drawn one: a
    trap is still there, so there is no second draw, which would draw everything again but the flavour and direction) is used
    as given and the flavour, direction, flattening length, master length and expansion stay where the seed drew them."""
    args = (16000.0, 12000.0, 250.0, TOP, THICK, 3)
    a, b = rollover(*args, kind="frio", density=0.0), rollover(*args, kind="frio", density=0.0)
    pinned = 0.9 * a.labels["regional_dip_deg"]
    c = rollover(*args, kind="frio", density=0.0, regional_dip=pinned)
    assert a.labels == b.labels and repr(a.faults) == repr(b.faults)
    assert a.labels["tries"] == c.labels["tries"] == 1
    assert c.labels["regional_dip_deg"] == pinned != a.labels["regional_dip_deg"]
    for key in ("kind", "azimuth_deg", "dip_deg", "flatten_m", "expansion"):
        assert c.labels[key] == a.labels[key]
    assert c.labels["masters"][0]["length_m"] == a.labels["masters"][0]["length_m"]
    for bad in (dict(kind="turtle"), dict(kind="frio", flatten=-5.0)):
        with pytest.raises(ValueError):
            rollover(*args, **bad)


@pytest.fixture(scope="module")
def mid_rollovers():
    """Twenty-four rollover models with their population on a 12 x 9 km map of 300 m cells (about a second each)."""
    return [rollover(12000.0, 9000.0, 300.0, TOP, THICK, seed) for seed in range(24)]


def test_the_density_of_the_faults_inside_a_rollover_is_log_normal_about_one_per_km2(mid_rollovers):
    """T28: 0.5-2 faults per km2 with 5 m of throw, a median of 1 and a log10 sd of 0.3 [J] (the range's ends are one sd out): over these 24
    draws the log10 sd is 0.15-0.42 and the median is within 0.3 dex of 1."""
    d = np.log10([m.labels["density_per_km2"] for m in mid_rollovers])
    assert 0.15 < d.std(ddof=1) < 0.42 and abs(np.median(d)) < 0.3


def test_each_zones_growth_starts_at_its_own_footwall_cutoff_and_steps_up_over_the_one_heave():
    """The growth isochore of zone z is laid down at the zone's middle depth, top + (z + 1/2) thickness / zones: its factor is 1 at the
    footwall cutoff of that depth (the explicit ramp-and-flattening plane there), half way up half a heave on and the full index a heave on, the
    one heave of a listric master's hanging wall. Three zones of 40 m, a master with a ramp of 60 degrees to 1,500 m and L = 2.5 km."""
    from resmill.faults import _frame
    thick = 120.0
    m = rollover(16000.0, 3000.0, 250.0, TOP, thick, 1, kind="frio", azimuth=90.0, dip=60.0, flatten=2500.0, ramp_base=1500.0, throw=250.0,
                 regional_dip=0.0, density=0.0, length=12000.0, expansion=2.0, zones=3)
    f, = [f for f in m.faults if f.kind == "master"]
    _, inverse = log_plane(f.dip, f.ramp_base, f.flatten)
    shift = float(inverse(f.z_center))                                       # the bend's trace to the trace at z_center, where `center` is
    heave = float(inverse(f.z_center + f.throw)) - shift
    _, h_centre = _frame(f, np.array([f.center[0]]), np.array([f.center[1]]))   # the bent trace's position at the centre, along the dip
    for z in range(3):
        cut = float(inverse(TOP + (z + 0.5) * thick / 3.0)) - shift          # the footwall cutoff of the zone's middle depth from the centre
        h = cut + heave * np.array([-0.05, 0.5, 1.0])
        x = f.center[0] + h - h_centre[0]
        assert m.isochore[z](x, np.full(3, f.center[1])) == pytest.approx([1.0, 1.5, 2.0], abs=1e-3)


def test_the_closure_of_the_finished_rollovers_is_about_the_rollover_law(mid_rollovers):
    """S14 (3): the closure is measured on the finished model, every fault sealing, and checked against a rollover law of
    about 10 / 40 / 150 m (P10 / P50 / P90): Tom O'Connor 90 m, most Niger Delta columns 15 m or less (T22, T25). Here the
    P50 is within a factor 2 of 40 m, the P10 above 3 m and the P90 within a factor 3 of 150 m, and nearly every model has a
    trap of its own to hold it (the all-sealing assumption makes a column no taller than it could be)."""
    traps = [m.labels["trap"] for m in mid_rollovers]
    assert np.mean([t is not None for t in traps]) >= 0.9
    height = np.array([t["height_m"] if t else 0.0 for t in traps])
    p10, p50, p90 = np.percentile(height, [10, 50, 90])
    assert 20.0 <= p50 <= 80.0 and p10 >= 3.0 and 50.0 <= p90 <= 450.0
    assert all(t["area_km2"] > 0.0 for t in traps if t)


def test_a_rollover_with_no_trap_to_frame_its_faults_is_drawn_again(monkeypatch):
    """The faults inside a trap are drawn on a closure that frames it. Where the masters and the regional dip leave none of
    at least 1 km2 (a drag that cannot turn the dip), unpinned values are drawn again, the flavour and direction kept (seed 2
    takes four draws, seed 3 one); and with no trap at any of the TRIES the model has its masters alone and says so, its
    pinned values as given."""
    import resmill.block_styles as bs
    args = (16000.0, 12000.0, 250.0, TOP, THICK)
    again, first = rollover(*args, 2, kind="frio", density=0.5), rollover(*args, 3, kind="frio", density=0.0)
    assert again.labels["tries"] == 4 and first.labels["tries"] == 1
    assert again.labels["n_faults"] > again.labels["n_masters"] and again.labels["trap"] is not None
    monkeypatch.setattr(bs, "_measure", lambda *a, **k: None)
    bare = rollover(*args, 2, kind="frio", regional_dip=3.0, throw=20.0)
    assert bare.labels["tries"] == bs.TRIES and bare.labels["n_faults"] == bare.labels["n_masters"] == 1
    assert bare.labels["trap"] is None and bare.labels["structure_trap"] is None
    assert bare.labels["regional_dip_deg"] == 3.0 and bare.labels["masters"][0]["throw_m"] == 20.0
    assert bare.labels["kind"] == again.labels["kind"] and bare.labels["azimuth_deg"] == again.labels["azimuth_deg"]


@pytest.mark.parametrize("style, kw", [("rollover", dict(kind="frio", density=0.5)), ("rollover", dict(kind="wilcox", density=0.5)),
                                       ("tilted_blocks", dict(kind="domino", density=0.5, azimuth=60.0))])
def test_faults_lists_every_contact_and_tread_of_a_block_model(tmp_path, style, kw):
    """The research's step 8: FAULTS has every face on which a cell meets a cell of the fault's other side and every tread,
    for every fault of a model (the listric masters and the planar faults alike), found by the brute-force check of the
    fault tests (each shared face sampled between its pillars)."""
    make = rollover if style == "rollover" else tilted_blocks
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


@pytest.mark.parametrize("make, kw", [(tilted_blocks, dict(kind="domino")), (rollover, dict(kind="wilcox"))])
def test_a_given_fold_is_added_to_the_structure_and_draws_nothing(make, kw):
    """``fold`` (a culmination for lateral closure) joins the model's own structure as it is: the same seed's faults are
    where they were, and the structure differs from the one without it by the fold alone."""
    from resmill import structure as st
    fold = st.closure(area=8e6, height=60.0, aspect=1.5, azimuth=20.0, center=(6000.0, 4500.0))
    args = (12000.0, 9000.0, 300.0, TOP, THICK, 4)
    plain, folded = make(*args, density=0.0, **kw), make(*args, density=0.0, fold=fold, **kw)
    x, y = np.random.default_rng(0).uniform(0.0, 9000.0, (2, 50))
    assert folded.structure(x, y) - plain.structure(x, y) == pytest.approx(fold(x, y), abs=1e-9)
    assert [f.center for f in folded.faults if f.kind in ("block", "master")] == [f.center for f in plain.faults if f.kind in ("block", "master")]


@pytest.mark.parametrize("make, kw", [(rollover, dict(kind="frio", zones=1, density=0.6)), (tilted_blocks, dict(kind="domino", density=0.6))])
def test_the_fault_seal_takes_a_block_model_listric_masters_and_growth_included(tmp_path, make, kw):
    """The research's step 8: to_grdecl(faults, seal) writes every face's transmissibility multiplier from its shale gouge ratio
    for a block model (a listric master with growth strata, or a domino's planar faults): MULTX, MULTY and MULTZ between 0 and 1,
    below 1 on some faces and 1 on the rest."""
    from resmill.fault_seal import Seal
    x_len, y_len, dx, nz = 12000.0, 9000.0, 300.0, 6
    m = make(x_len, y_len, dx, TOP, 120.0, 7, **kw)
    nx, ny = int(x_len / dx), int(y_len / dx)
    layer = Layer(nx, ny, nz, x_len, y_len, 120.0, top_depth=TOP, kzkx=0.1)
    rng = np.random.default_rng(0)
    layer.poro_mat, layer.perm_mat = np.full((nx, ny, nz), 0.2), 10.0 ** rng.uniform(0.0, 3.0, (nx, ny, nz))
    to_grdecl(layer, tmp_path / "m.grdecl", structure=m.structure, faults=m.faults, isochore=m.isochore,
              seal=Seal(vsh=rng.uniform(0.0, 0.6, (nx, ny, nz)), scatter=0.2, seed=1))
    text = (tmp_path / "m.grdecl").read_text()
    for key in ("MULTX", "MULTY", "MULTZ"):
        values = []
        for token in text.split(f"\n{key}\n")[1].split("\n/\n")[0].split():
            count, _, value = token.rpartition("*")
            values += [float(value)] * (int(count) if count else 1)
        values = np.array(values)
        assert values.size == nx * ny * nz and values.min() >= 0.0 and values.max() <= 1.0
        assert 0.0 < np.mean(values < 0.999) < 0.5
