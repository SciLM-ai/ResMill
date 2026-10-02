"""``strat_trap``: the kinds, their closures, the refusals, the grid, the wander and the inputs."""
import numpy as np
import pytest

from resmill import structure as st
from resmill.export import _build_geometry
from resmill.strat_traps import strat_trap, trap_report
from tests.strat_helpers import NOSE, _fine_traps, _layers, _length


# The research note's three worked examples (step6_stratigraphic.md, section 3.4): the closure of a pure stratigraphic
# trap is tan(dip) x the trap's length along dip, L = 2 sqrt(A / (pi r)), at the P10, P50 and P90 of BOEM's trap areas
# (0.7, 3.4 and 13.6 km2, Gulf of America sands, 177 fields): 4 m, 27 m (Berg's observed columns are 13 to 37 m) and
# 103 m.
EXAMPLES = [  # dip, area, aspect, model: x, y, cell (m), closure (m)
    (0.5, 0.7e6, 4.0, (6000.0, 4000.0, 25.0), 4.0),
    (1.1, 3.4e6, 2.2, (8000.0, 6000.0, 50.0), 27.0),
    (2.0, 13.6e6, 2.0, (12000.0, 10000.0, 100.0), 103.0),
]


@pytest.mark.parametrize("dip,area,aspect,model,closure", EXAMPLES, ids=["P10", "P50", "P90"])
def test_a_pinch_out_tongue_closes_by_the_dip_times_its_length_and_has_the_area_it_was_given(dip, area, aspect, model,
                                                                                         closure):
    """A sand wedge whose updip edge has a tongue (half an ellipse of the given area and aspect) protruding from a
    sheet: the closure the trap report measures is the dip times the tongue's length along dip, to a cell's rise and
    2 % of it, and the area is the given one plus the columns the edge touches (under 10 %), at BOEM's three trap
    areas."""
    x_len, y_len, dx = model
    built = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=1, dip=dip, taper_angle=0.5, area=area,
                       aspect=aspect, warp=0.0)
    assert built["meta"]["closure_nominal"] == pytest.approx(closure, abs=0.5)
    assert built["meta"]["closure_nominal"] == pytest.approx(np.tan(np.radians(dip)) * _length(area, aspect))
    (trap,) = trap_report(_layers(x_len, y_len, dx, [10.0]), built)
    cell = np.tan(np.radians(dip)) * dx                                   # the depth a cell's width is worth
    assert trap["limited_by"] == "spill"
    assert trap["height"] == pytest.approx(closure, abs=cell + 0.02 * closure)
    assert trap["crest_depth"] == pytest.approx(built["meta"]["crest_nominal"], abs=cell)
    assert trap["spill_depth"] == pytest.approx(built["meta"]["spill_nominal"], abs=cell)
    assert area * 0.97 < trap["area"] < area * 1.10


@pytest.mark.parametrize("azimuth", [30.0, 90.0, 135.0, 250.0, 315.0])
@pytest.mark.parametrize("kind,mound", [("pinchout", False), ("lens", False), ("lens", True)])
def test_a_trap_on_a_dip_in_any_direction_closes_by_the_same_dip_times_its_length(kind, mound, azimuth):
    """The model is a rectangle and the dip may point anywhere: the shape is placed along the dip direction through
    the centre of the model, whatever its azimuth (the P50 trap: 27 m of closure for a pinch-out or a lens hung from
    a flat top, to the rise of a cell and a half and 2 %, and its area to 5 % below and 15 % above). A mound's top
    bulges up by the thickness it has, so its closure is what its own geometry gives, found on the fine map."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    built = strat_trap(kind, x_len, y_len, 2000.0, [10.0], seed=1, dip=1.1, taper_angle=0.5, area=3.4e6, aspect=2.2,
                       warp=0.0, azimuth=azimuth, mound=mound)
    trap = trap_report(_layers(x_len, y_len, dx, [10.0]), built)[0]
    cell = 1.5 * np.tan(np.radians(1.1)) * dx
    fine = _fine_traps(built, 2000.0, x_len, y_len)[0]
    expected = fine["height"] if mound else built["meta"]["closure_nominal"]
    assert trap["height"] == pytest.approx(expected, abs=cell + 0.02 * 27.0)
    assert fine["height"] == pytest.approx(built["meta"]["closure_nominal"], abs=0.5 + (4.0 if mound else 0.0))
    assert trap["crest_depth"] == pytest.approx(fine["crest_depth"] if mound else built["meta"]["crest_nominal"],
                                                abs=cell)
    assert 3.4e6 * 0.95 < trap["area"] < 3.4e6 * 1.15


@pytest.mark.parametrize("kind", ["pinchout", "facies_change", "lens", "truncation", "onlap"])
def test_lobes_warped_by_a_quarter_close_by_the_dip_times_the_length_of_the_shape_that_was_made(kind):
    """The warp of the lobes (a quarter of their width) moves their ends away from the ellipse they were drawn as, so
    the closure follows the length along dip that the footprint really has (the fine map of the geometry), not the
    nominal one: over 8 seeds the measured closure is within 1.5 cells' rise and 8 % of it (the edge is rough, 40 m
    rms)."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    thick = [10.0, 6.0] if kind == "facies_change" else [10.0]
    errors = []
    for seed in range(8):
        try:
            built = strat_trap(kind, x_len, y_len, 2000.0, thick, seed=seed, barrier=kind == "facies_change", dip=1.2,
                               taper_angle=0.6, area=3.0e6, aspect=2.0, warp=0.25, wander=40.0, range_m=1000.0,
                               floor_m=100.0, column=30.0 if kind == "facies_change" else None)
        except ValueError:
            continue
        (trap,) = trap_report(_layers(x_len, y_len, dx, thick), built)[:1]
        expected = _fine_traps(built, 2000.0, x_len, y_len)[0]["height"]
        errors.append(abs(trap["closure"] - expected) - (1.5 * np.tan(np.radians(1.2)) * dx + 0.08 * expected))
        assert built["meta"]["closure_nominal"] == pytest.approx(np.tan(np.radians(1.2)) * _length(3.0e6, 2.0))
    assert len(errors) >= 5 and max(errors) <= 0.0


def test_a_straight_pinch_out_line_on_a_plane_monocline_has_no_closure():
    """Every point of the line lies at one depth, so the oil spills along it: the trap has no height or area."""
    built = strat_trap("pinchout", 8000.0, 6000.0, 2000.0, [10.0], seed=1, dip=1.0, taper_angle=0.5, area=None)
    (trap,) = trap_report(_layers(8000.0, 6000.0, 50.0, [10.0]), built)
    assert trap["height"] == 0.0 and trap["area"] == 0.0
    assert built["meta"]["closure_nominal"] == 0.0


@pytest.mark.parametrize("mound", [False, True], ids=["flat top", "mound"])
@pytest.mark.parametrize("dip,area,aspect,model,closure", EXAMPLES[1:], ids=["P50", "P90"])
def test_an_enclosed_lens_is_sealed_and_holds_the_dip_times_its_whole_length(dip, area, aspect, model, closure, mound):
    """A lens hung from a flat top holds the tangent of the dip times its length along dip. A mound (the base the plane
    of the beds, the top convex up) holds what its own geometry gives, found on the fine map: for a parabolic cap of
    thickness T and length L along dip the top lies T (s / R)^2 below the plane of the base at s from its middle,
    R = L / 2, and with x = tan(dip) L / (4 T) the closure is T (1 + x)^2 for x <= 1 (a gentle dip: the crest inside the
    mound, the mound's own thickness raising it) and 4 x T = tan(dip) L, the nominal, for x >= 1 (the crest at the
    updip rim): at P50 x = 0.67 and 28 m, at P90 x = 2.6 and 103 m."""
    x_len, y_len, dx = model
    built = strat_trap("lens", x_len, y_len, 2000.0, [10.0], seed=2, dip=dip, taper_angle=0.5, area=area,
                       aspect=aspect, warp=0.0, mound=mound)
    (trap,) = trap_report(_layers(x_len, y_len, dx, [10.0]), built)
    assert trap["limited_by"] == "sealed" and trap["spill_depth"] is None
    expected = np.tan(np.radians(dip)) * _length(area, aspect)
    assert built["meta"]["closure_nominal"] == pytest.approx(expected)
    cell = np.tan(np.radians(dip)) * dx
    fine = _fine_traps(built, 2000.0, x_len, y_len)[0]
    assert fine["limited_by"] == "sealed"
    if mound:
        x = expected / 40.0
        assert fine["height"] == pytest.approx(10.0 * (1.0 + x) ** 2 if x <= 1.0 else expected, rel=0.03, abs=0.3)
    else:
        assert fine["height"] == pytest.approx(expected, rel=0.02, abs=0.3)
    assert trap["height"] == pytest.approx(fine["height"], abs=1.5 * cell + 0.02 * expected)
    assert trap["crest_depth"] == pytest.approx(fine["crest_depth"], abs=cell)
    assert area * 0.97 < trap["area"] < area * 1.10


def test_the_same_seed_builds_the_same_trap_and_another_builds_another():
    kw = dict(dip=1.0, taper_angle=0.4, area=2.0e6, aspect=2.0, warp=0.3, wander=50.0, range_m=700.0, cell=50.0)
    a, b = (strat_trap("pinchout", 6000.0, 5000.0, 1800.0, [8.0], seed=s, **kw) for s in (7, 7))
    c = None
    for seed in range(8, 30):                                               # the first seed whose lobes fit the model
        try:
            c = strat_trap("pinchout", 6000.0, 5000.0, 1800.0, [8.0], seed=seed, **kw)
            break
        except ValueError:
            continue
    X, Y = np.meshgrid(np.linspace(0.0, 6000.0, 61), np.linspace(0.0, 5000.0, 51), indexing="ij")
    for key in ("structure", "isochore"):
        va = a["kwargs"][key][0](X, Y) if key == "isochore" else a["kwargs"][key](X, Y)
        vb = b["kwargs"][key][0](X, Y) if key == "isochore" else b["kwargs"][key](X, Y)
        assert np.array_equal(va, vb)
    assert not np.array_equal(a["kwargs"]["isochore"][0](X, Y), c["kwargs"]["isochore"][0](X, Y))
    assert a["kwargs"]["structure"](3000.0, 3500.0) - a["kwargs"]["structure"](3000.0, 2500.0) == pytest.approx(
        1000.0 * np.tan(np.radians(1.0)))                                   # the ramp deepens 1 degree along +y


@pytest.mark.parametrize("args,kw,message", [
    (("seam",), {}, "kind"),
    (("facies_change",), dict(barrier=False), "barrier"),
    (("pinchout", ), dict(barrier=True), "barrier"),
    (("pinchout",), dict(taper_angle=0.0), "taper_angle"),
    (("pinchout",), dict(area=60.0e6), "fit"),
    (("lens",), dict(area=None), "area"),
])
def test_strat_trap_refuses_what_it_cannot_build(args, kw, message):
    kw = dict(dict(x_len=8000.0, y_len=6000.0, top=2000.0, thicknesses=[10.0, 8.0], seed=1), **kw)
    with pytest.raises(ValueError, match=message):
        strat_trap(args[0], **kw)


def _profile(layers, built, i=None):
    """A column of cells along dip (azimuth 0) in the middle of the model: the number of active cells per column, the
    depth of the top of the first active cell (nan where none), the index of the first row each layer is active in
    (top-down; -1 if never) and the cell width along dip. The geometry is what ``to_grdecl`` writes."""
    _, _, zc, act = _build_geometry(layers, **built["kwargs"])
    i = act.shape[0] // 2 if i is None else i
    column = act[i]                                                        # (ny, nz)
    interface = 0.25 * (zc[2 * i, 0::2] + zc[2 * i, 1::2] + zc[2 * i + 1, 0::2] + zc[2 * i + 1, 1::2])   # (ny, nz + 1)
    first = column.argmax(axis=1)
    depth = np.where(column.any(axis=1), interface[np.arange(len(first)), first], np.nan)
    starts = [int(np.argmax(column[:, k])) if column[:, k].any() else -1 for k in range(column.shape[1])]
    return column.sum(axis=1), depth, starts, layers[0].dy


@pytest.mark.parametrize("angle,width", [(0.25, 2290.0), (1.0, 573.0), (3.0, 191.0)])
def test_the_subcrop_strip_of_a_truncation_is_as_wide_as_the_thickness_over_the_tangent_of_the_discordance(angle,
                                                                                                           width):
    """Beds of 3.5 degrees dip cut by an erosion surface that dips less: for a 10 m sand the beds are cut away from
    the top over a strip of T / tan(discordance), the research note's 2.29 km at 0.25 degrees, 573 m at 1 and 191 m at
    3, measured on the exported grid as the strip where the top lies below the bed top. The discordance averaged over
    the strip is atan(T / width) = the drawn angle; the surface leaves the beds tangentially (T (1 - u)^2 below the
    bed top, u the share of the strip), so that there is no hinge, and cuts the base of the sand at twice the angle."""
    layers = _layers(4000.0, 6000.0, 25.0, [10.0], dz=1.0)
    built = strat_trap("truncation", 4000.0, 6000.0, 2000.0, [10.0], seed=1, dip=3.5, taper_angle=angle, area=None)
    count, top, _, dy = _profile(layers, built)
    nz, line = 10, built["meta"]["line"]
    rows = (np.arange(len(top)) + 0.5) * dy
    plane = 2000.0 + (rows - 3000.0) * np.tan(np.radians(3.5))   # the bed top
    below = np.nan_to_num(top - plane)                              # how far the top lies under the bed top
    strip = below > 1e-3
    assert strip.sum() * dy == pytest.approx(width, rel=0.05, abs=2 * dy)
    assert np.degrees(np.arctan(10.0 / (strip.sum() * dy))) == pytest.approx(angle, rel=0.06)
    assert np.allclose(top[count == nz][~strip[count == nz]], plane[count == nz][~strip[count == nz]], atol=1e-6)
    u = (rows - line) / built["meta"]["taper_m"]
    inside = (u > 0.0) & (u < 1.0) & (count > 0)
    assert np.allclose(below[inside], 10.0 * (1.0 - u[inside]) ** 2, atol=0.05 * 10.0 + dy * np.tan(np.radians(3.5)))
    slope = np.gradient(below, dy)                                # the surface's slope against the beds
    last = inside & (u > 0.85)
    assert np.abs(slope[last]).max() < 0.4 * np.tan(np.radians(angle)) + 1e-9


@pytest.mark.parametrize("kind", ["truncation", "onlap"])
def test_layers_end_against_the_surface_over_the_thickness_over_the_tangent_of_the_angle(kind):
    """With n layers of a stack of thickness T the strip is T / tan(angle) long and the thickness is T u (2 - u), u the
    share of the strip, so layer k (top-down) is first present a share 1 - sqrt(1 - k / n) of the way along it where
    the older surface cuts the layers from below (onlap: the top layer first, the oldest last), 1 - sqrt((k + 1) / n)
    where the erosion surface cuts them from above (truncation: the oldest layer reaches farthest updip), each to a
    column and a half."""
    layers = _layers(4000.0, 6000.0, 25.0, [10.0], dz=1.0)
    built = strat_trap(kind, 4000.0, 6000.0, 2000.0, [10.0], seed=1, dip=3.5, taper_angle=1.0, area=None)
    count, top, starts, dy = _profile(layers, built)
    line, strip = built["meta"]["line"], 10.0 / np.tan(np.radians(1.0))
    assert strip == pytest.approx(573.0, abs=1.0) and built["meta"]["taper_m"] == pytest.approx(strip)
    for k, row in enumerate(starts):
        share = 1.0 - np.sqrt(1.0 - k / 10.0) if kind == "onlap" else 1.0 - np.sqrt((k + 1) / 10.0)
        assert (row + 0.5) * dy == pytest.approx(line + share * strip, abs=1.5 * dy)
    assert np.all(np.diff(count[count > 0]) >= 0) and count.max() == 10     # layers only join as the strip deepens
    if kind == "onlap":                                                     # the younger layers keep the bed top
        whole = np.nonzero(count > 0)[0]
        rows = (whole + 0.5) * dy
        assert np.allclose(top[whole], 2000.0 + (rows - 3000.0) * np.tan(np.radians(3.5)), atol=1e-6)


@pytest.mark.parametrize("kind", ["pinchout", "truncation", "onlap"])
def test_a_tongue_closes_by_the_dip_times_its_length_whichever_surface_cuts_the_sand(kind):
    """Pinched out, truncated or onlapped, the tongue protruding updip from the sheet holds tan(dip) times its length
    (27 m at 1.1 degrees for 3.4 km2 and aspect 2.2); a truncation's top is the erosion surface, so its crest and
    spill lie T = 10 m deeper than the pinch-out's, the base of the sand at its edge, and its trap is larger than the
    tongue: where the sand is whole across the base of the tongue its top is the bed top, T above the surface at the
    tongue's edge, so the trap reaches downdip of the line by up to T / tan(dip) there."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    built = strat_trap(kind, x_len, y_len, 2000.0, [10.0], seed=1, dip=1.1, taper_angle=0.5, area=3.4e6, aspect=2.2,
                       warp=0.0)
    (trap,) = trap_report(_layers(x_len, y_len, dx, [10.0]), built)
    cell = np.tan(np.radians(1.1)) * dx
    assert trap["limited_by"] == "spill" and trap["height"] == pytest.approx(27.0, abs=cell + 0.5)
    assert trap["crest_depth"] == pytest.approx(built["meta"]["crest_nominal"], abs=cell)
    assert trap["spill_depth"] == pytest.approx(built["meta"]["spill_nominal"], abs=cell)
    assert built["meta"]["spill_nominal"] == pytest.approx(1980.9 + (10.0 if kind == "truncation" else 0.0), abs=0.1)
    fine = _fine_traps(built, 2000.0, x_len, y_len)[0]
    assert trap["area"] == pytest.approx(fine["area"], rel=0.06) and 3.4e6 * 0.97 < trap["area"]
    assert (trap["area"] > 3.4e6 * 1.1) == (kind == "truncation")


def test_truncation_needs_the_erosion_surface_to_dip_the_same_way_as_the_beds_and_no_barrier():
    for kw, message in ((dict(dip=0.4, taper_angle=0.5), "dip"), (dict(dip=2.0, barrier=True), "barrier")):
        with pytest.raises(ValueError, match=message):
            strat_trap("truncation", 8000.0, 6000.0, 2000.0, [10.0, 5.0], seed=1, **kw)


@pytest.mark.parametrize("kind", ["pinchout_nose", "truncation_nose"])
def test_an_updip_edge_across_a_nose_closes_by_the_height_of_the_nose(kind):
    """A combination trap: a nose (closure(): a lobate bump of the given area and relief) on a 1 degree ramp, the edge
    of the sand a straight line across its crest. The trap closes by the nose's relief (the oil spills along the
    line, out of the nose's footprint), which is more than the structure alone closes by: its own updip saddle is
    in the absent zone. The crest sits on the line and the trap lies downdip of it."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    built = strat_trap(kind, x_len, y_len, 2000.0, [10.0], seed=4, dip=1.0, taper_angle=0.5, nose=NOSE)
    layers = _layers(x_len, y_len, dx, [10.0])
    (trap,) = trap_report(layers, built)
    cell = np.tan(np.radians(1.0)) * dx
    assert built["meta"]["closure_nominal"] == 60.0
    assert trap["limited_by"] == "spill" and trap["height"] == pytest.approx(60.0, abs=cell + 0.03 * 60.0)
    assert trap["crest_depth"] == pytest.approx(built["meta"]["crest_nominal"], abs=2.0 * cell)
    assert trap["spill_depth"] == pytest.approx(built["meta"]["spill_nominal"], abs=2.0 * cell)
    rows = np.nonzero(trap["mask"].any(axis=0))[0]
    line = int(built["meta"]["line"] // dx)
    assert rows.min() in (line, line + 1) and trap["crest"][1] in (line, line + 1)    # downdip of the line, crest on it
    assert abs(trap["crest"][0] - x_len / dx / 2) <= 2                                # on the nose's axis
    base = built["kwargs"]["structure"]
    X, Y = np.meshgrid((np.arange(160) + 0.5) * dx, (np.arange(120) + 0.5) * dx, indexing="ij")
    alone = st.closure_stats(2000.0 + base(X, Y), dx, dx)                           # ramp and nose, no edge
    assert alone["height"] < trap["height"] - 10.0


@pytest.mark.parametrize("kw,message", [
    (dict(kind="pinchout_nose"), "nose"),
    (dict(kind="pinchout_nose", nose=dict(NOSE, height=400.0)), "335"),
    (dict(kind="truncation_nose", nose=NOSE, dip=0.4, taper_angle=0.5), "dip"),
    (dict(kind="pinchout_nose", nose=dict(NOSE, area=90.0e6)), "fit"),
    (dict(kind="pinchout_nose", nose=dict(NOSE, tilt=0.2)), "tilt"),
    (dict(kind="pinchout", nose=NOSE), "nose"),
])
def test_a_nose_belongs_to_the_combination_kinds_and_is_capped_at_the_largest_closure_found(kw, message):
    args = dict(x_len=8000.0, y_len=6000.0, top=2000.0, thicknesses=[10.0], seed=1, dip=1.0, taper_angle=0.3)
    with pytest.raises(ValueError, match=message):
        strat_trap(**{**args, **kw})


@pytest.mark.parametrize("kind,extra", [("pinchout", {}), ("lens", {}), ("pinchout_nose", dict(nose=NOSE))])
def test_lobes_that_warp_out_of_the_model_are_refused_and_the_rest_leave_a_trap_that_does_not_touch_its_edge(kind,
                                                                                                          extra):
    """A trap that reaches the edge of the model leaks over it, whatever it was meant to hold. Strongly warped lobes
    do (the warp displaces them along strike by a share of their width, more for a long, thin trap), so such a seed
    is refused, whenever the lobes come within 2 % of the edge; a seed that is accepted gives a trap that touches no
    edge."""
    x_len, y_len, dx = 8000.0, 6000.0, 100.0
    kw = dict(extra) if extra else dict(area=6.0e6, aspect=4.5, warp=0.4)
    if extra:
        kw["nose"] = dict(NOSE, warp=0.4, area=8.0e6, aspect=0.5)
    refused, accepted = 0, 0
    for seed in range(24):
        try:
            built = strat_trap(kind, x_len, y_len, 2000.0, [10.0], seed=seed, dip=1.0, taper_angle=0.5, **kw)
        except ValueError as error:
            assert "edge" in str(error)
            refused += 1
            continue
        accepted += 1
        edge = np.zeros((int(x_len / dx), int(y_len / dx)), dtype=bool)
        edge[[0, -1], :] = edge[:, [0, -1]] = True
        trap = trap_report(_layers(x_len, y_len, dx, [10.0], dz=5.0), built)[0]
        assert not (trap["mask"] & edge).any()
    assert refused >= 3 and accepted >= 3


def test_the_edge_and_the_top_have_no_octave_finer_than_two_cells_unless_the_caller_gives_a_floor():
    """``cell`` is the model's widest cell: the finest wavelength of the edge's relief and of the top's is two of them
    (200 m for cells of 100 m, over the 31 m and 62 m a 32nd of the ranges would give), a 32nd of the range where that
    is longer, and ``floor_m`` where the caller gives one. The surfaces are those of ``relief`` with that floor and
    the streams ``strat_trap`` draws them from."""
    x_len, y_len = 8000.0, 6000.0
    kw = dict(dip=1.0, taper_angle=0.5, area=None, wander=100.0, range_m=1000.0, relief_sd=1.5, relief_range=2000.0)
    ss = np.random.SeedSequence(3).spawn(5)
    x, y = np.meshgrid(np.linspace(0.0, x_len, 41), np.linspace(0.0, y_len, 31), indexing="ij")
    ramp = st.ramp(1.0, 0.0, center=(0.5 * x_len, 0.5 * y_len))
    for args, floor_edge, floor_top in (({"cell": 100.0}, 200.0, 200.0), ({"cell": 10.0}, 1000.0 / 32.0, 2000.0 / 32.0),
                                        ({"cell": 100.0, "floor_m": 50.0}, 50.0, 50.0), ({"floor_m": 300.0}, 300.0, 300.0)):
        built = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=3, **args, **kw)
        top = st.relief(1.5, 2000.0, x_len, y_len, 0.75, floor_top, seed=ss[2])
        assert np.allclose(built["kwargs"]["structure"](x, y), ramp(x, y) + top(x, y), atol=1e-9)
        assert built["meta"]["floor_m"] == floor_edge and built["meta"]["cell"] == args.get("cell")
        rough = st.relief(100.0, 1000.0, x_len, y_len, 0.75, floor_edge, seed=ss[1])
        meta = built["meta"]
        factor = st.taper(meta["line"], meta["taper_m"], 0.0, edge=rough)
        assert np.allclose(built["kwargs"]["isochore"][0](x, y), factor(x, y), atol=1e-12)
        assert not np.allclose(factor(x, y), st.taper(meta["line"], meta["taper_m"], 0.0)(x, y))   # the edge is rough


def test_a_relief_without_a_cell_or_a_floor_is_refused_and_so_is_a_range_below_two_cells():
    kw = dict(dip=1.0, taper_angle=0.5, area=None)
    with pytest.raises(ValueError, match="cell"):
        strat_trap("pinchout", 8000.0, 6000.0, 2000.0, [10.0], seed=1, wander=100.0, **kw)
    with pytest.raises(ValueError, match="cell"):
        strat_trap("pinchout", 8000.0, 6000.0, 2000.0, [10.0], seed=1, relief_sd=1.0, **kw)
    strat_trap("pinchout", 8000.0, 6000.0, 2000.0, [10.0], seed=1, **kw)              # none asked: no cell needed
    with pytest.raises(ValueError, match="range_m"):
        strat_trap("pinchout", 8000.0, 6000.0, 2000.0, [10.0], seed=1, wander=100.0, range_m=300.0, cell=200.0, **kw)
    with pytest.raises(ValueError, match="relief_range"):
        strat_trap("pinchout", 8000.0, 6000.0, 2000.0, [10.0], seed=1, relief_sd=1.0, relief_range=300.0, cell=200.0,
                   **kw)


# A draw of the plan's ranges (seed 1446768185, a 11.2 x 9 km model) whose sand edge, wandering 329 m, with a top
# relief of 1.9 m, shows a grid dependence when its octaves go down to 31 m: cells of 100 and 200 m read closures of 40
# and 54 m against 36 m on the fine map.
GRID_DRAW = dict(dip=1.153, taper_angle=0.2143, area=4848934.0, aspect=1.7735, warp=0.1699, wander=329.31,
                 relief_sd=1.9424)


def _grid_draw_closure(dx, **floor):
    built = strat_trap("pinchout", 11200.0, 9000.0, 2000.0, [8.0], seed=1446768185, **GRID_DRAW, **floor)
    fine = _fine_traps(built, 2000.0, 11200.0, 9000.0)[0]["height"]
    return trap_report(_layers(11200.0, 9000.0, dx, [8.0]), built)[0]["height"], fine, np.tan(np.radians(1.153)) * dx


@pytest.mark.parametrize("dx", [50.0, 100.0, 200.0])
def test_with_a_floor_of_two_cells_every_grid_up_to_that_cell_reads_the_closure_the_fine_map_has(dx):
    """One build, read on cells of 50, 100 and 200 m (the build's own cell): the closure is that of the fine analytic
    map within 1.5 cells' rise and 8 % of it. With the floor of 31 m that a 32nd of the range gives, the same draw
    reads 51 % more on the 200 m cells than the fine map: the sand's necks of under a cell open and shut."""
    height, fine, rise = _grid_draw_closure(dx, cell=200.0)
    assert height == pytest.approx(fine, abs=1.5 * rise + 0.08 * fine)
    if dx == 200.0:
        height_old, fine_old, _ = _grid_draw_closure(dx, floor_m=1000.0 / 32.0)
        assert height_old > 1.3 * fine_old


def test_the_largest_model_of_the_plan_builds_in_under_a_gigabyte():
    """The plan's largest tongue (40 km2 on 42.7 x 34.2 km) with an edge and a top with relief, on cells of 100 m: the
    octaves go down to 200 m, so the surfaces hold a few million points (22 s and 7 GB with a floor of 31 m)."""
    import time
    import tracemalloc
    tracemalloc.start()
    t0 = time.time()
    strat_trap("pinchout", 42700.0, 34160.0, 2000.0, [8.0], seed=1, dip=1.0, taper_angle=0.3, area=40e6, aspect=1.0,
               warp=0.3, wander=710.0, relief_sd=2.0, cell=100.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 0.5e9 and time.time() - t0 < 30.0


# Two draws of the plan's ranges (pinch-outs, cells of 100 m) whose wandered sand reaches the model's edge: the sand
# comes 2.9 and 1.7 km updip of the line against tongues of 1.8 and 1.2 km, the shallowest active column is on the
# map's boundary and the spill is the crest, 0 m against nominal closures of 39 and 34 m. The nominal outline, which
# the refusal used to read, is well inside the model.
EDGE_HITS = [
    dict(dip=1.255, taper_angle=0.328, area=10063799.096, aspect=3.938, warp=0.12, wander=319.971, relief_sd=0.11,
         azimuth=193.423, seed=1613526909, x_len=10800.0, y_len=8700.0),
    dict(dip=1.585, taper_angle=0.309, area=3161456.307, aspect=2.623, warp=0.364, wander=175.799, relief_sd=4.667,
         azimuth=352.799, seed=1511112119, x_len=8500.0, y_len=6800.0),
]


@pytest.mark.parametrize("draw", EDGE_HITS, ids=["crest on the edge", "crest at the edge"])
def test_a_draw_whose_wander_puts_the_crest_on_the_models_edge_is_refused(draw):
    """The refusal reads the sand that was drawn, wander and relief included, not the nominal outline: no pool of the plan
    grid can hold a fifth of the nominal closure (the edge's own crest is shallower than every pool's: no contact is
    admissible), so the caller is told to draw again."""
    draw = dict(draw)
    x_len, y_len, seed = draw.pop("x_len"), draw.pop("y_len"), draw.pop("seed")
    with pytest.raises(ValueError, match="wander"):
        strat_trap("pinchout", x_len, y_len, 2000.0, [8.0], seed, cell=100.0, **draw)
    smooth = dict(draw, wander=0.0, relief_sd=0.0)                       # the same shapes without the roughness
    built = strat_trap("pinchout", x_len, y_len, 2000.0, [8.0], seed, **smooth)
    assert trap_report(_layers(x_len, y_len, 100.0, [8.0], dz=2.0), built)[0]["height"] > 0.5 * built["meta"][
        "closure_nominal"]


@pytest.mark.parametrize("kind", ["pinchout", "truncation"])
def test_what_is_accepted_at_the_largest_wander_still_closes_and_what_is_not_is_refused(kind):
    """24 draws at a wander of 0.18 of the main tongue's length (the plan's range ends at 0.2): the oil that a contact can
    hold in the cells' main trap (``contact_limit``) is never under a tenth of the nominal closure (before, 3 % of the
    pinch-outs and 1 % of the truncations were accepted with about none), and some draws are refused for it."""
    x_len, y_len, dx = 9000.0, 7200.0, 100.0
    rng = np.random.default_rng(5)
    accepted, refused = [], 0
    for _ in range(24):
        area, aspect = float(rng.uniform(1.5e6, 4.0e6)), float(rng.uniform(1.0, 3.0))
        length = 2.0 * np.sqrt(area / (np.pi * aspect))
        taper = 0.4 if kind == "pinchout" else 0.3
        try:
            built = strat_trap(kind, x_len, y_len, 2000.0, [8.0], int(rng.integers(1, 2 ** 31 - 1)), dip=1.4,
                               taper_angle=taper, area=area, aspect=aspect, warp=0.25, wander=0.18 * length if kind ==
                               "pinchout" else 0.18 * 8.0 / np.tan(np.radians(taper)), relief_sd=1.0, cell=dx,
                               azimuth=float(rng.uniform(0.0, 360.0)))
        except ValueError as error:
            assert "wander" in str(error) or "edge" in str(error) or "fit" in str(error)
            refused += "wander" in str(error)
            continue
        trap = trap_report(_layers(x_len, y_len, dx, [8.0], dz=2.0), built)[0]
        accepted.append((trap["contact_limit"] - trap["crest_depth"]) / built["meta"]["closure_nominal"])
    assert len(accepted) >= 8 and refused >= 1 and min(accepted) > 0.1


@pytest.mark.parametrize("kind,share,fits", [
    ("pinchout", 0.99, True), ("pinchout", 1.02, False), ("truncation", 0.99, True), ("truncation", 1.02, False),
    ("onlap", 1.02, False), ("pinchout_nose", 0.99, True), ("pinchout_nose", 1.02, False),
    ("truncation_nose", 1.02, False)])
def test_the_wander_is_at_most_a_fifth_of_the_tongue_or_a_quarter_of_the_taper(kind, share, fits):
    """The owner's range: more breaks the sand into pieces that hold no trap. A depositional edge with a tongue may
    wander 0.2 of its length (``share`` is that of the limit); the edge of an erosion surface (truncation, onlap) or
    one across a nose, a quarter of the taper."""
    x_len, y_len, angle = 14000.0, 11000.0, 0.4
    taper_m = 8.0 / np.tan(np.radians(angle))
    length = 2.0 * np.sqrt(3.4e6 / (np.pi * 2.2))
    erosion = kind != "pinchout"
    args = dict(dip=1.2, taper_angle=angle, wander=share * (0.25 * taper_m if erosion else 0.2 * length), cell=100.0,
                **(dict(nose=NOSE) if kind.endswith("_nose") else dict(area=3.4e6, aspect=2.2)))
    if not fits:
        with pytest.raises(ValueError, match="at most"):
            strat_trap(kind, x_len, y_len, 2000.0, [8.0], 1, **args)
    else:
        for seed in range(1, 12):                          # a bound is not a guarantee: the collapse refusal may speak
            try:
                strat_trap(kind, x_len, y_len, 2000.0, [8.0], seed, **args)
                break
            except ValueError as error:
                assert "at most" not in str(error)
        else:
            raise AssertionError("every seed was refused")


@pytest.mark.parametrize("kw,message", [
    (dict(dip=-1.0), "dip"), (dict(dip=0.0), "dip"),
    (dict(thicknesses=[10.0, -2.0]), "thicknesses"), (dict(thicknesses=[0.0, 4.0]), "thicknesses"),
    (dict(area=-1.0e6), "area"), (dict(aspect=0.0), "aspect"), (dict(tongues=[(1.0e6, -1.0)]), "tongues"),
    (dict(tongues=[(0.0, 1.0)]), "tongues"), (dict(wander=-50.0, cell=100.0), "wander"),
    (dict(relief_sd=-1.0, cell=100.0), "relief_sd"), (dict(stagger=-100.0), "stagger"), (dict(warp=-0.1), "warp"),
    (dict(cell=0.0), "cell"), (dict(floor_m=-5.0, wander=10.0), "floor_m"), (dict(range_m=0.0, cell=100.0), "range_m"),
    (dict(nose=dict(area=6.0e6, aspect=0.7), kind="pinchout_nose", area=None), "height"),
])
def test_strat_trap_refuses_inputs_that_are_not_a_trap(kw, message):
    """A negative or zero dip, thickness, area or aspect, a negative wander, relief or stagger, a nose without its
    height: ValueError before anything is drawn (a dip of -1 degree returned a closure of -24 m and no trap)."""
    args = dict(kind="facies_change", x_len=8000.0, y_len=6000.0, top=2000.0, thicknesses=[10.0, 8.0], seed=1,
                barrier=True, column=12.0, dip=1.0, taper_angle=0.3, area=3.4e6, aspect=2.2)
    if kw.get("kind", "").endswith("_nose"):
        args["barrier"], args["column"], args["thicknesses"] = False, None, [10.0]
    with pytest.raises(ValueError, match=message):
        strat_trap(**{**args, **kw})


def test_a_report_of_layers_that_are_not_the_ones_the_trap_was_built_for_is_refused():
    """A truncation built for 10 m of sand and read on layers of 20 m had no closure, silently (26.9 m for the layers it
    was built for); the report compares the model's layers with the thicknesses, the size and the top that were
    drawn."""
    built = strat_trap("truncation", 8000.0, 6000.0, 2000.0, [10.0], seed=1, dip=1.1, taper_angle=0.5, area=3.4e6,
                       aspect=2.2, warp=0.0)
    assert trap_report(_layers(8000.0, 6000.0, 50.0, [10.0]), built)[0]["height"] == pytest.approx(27.0, abs=3.0)
    for layers in (_layers(8000.0, 6000.0, 50.0, [20.0]), _layers(8000.0, 6000.0, 50.0, [10.0], top=1900.0),
                   _layers(7000.0, 6000.0, 50.0, [10.0]), _layers(8000.0, 6000.0, 50.0, [5.0, 5.0])):
        with pytest.raises(ValueError, match="built for"):
            trap_report(layers, built)


@pytest.mark.parametrize("kind,kw", [("pinchout", dict(taper_angle=0.5)), ("truncation", dict(taper_angle=0.5, dip=1.2)),
                                     ("onlap", dict(taper_angle=0.5))])
def test_a_sand_of_several_layers_thins_over_its_whole_thickness_as_one_layer_of_that_thickness_does(kind, kw):
    """Three sand layers of 4 m are one sand of 12 m: the taper (``meta["taper_m"]``), the subcrop strip and the
    thickness of every column are those of a single layer of 12 m, not of the first layer's 4 m (a taper of 458 m
    where 1,375 m is drawn), whichever surface cuts the sand."""
    x_len, y_len, dx = 4000.0, 6000.0, 25.0
    args = dict(dict(dip=1.5, area=None), **kw)
    many = strat_trap(kind, x_len, y_len, 2000.0, [4.0, 4.0, 4.0], seed=1, **args)
    one = strat_trap(kind, x_len, y_len, 2000.0, [12.0], seed=1, **args)
    assert many["meta"]["thickness"] == 12.0 and many["meta"]["taper_m"] == pytest.approx(12.0 / np.tan(np.radians(0.5)))
    assert many["meta"]["taper_m"] == one["meta"]["taper_m"]
    zm = _build_geometry(_layers(x_len, y_len, dx, [4.0, 4.0, 4.0], dz=1.0), **many["kwargs"])[2]
    z1 = _build_geometry(_layers(x_len, y_len, dx, [12.0], dz=1.0), **one["kwargs"])[2]
    assert np.allclose(zm[:, :, -1] - zm[:, :, 0], z1[:, :, -1] - z1[:, :, 0], atol=1e-6)
    assert np.allclose(zm[:, :, 0], z1[:, :, 0], atol=1e-6)
