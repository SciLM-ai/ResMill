"""Stratigraphic traps: the trap of a zone that pinches out, Berg's barrier columns, and the trap builder."""
import numpy as np
import pytest
from scipy import ndimage

from resmill import structure as st
from resmill.export import _build_geometry
from resmill.layers.base import Layer
from resmill.strat_traps import (_traps, barrier_column, effective_grain_size, strat_trap, trap_report, zone_top,
                                 zone_trap)

FT = 0.3048


def _disc(cx, cy, radius):
    """An outline of a disc: negative inside, as ``closure`` is (a distance, so its footprint is exact)."""
    return st.Structure(lambda x, y: np.hypot(np.asarray(x) - cx, np.asarray(y) - cy) - radius)


def _layers(x_len, y_len, dx, thicknesses, top=2000.0, dz=2.0):
    """Flat layers, top to bottom, ``thicknesses`` (m) thick, of cells ``dx`` m wide and about ``dz`` m high."""
    layers, depth = [], top
    for t in thicknesses:
        layers.append(Layer(int(round(x_len / dx)), int(round(y_len / dx)), max(int(round(t / dz)), 1), x_len, y_len, t,
                            depth))
        depth += t
    return layers


def _length(area, aspect):
    """The research note's length along dip of an elliptical trap of ``area`` and ``aspect`` (strike over dip)."""
    return 2.0 * np.sqrt(area / (np.pi * aspect))


def _fine_traps(built, top, x_len, y_len, n=401):
    """The traps of the geometry a generator drew, on a fine analytic map of the top of its sand (n x n points): the
    structure and the thickness factor or erosion surface it returned, and the flood of the cells' report. It reads
    neither the corner-point grid nor which cells ACTNUM keeps, so it is what the cells' trap is checked against."""
    kw, thick = built["kwargs"], built["meta"]["thickness"]
    x, y = np.meshgrid(np.linspace(0.0, x_len, n), np.linspace(0.0, y_len, n), indexing="ij")
    bed = top + kw["structure"](x, y)                              # the top of the beds
    if "erode_above" in kw:                                        # truncation: the surface is the top, where it cuts
        depth, present = kw["erode_above"](x, y), kw["erode_above"](x, y) < bed + thick - 1e-9
    elif "erode_below" in kw:                                      # onlap: the layers follow the top, to the surface
        depth, present = bed, kw["erode_below"](x, y) > bed + 1e-9
    else:
        depth, present = bed, kw["isochore"][0](x, y) > 0.0
    return sorted(_traps(np.where(present, depth, np.inf), x_len / (n - 1), y_len / (n - 1)),
                  key=lambda trap: -trap["height"])


def _prototype():
    """The research note's prototype (step6_stratigraphic.md, section 3.3): a 20 m sand on a 1.1 degree ramp, 8 x 6 km
    of 50 m cells; the thickness factor is a lobate bump above the line y = 0.42 Y (a tongue protruding updip) and 1
    below it (a sheet that continues downdip to the model's edge). Returns the geometry's (zc, act), dx and Y."""
    nx, ny, dx = 160, 120, 50.0
    x_len, y_len = nx * dx, ny * dx
    layer = Layer(nx, ny, 20, x_len, y_len, 20.0, 2000.0)
    bump = st.closure(area=5.0e6, height=1.0, aspect=2.2, azimuth=0.0, center=(0.5 * x_len, 0.42 * y_len), warp=0.3,
                      seed=3)
    factor = st.Structure(lambda x, y: np.where(np.asarray(y) > 0.42 * y_len, 1.0,
                                                np.clip(-2.0 * bump(x, y), 0.0, 1.0)))
    _, _, zc, act = _build_geometry([layer], structure=st.ramp(1.1, azimuth=0.0), isochore=[factor])
    return zc, act, dx, y_len


def test_a_tongue_of_sand_protruding_updip_from_a_sheet_closes_by_the_dip_times_its_length():
    """The prototype of the research note (its own flood): crest 2033.1 m, spill 2048.5 m, closure 15.4 m over
    2.19 km2, where reading an absent neighbour as an exit gives 0 m. The spill is the depth of the sheet's updip
    edge, 2000 + 0.42 Y tan(1.1 degrees) = 2048.4 m, and the closure is the dip times the tongue's length."""
    zc, act, dx, y_len = _prototype()
    (trap,) = zone_trap(zc, act, dx, dx)
    assert trap["crest_depth"] == pytest.approx(2033.1, abs=0.05)
    assert trap["spill_depth"] == pytest.approx(2048.5, abs=0.05)
    assert trap["height"] == pytest.approx(15.4, abs=0.05) and trap["limited_by"] == "spill"
    assert trap["area"] == pytest.approx(2.19e6, rel=0.005)
    assert trap["spill_depth"] == pytest.approx(2000.0 + 0.42 * y_len * np.tan(np.radians(1.1)), abs=1.0)
    j = np.nonzero(act.any(axis=(0, 2)))[0]                           # rows of present columns, nearest the tip first
    tip, sheet = j.min(), int(0.42 * y_len / dx) + 1                  # the tongue's tip row, the sheet's first row
    assert trap["height"] == pytest.approx(np.tan(np.radians(1.1)) * (sheet - tip) * dx, abs=1.5)
    assert trap["mask"].sum() * dx * dx == pytest.approx(trap["area"]) and not (trap["mask"] & ~act.any(axis=2)).any()


def test_a_lens_of_sand_enclosed_by_absent_columns_is_sealed_and_fills_to_its_deepest_point():
    """Nothing reaches an enclosed lens, so it has no spill; it holds the dip times its length along dip, 2,000 m of
    disc at 1 degree = 34.9 m (to within a cell), over its whole footprint (the columns the sand touches)."""
    nx, ny, dx = 60, 40, 100.0
    layer = Layer(nx, ny, 4, nx * dx, ny * dx, 20.0, 1500.0)
    lens = st.taper(None, 300.0, outline=_disc(3000.0, 2000.0, 1000.0), x_len=nx * dx, y_len=ny * dx)
    _, _, zc, act = _build_geometry([layer], structure=st.ramp(1.0, azimuth=0.0), isochore=[lens])
    (trap,) = zone_trap(zc, act, dx, dx)
    assert trap["spill_depth"] is None and trap["spill_point"] is None and trap["limited_by"] == "sealed"
    assert trap["height"] == pytest.approx(np.tan(np.radians(1.0)) * 2000.0, abs=1.5 * np.tan(np.radians(1.0)) * dx)
    assert trap["limit_depth"] == pytest.approx(trap["crest_depth"] + trap["height"])
    assert trap["area"] == pytest.approx(np.pi * (1000.0 + 0.5 * dx) ** 2, rel=0.05)   # the columns the disc touches
    assert np.array_equal(trap["mask"], act.any(axis=2))              # the whole lens, its deepest columns included


def test_the_top_of_a_zone_is_the_first_active_cell_of_each_column_and_infinite_where_it_is_absent():
    """A plane of 1.1 degrees: the top of a cell is the plane at its centre, 1,000 m + y tan(1.1 degrees) at the
    middle of a cell row; the barrier zone's cells under the sand, active everywhere, are no part of the sand's top."""
    nx, ny, dx = 60, 40, 100.0
    sand, barrier = Layer(nx, ny, 2, nx * dx, ny * dx, 6.0, 1000.0), Layer(nx, ny, 2, nx * dx, ny * dx, 6.0, 1006.0)
    f = st.taper(None, 300.0, outline=_disc(3000.0, 2000.0, 1000.0), x_len=nx * dx, y_len=ny * dx)
    g = st.Structure(lambda x, y: 2.0 - f(x, y))
    _, _, zc, act = _build_geometry([sand, barrier], structure=st.ramp(1.1, azimuth=0.0), isochore=[f, g])
    top = zone_top(zc, act, slice(0, 2))
    present = act[:, :, :2].any(axis=2)
    assert np.isfinite(top).sum() == present.sum() and np.isinf(top[~present]).all() and present.any()
    rows = np.broadcast_to((np.arange(ny) + 0.5) * dx, top.shape)
    assert np.allclose(top[present], 1000.0 + rows[present] * np.tan(np.radians(1.1)), atol=1e-6)
    assert np.allclose(zone_top(zc, act), 1000.0 + rows * np.tan(np.radians(1.1)))   # the whole interval: a plane


def test_each_body_of_sand_has_its_own_trap_shallowest_crest_first_and_no_sand_has_none():
    nx, ny, dx = 60, 40, 100.0
    layer = Layer(nx, ny, 2, nx * dx, ny * dx, 10.0, 1500.0)
    two = st.taper(None, 200.0, outline=_disc(1500.0, 2000.0, 700.0), x_len=nx * dx, y_len=ny * dx) \
        + st.taper(None, 200.0, outline=_disc(4500.0, 1200.0, 500.0), x_len=nx * dx, y_len=ny * dx)
    _, _, zc, act = _build_geometry([layer], structure=st.ramp(1.0, azimuth=0.0), isochore=[two])
    first, second = zone_trap(zc, act, dx, dx)
    assert first["crest_depth"] < second["crest_depth"]
    assert first["crest"][0] > 40 > second["crest"][0]                # the lens on the right sits shallower
    assert not (first["mask"] & second["mask"]).any()
    assert (first["mask"] | second["mask"]).sum() == act.any(axis=2).sum()   # and together they are all the sand
    assert zone_trap(zc, np.zeros_like(act), dx, dx) == []


def test_a_zone_is_the_layers_it_is_asked_for_and_a_barrier_below_is_not_part_of_the_surface():
    """The sand zone above a barrier zone that takes the thickness the sand loses: the barrier's cells are active
    under the whole interval, so the top of the interval is a plane with no closure, but the sand's own top has the
    pinch-out's trap, as for the sand alone."""
    nx, ny, dx = 60, 40, 100.0
    sand, barrier = Layer(nx, ny, 4, nx * dx, ny * dx, 10.0, 1500.0), Layer(nx, ny, 4, nx * dx, ny * dx, 10.0, 1510.0)
    f = st.taper(None, 300.0, outline=_disc(3000.0, 2000.0, 1000.0), x_len=nx * dx, y_len=ny * dx)
    g = st.Structure(lambda x, y: 2.0 - f(x, y))                       # total thickness 20 m everywhere
    _, _, zc, act = _build_geometry([sand, barrier], structure=st.ramp(1.0, azimuth=0.0), isochore=[f, g])
    assert act.any(axis=2).all()                                   # the barrier is under every column
    (whole,) = zone_trap(zc, act, dx, dx)
    assert whole["spill_depth"] is not None and whole["height"] == pytest.approx(0.0, abs=1e-9)
    (net,) = zone_trap(zc, act, dx, dx, k=slice(0, 4))
    assert net["limited_by"] == "sealed" and net["height"] == pytest.approx(np.tan(np.radians(1.0)) * 2000.0, abs=1.8)


def test_a_barrier_column_below_the_spill_limits_the_trap_and_one_above_it_does_not():
    zc, act, dx, _ = _prototype()
    (free,) = zone_trap(zc, act, dx, dx)
    (held,) = zone_trap(zc, act, dx, dx, column=10.0)
    assert held["limited_by"] == "barrier" and held["height"] == pytest.approx(10.0)
    assert held["limit_depth"] == pytest.approx(free["crest_depth"] + 10.0)
    assert held["spill_depth"] == free["spill_depth"] and held["spill_point"] == free["spill_point"]
    assert 0.0 < held["area"] < free["area"]                          # the filled part of the closure is smaller
    assert not (held["mask"] & ~free["mask"]).any()
    (roomy,) = zone_trap(zc, act, dx, dx, column=100.0)
    assert roomy["limited_by"] == "spill" and roomy["height"] == free["height"]
    (shut,) = zone_trap(zc, act, dx, dx, column=0.0)                  # a barrier that holds nothing: no trap
    assert shut["height"] == 0.0 and shut["area"] == 0.0 and shut["limited_by"] == "barrier"
    with pytest.raises(ValueError, match="column"):
        zone_trap(zc, act, dx, dx, column=-1.0)


# Berg (1975, AAPG Bull. 59:939-956), Table 1: effective grain sizes D (m), fluids and the oil columns he calculated
# (ft). Milbur: reservoir 32 %, 900 mD; barrier 20 %, 25 mD, or 24 %, 153 mD (D 5.5e-5 m, which his equation takes to
# 35 ft, not the 53 ft of his table: left out). Lane: channel 21 %, 533 mD; J1 18 %, 65 mD; J2 24 %, 77 mD. Main Pass
# 35: reservoir 9.5e-5 m; barriers 26 %, 75 mD and 29 %, 170 mD. Paduca: 24 %, 25 mD against 20 %, 5 mD.
BERG_TABLE = [
    # field, reservoir D, barrier D, water - oil density (kg/m3), interfacial tension (N/m), calculated column (ft)
    ("Milbur, barrier 20 %, 25 mD", 6.0e-5, 3.5e-5, 90.0, 0.030, 64.0),
    ("Lane, J1", 13.5e-5, 6.9e-5, 250.0, 0.035, 14.0),
    ("Lane, J2", 13.5e-5, 3.6e-5, 250.0, 0.035, 30.0),
    ("Main Pass 35, upper barrier", 9.5e-5, 2.9e-5, 320.0, 0.035, 29.0),
    ("Main Pass 35, lower barrier", 9.5e-5, 3.3e-5, 320.0, 0.035, 25.0),
    ("Main Pass 35, gas", 9.5e-5, 2.9e-5, 920.0, 0.035, 10.0),
    ("Paduca", 2.1e-5, 1.5e-5, 440.0, 0.035, 34.0),
]


@pytest.mark.parametrize("name,d_res,d_bar,delta_rho,sigma,feet", BERG_TABLE, ids=[row[0] for row in BERG_TABLE])
def test_barrier_column_reproduces_the_columns_berg_calculated(name, d_res, d_bar, delta_rho, sigma, feet):
    """The column a finer barrier holds in a coarser reservoir, 2 sigma (1 / r_throat - 1 / r_pore) / (g delta_rho)
    with the rhombohedral packing's radii (0.077 D of the barrier, 0.207 D of the reservoir), against the columns of
    Berg's Table 1 (he rounds to the foot)."""
    assert barrier_column(delta_rho, d_res, d_bar, sigma) / FT == pytest.approx(feet, abs=1.0)


def test_barrier_column_reproduces_bergs_worked_examples():
    """Berg's text: a 0.2 mm reservoir sand with a 0.05 mm coarse-silt barrier holds 55 ft of low-gravity oil (density
    contrast 0.1 g/cm3, interfacial tension 35 dyn/cm) and about 5 ft of gas (1.0), a 0.01 mm fine silt 300 ft and
    30 ft; oil migrating up through the 0.2 mm sand needs a stringer of 300 cm, and 760 cm to pass into sand of half the
    grain size."""
    args = dict(sigma=0.035)
    assert barrier_column(100.0, 0.2e-3, 0.05e-3, **args) / FT == pytest.approx(55.0, abs=1.0)
    assert barrier_column(1000.0, 0.2e-3, 0.05e-3, **args) / FT == pytest.approx(5.0, abs=0.7)
    assert barrier_column(100.0, 0.2e-3, 0.01e-3, **args) / FT == pytest.approx(300.0, abs=3.0)
    assert barrier_column(1000.0, 0.2e-3, 0.01e-3, **args) / FT == pytest.approx(30.0, abs=0.5)
    assert barrier_column(100.0, 0.2e-3, 0.2e-3, **args) == pytest.approx(3.0, rel=0.05)
    assert barrier_column(100.0, 0.2e-3, 0.1e-3, **args) == pytest.approx(7.6, rel=0.01)


def test_barrier_column_scales_as_the_equation_says_and_is_never_negative():
    base = barrier_column(300.0, 1.0e-4, 3.0e-5)
    assert barrier_column(600.0, 1.0e-4, 3.0e-5) == pytest.approx(base / 2.0)           # 1 / (density contrast)
    assert barrier_column(300.0, 1.0e-4, 3.0e-5, sigma=0.060) == pytest.approx(2.0 * base)   # interfacial tension
    assert barrier_column(300.0, 1.0e-4, 1.0e-4) > 0.0                                  # throat against its pore
    assert barrier_column(300.0, 3.0e-5, 1.0e-4) == 0.0                                 # a barrier 3 times as coarse


@pytest.mark.parametrize("k,phi,d", [(900, 0.32, 6.0e-5), (533, 0.21, 13.5e-5), (65, 0.18, 6.9e-5),
                                     (77, 0.24, 3.6e-5), (75, 0.26, 2.9e-5), (170, 0.29, 3.3e-5),
                                     (25, 0.24, 2.1e-5), (5, 0.20, 1.5e-5)])
def test_effective_grain_size_is_berg_empirical_equation_of_permeability_and_porosity(k, phi, d):
    """D = (1.89 k n^-5.1)^0.5 cm (k in mD, the porosity n in percent), against the grain sizes of his Table 1 that
    follow from their own porosity and permeability (two entries of the Milbur barrier do not: 5 to 8 % off)."""
    assert effective_grain_size(k, phi) == pytest.approx(d, rel=0.03)


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
    R = L / 2, and the closure is T (1 + tan(dip) L / (4 T))^2 where that is more than tan(dip) L (a gentle dip, the
    mound's own thickness raising the crest) and tan(dip) L where it is not (the crest at the updip rim)."""
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
        cap = 10.0 * (1.0 + expected / 40.0) ** 2
        assert 0.97 * expected < fine["height"] < max(expected, cap) + 2.0
    assert trap["height"] == pytest.approx(fine["height"], abs=1.5 * cell + 0.02 * expected)
    assert trap["crest_depth"] == pytest.approx(fine["crest_depth"], abs=cell)
    assert area * 0.97 < trap["area"] < area * 1.10


def test_a_facies_change_is_the_same_trap_with_a_barrier_zone_taking_the_sand_s_thickness():
    """The barrier zone thickens as the sand thins, so the interval keeps its thickness where the sand is, and the sand's
    trap is that of the pinch-out of the same sand; the barrier is under all of it and a rim beyond (200 m in from the
    rough edge the interval is whole)."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    kw = dict(seed=5, dip=1.1, taper_angle=0.5, area=3.4e6, aspect=2.2, warp=0.3, wander=40.0, range_m=800.0, cell=dx)
    sand_only = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], **kw)
    both = strat_trap("facies_change", x_len, y_len, 2000.0, [10.0, 8.0], barrier=True, column=12.0, **kw)
    layers = _layers(x_len, y_len, dx, [10.0, 8.0])
    _, _, zc, act = _build_geometry(layers, **both["kwargs"])
    sand = ndimage.binary_erosion(act[:, :, :5].any(axis=2), iterations=4)   # 200 m in from the sand's rough edge
    thick = zc[:, :, -1] - zc[:, :, 0]
    assert np.allclose(thick[np.repeat(np.repeat(sand, 2, axis=0), 2, axis=1)], 18.0, atol=1e-6)
    assert act[:, :, 5:].any(axis=2)[act[:, :, :5].any(axis=2)].all()      # barrier under the sand
    (net,) = trap_report(layers, both)
    (alone,) = trap_report(layers[:1], sand_only)
    assert net["crest"] == alone["crest"] and net["spill_point"] == alone["spill_point"]
    for key in ("crest_depth", "spill_depth", "closure"):
        assert net[key] == pytest.approx(alone[key])
    assert both["meta"]["net_layers"] == 1 and both["meta"]["barrier"] is True and alone["barrier_top"] is None
    assert net["barrier_top"] < net["crest_depth"] and net["limited_by"] == "barrier"
    assert net["height"] == pytest.approx(net["barrier_top"] + 12.0 - net["crest_depth"])


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


NOSE = dict(area=6.0e6, height=60.0, aspect=0.7)       # 6 km2, 60 m of relief, longer along dip than across


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


# ---------------------------------------------------------------------------------------------------------------
# Irregular edges, several tongues, mounds, smooth tapers and relief (the owner's review of 2026-10-01)

def _sand(kind, built, nz=6, x_len=8000.0, y_len=6000.0, dx=25.0, thick=6.0, top=2000.0):
    """The exported geometry of a single sand layer: the columns that are sand, their centres and the interfaces."""
    layer = Layer(int(round(x_len / dx)), int(round(y_len / dx)), nz, x_len, y_len, thick, top)
    Xc, Yc, zc, act = _build_geometry([layer], **built["kwargs"])
    xc, yc = np.meshgrid((np.arange(layer.nx) + 0.5) * dx, (np.arange(layer.ny) + 0.5) * dx, indexing="ij")
    return act.any(axis=2), xc, yc, zc, layer


TONGUES = [(1.6e6, 1.0), (0.9e6, 0.5), (0.6e6, 0.4), (0.8e6, 0.45), (0.5e6, 0.35)]


@pytest.mark.parametrize("kind", ["pinchout", "truncation", "onlap"])
@pytest.mark.parametrize("n", [1, 3, 5])
def test_the_tongues_are_as_many_as_drawn_and_each_has_the_length_and_width_it_was_drawn_with(kind, n):
    """Every tongue is a separate lobe of sand updip of the line: counted on the exported grid (the connected bodies of
    active columns more than two cells updip of the line, azimuth 0) there are as many as were drawn (the main one and
    the ``tongues`` of the caller), they stand along the line in the order of their offsets, each reaches updip of the
    line by its drawn length (two cells and 6 %) and is as wide at its root as aspect times that."""
    x_len, y_len, dx = 8000.0, 6000.0, 25.0
    built = strat_trap(kind, x_len, y_len, 2000.0, [6.0], seed=4, dip=1.5, taper_angle=0.5, area=TONGUES[0][0],
                       aspect=TONGUES[0][1], tongues=TONGUES[1:n], warp=0.0)
    sand, xc, yc, _, _ = _sand(kind, built)
    line, drawn = built["meta"]["line"], built["meta"]["tongues"]
    labels, count = ndimage.label(sand & (yc < line - 2.0 * dx))
    assert len(drawn) == n and count == n
    found = sorted(range(1, n + 1), key=lambda c: xc[labels == c].mean())
    for tongue, c in zip(sorted(drawn, key=lambda d: d["offset"]), found):
        body = labels == c
        length = line - yc[body].min() + 0.5 * dx
        assert length == pytest.approx(tongue["length"], abs=2 * dx + 0.06 * tongue["length"])
        row = body[:, np.argmin(abs(yc[0] - (line - 3.0 * dx)))]                 # the row three cells updip
        width = row.sum() * dx
        assert width == pytest.approx(tongue["aspect"] * tongue["length"], rel=0.12, abs=2 * dx)
        assert xc[body].mean() == pytest.approx(0.5 * x_len + tongue["offset"], abs=0.12 * width + 2 * dx)


def test_tongues_that_do_not_fit_are_refused_and_a_lens_a_nose_or_a_straight_edge_take_none():
    args = (8000.0, 6000.0, 2000.0, [10.0], 1)
    with pytest.raises(ValueError, match="fit"):
        strat_trap("pinchout", *args, area=3.4e6, aspect=2.2, tongues=[(3.4e6, 2.2)] * 4)
    for kind, kw in (("lens", {}), ("pinchout", dict(area=None)), ("pinchout_nose", dict(area=None, nose=NOSE))):
        with pytest.raises(ValueError, match="tongue"):
            strat_trap(kind, *args, tongues=[(1.0e6, 1.0)], **kw)


@pytest.mark.parametrize("hurst", [0.2, 0.4, 0.6])
def test_the_edge_wanders_as_rough_as_the_hurst_exponent_drawn(hurst):
    """With no tongue the edge is the line moved by a ``relief`` surface, so the first active row of each column of the
    exported grid is a profile of that surface along the line: its structure function S(lag) = mean (y(x + lag) -
    y(x))^2 rises as lag^(2 hurst) over 100-1,200 m (the exponent within 0.4 of 2 hurst, over 6 seeds: the cells and
    the Gaussian covariance of the octaves flatten it at the high end) and its rms is the ``wander`` asked for (to
    35 %). With no wander the edge is straight."""
    x_len, y_len, dx = 8000.0, 6000.0, 25.0
    lags, slopes, rms = np.array([4, 6, 8, 12, 16, 24, 32, 48]), [], []
    for seed in range(6):
        built = strat_trap("pinchout", x_len, y_len, 2000.0, [4.0], seed=seed, dip=1.0, taper_angle=0.5, area=None,
                           wander=200.0, range_m=3000.0, hurst=hurst, floor_m=dx)
        sand, _, yc, _, _ = _sand("pinchout", built, nz=1, thick=4.0)
        edge = yc[0][sand.argmax(axis=1)]
        s = [np.mean((edge[lag:] - edge[:-lag]) ** 2) for lag in lags]
        slopes.append(np.polyfit(np.log(lags * dx), np.log(s), 1)[0])
        rms.append(edge.std())
    assert np.mean(slopes) == pytest.approx(2.0 * hurst, abs=0.4)
    assert np.mean(rms) == pytest.approx(200.0, rel=0.35)
    flat = strat_trap("pinchout", x_len, y_len, 2000.0, [4.0], seed=0, dip=1.0, taper_angle=0.5, area=None)
    sand, _, yc, _, _ = _sand("pinchout", flat, nz=1, thick=4.0)
    assert np.ptp(sand.argmax(axis=1)) == 0


def test_a_rougher_hurst_exponent_gives_a_rougher_edge_whatever_the_seed():
    """Over 6 seeds the structure function slope of the edge rises with the drawn Hurst exponent: 0.2 < 0.4 < 0.6 <
    0.8, each step."""
    x_len, y_len, dx = 8000.0, 6000.0, 25.0
    lags, mean = np.array([4, 8, 16, 32]), []
    for hurst in (0.2, 0.4, 0.6, 0.8):
        slopes = []
        for seed in range(6):
            built = strat_trap("pinchout", x_len, y_len, 2000.0, [4.0], seed=seed, dip=1.0, taper_angle=0.5, area=None,
                               wander=200.0, range_m=3000.0, hurst=hurst, floor_m=dx)
            sand, _, yc, _, _ = _sand("pinchout", built, nz=1, thick=4.0)
            edge = yc[0][sand.argmax(axis=1)]
            s = [np.mean((edge[lag:] - edge[:-lag]) ** 2) for lag in lags]
            slopes.append(np.polyfit(np.log(lags * dx), np.log(s), 1)[0])
        mean.append(np.mean(slopes))
    assert np.all(np.diff(mean) > 0.05)


def test_a_lens_is_a_mound_on_a_flat_base_and_the_fill_of_a_channel_hangs_from_a_flat_top():
    """By default the base of the sand is the plane of the beds, at every column of the exported grid (the cells of
    absent sand collapse on it), and the top is a mound: as thick as the sand in the middle (10 m, to 2 %) and concave
    down along the dip axis where it is thick. With ``mound=False`` the top is the plane and the base sags below it as
    a bowl, the lens hanging from a flat top as the fill of a channel does."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    layers = _layers(x_len, y_len, dx, [10.0], dz=1.0)
    for mound in (True, False):
        built = strat_trap("lens", x_len, y_len, 2000.0, [10.0], seed=2, dip=1.1, taper_angle=0.3, area=3.4e6,
                           aspect=2.2, warp=0.0, mound=mound)
        assert built["meta"]["mound"] is mound
        Xc, Yc, zc, act = _build_geometry(layers, **built["kwargs"])
        plane = 2000.0 + (Yc - 3000.0) * np.tan(np.radians(1.1))
        thick = zc[:, :, -1] - zc[:, :, 0]
        centre = thick[len(thick) // 2, 0::2]                            # along dip through the middle, at nodes
        inner = (centre[1:-1] > 0.3 * 10.0) & (centre[:-2] > 0.3 * 10.0) & (centre[2:] > 0.3 * 10.0)
        if mound:
            assert np.allclose(zc[:, :, -1], plane + 10.0, atol=1e-6)
            assert thick.max() == pytest.approx(10.0, rel=0.02)
            assert (np.diff(centre, 2)[inner] < 1e-9).all() and inner.sum() > 5
        else:
            assert np.allclose(zc[:, :, 0], plane, atol=1e-6)
            assert 0.0 < thick.max() < 10.0 and (np.diff(centre, 2)[inner] < 1e-9).all()
            assert np.allclose(zc[:, :, -1] - plane, thick, atol=1e-6)         # the bowl is what the sand fills


def test_a_barrier_zone_under_a_mound_stays_a_flat_slab_and_under_a_hung_lens_takes_the_sand_s_thickness():
    """The barrier is the last zone: under a mound it keeps its thickness (8 m) and its base is the plane of the beds, 8 m
    under the mound's base; with the lens hung from a flat top it takes what the sand loses, so that the interval keeps
    its 10 + 8 m: both where the sand is (two cells in from its edge: the barrier is a rim round it, thinning to nothing
    beyond)."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    layers = _layers(x_len, y_len, dx, [10.0, 8.0], dz=2.0)
    for mound in (True, False):
        built = strat_trap("lens", x_len, y_len, 2000.0, [10.0, 8.0], seed=2, dip=1.1, taper_angle=0.3, area=3.4e6,
                           aspect=2.2, warp=0.0, barrier=True, column=12.0, mound=mound)
        Xc, Yc, zc, act = _build_geometry(layers, **built["kwargs"])
        plane = 2000.0 + (Yc - 3000.0) * np.tan(np.radians(1.1))
        sand = act[:, :, :5].any(axis=2)
        assert act[:, :, 5:].any(axis=2)[sand].all() and (act[:, :, 5:].any(axis=2) & ~sand).any()
        inside = np.repeat(np.repeat(ndimage.binary_erosion(sand, iterations=2), 2, axis=0), 2, axis=1)
        if mound:
            assert np.allclose((zc[:, :, -1] - zc[:, :, 5])[inside], 8.0, atol=1e-6)
            assert np.allclose((zc[:, :, -1] - plane)[inside], 18.0, atol=1e-6)
        else:
            assert np.allclose((zc[:, :, -1] - zc[:, :, 0])[inside], 18.0, atol=1e-6)


def test_the_taper_thins_without_a_step_or_a_hinge_to_the_edge():
    """Across a straight edge the sand's thickness along dip, read at the nodes of the exported grid, rises from 0 to
    its full 10 m over T / tan(angle) = 1,146 m (the wedge's width, to a node), its mean slope the drawn angle; it
    leaves full thickness with no slope (under 10 % of the mean there, where a linear ramp has all of it) and ends at
    the edge at twice the mean."""
    x_len, y_len, dx = 4000.0, 6000.0, 25.0
    angle = 0.5
    built = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=1, dip=1.0, taper_angle=angle, area=None)
    _, _, _, zc, _ = _sand("pinchout", built, nz=10, x_len=x_len, y_len=y_len, thick=10.0)
    h = (zc[:, :, -1] - zc[:, :, 0])[len(zc) // 4, 0::2]                     # along dip, one column, at the nodes
    mean = np.tan(np.radians(angle))
    taper = (h > 1e-9) & (h < 10.0 - 1e-6)
    assert taper.sum() * dx == pytest.approx(10.0 / mean, abs=2 * dx)
    slope = np.diff(h) / dx
    first, last = np.nonzero(taper)[0][[0, -1]]
    assert slope[first] == pytest.approx(2.0 * mean, rel=0.1)
    assert slope[last] < 0.1 * mean and np.all(np.diff(slope[first:last + 1]) <= 1e-12)
    assert (h[last + 1] - h[first - 1]) / ((last + 2 - first) * dx) == pytest.approx(mean, rel=0.05)


def test_relief_makes_the_top_and_the_base_of_the_zone_uneven_together_and_none_leaves_them_planes():
    """``relief_sd`` is the rms of the relief of the zone's top, 2 m here (to 40 %: one surface of a few correlation
    lengths), and the base follows it, the full sand being 10 m thick still; with none the top is exactly the plane."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    flat = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=3, dip=1.0, taper_angle=0.5, area=None)
    rough = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=3, dip=1.0, taper_angle=0.5, area=None,
                       relief_sd=2.0, relief_range=2000.0, cell=dx)
    layers = _layers(x_len, y_len, dx, [10.0], dz=1.0)
    Xc, Yc, zc0, _ = _build_geometry(layers, **flat["kwargs"])
    _, _, zc, act = _build_geometry(layers, **rough["kwargs"])
    plane = 2000.0 + (Yc - 3000.0) * np.tan(np.radians(1.0))
    sheet = Yc > 4200.0                                                    # full sand, far from the line
    top, base = zc[:, :, 0][sheet] - plane[sheet], zc[:, :, -1][sheet] - plane[sheet] - 10.0
    assert np.allclose(zc0[:, :, 0][sheet], plane[sheet], atol=1e-9)
    assert top.std() == pytest.approx(2.0, rel=0.4) and top.std() > 0.5
    assert np.allclose(base, top, atol=1e-9)                              # the zone keeps its thickness
    assert np.allclose(zc[:, :, -1][sheet] - zc[:, :, 0][sheet], 10.0, atol=1e-9)


def _same_fields(a, b):
    """Two builds that hold the same fields: the structure, the surfaces and every isochore agree on a grid."""
    x, y = np.meshgrid(np.linspace(0.0, 8000.0, 41), np.linspace(0.0, 6000.0, 31), indexing="ij")
    keys = [k for k in a["kwargs"] if k != "isochore"]
    return a["meta"] == b["meta"] and all(np.array_equal(a["kwargs"][k](x, y), b["kwargs"][k](x, y)) for k in keys) \
        and all(np.array_equal(f(x, y), g(x, y)) for f, g in zip(a["kwargs"].get("isochore", []),
                                                                  b["kwargs"].get("isochore", [])))


@pytest.mark.parametrize("kind", ["pinchout", "facies_change", "lens", "truncation", "onlap"])
def test_everything_drawn_follows_the_seed(kind):
    """The same seed gives the same tongues, edge and relief, to the bit, and another seed another one (the seeds whose
    lobes fit and whose edge leaves a trap)."""
    thick, barrier = ([10.0, 6.0], True) if kind == "facies_change" else ([10.0], False)
    kw = dict(barrier=barrier, column=15.0 if barrier else None, dip=1.4, taper_angle=0.4, area=2.0e6, aspect=1.5,
              warp=0.2, wander=120.0, range_m=1200.0, hurst=0.5, floor_m=60.0, relief_sd=1.5,
              **({} if kind == "lens" else dict(tongues=[(0.8e6, 0.5)])))
    builds = []
    for seed in range(11, 60):
        try:
            builds.append(strat_trap(kind, 8000.0, 6000.0, 2000.0, thick, seed=seed, **kw))
        except ValueError:
            continue
        if len(builds) == 2:
            break
    a, c = builds
    b = strat_trap(kind, 8000.0, 6000.0, 2000.0, thick, seed=a["meta"]["seed"], **kw)
    assert _same_fields(a, b) and not _same_fields(a, c)


def test_a_facies_change_with_tongues_a_rough_edge_and_relief_keeps_the_thickness_of_its_interval_where_the_sand_is():
    """The barrier zone is the complement of the sand wherever the sand is, whatever the edge: the interval is 10 + 8 m
    thick there, the barrier active under it and a rim beyond, and the sand has gone where the barrier alone is."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    built = strat_trap("facies_change", x_len, y_len, 2000.0, [10.0, 8.0], seed=5, barrier=True, column=15.0, dip=1.1,
                       taper_angle=0.5, area=2.0e6, aspect=1.5, tongues=[(0.8e6, 0.5)], warp=0.2, wander=150.0,
                       range_m=1200.0, hurst=0.5, floor_m=100.0, relief_sd=1.5)
    layers = _layers(x_len, y_len, dx, [10.0, 8.0])
    _, _, zc, act = _build_geometry(layers, **built["kwargs"])
    sand = act[:, :, :5].any(axis=2)
    inside = np.repeat(np.repeat(ndimage.binary_erosion(sand, iterations=10), 2, axis=0), 2, axis=1)   # 500 m in
    assert np.allclose((zc[:, :, -1] - zc[:, :, 0])[inside], 18.0, atol=1e-6) and inside.any()
    assert act[:, :, 5:].any(axis=2)[sand].all() and (act[:, :, 5:].any(axis=2) & ~sand).any() and sand.any()
    assert built["meta"]["net_layers"] == 1 and len(built["meta"]["tongues"]) == 2


def test_the_erosion_surface_of_a_truncation_has_the_relief_the_wander_maps_to():
    """A truncation's edge is where the beds meet an erosion surface with a relief of its own: the lateral wander of
    the edge R is the vertical relief tan(discordance) R of the surface (``erosion_relief_m``: 0.9 m for a wander of
    300 m at 0.3 degrees), so that the surface moves from the smooth one by that much in the strip (twice it at the
    edge, nothing where it meets the beds: the rms over the strip is within a factor 2 of it), and the sand reaches
    updip in the valleys and is cut back on the ridges by R."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    kw = dict(dip=1.2, taper_angle=0.3, area=None, range_m=1500.0, hurst=0.5, floor_m=100.0)
    smooth = strat_trap("truncation", x_len, y_len, 2000.0, [10.0], seed=7, **kw)
    rough = strat_trap("truncation", x_len, y_len, 2000.0, [10.0], seed=7, wander=300.0, **kw)
    assert smooth["meta"]["erosion_relief_m"] == 0.0
    assert strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=7, area=None, wander=300.0, cell=dx)["meta"][
        "erosion_relief_m"] is None
    assert rough["meta"]["erosion_relief_m"] == pytest.approx(300.0 * np.tan(np.radians(0.3)))
    x, y = np.meshgrid((np.arange(160) + 0.5) * dx, (np.arange(120) + 0.5) * dx, indexing="ij")
    e0, e1 = smooth["kwargs"]["erode_above"](x, y), rough["kwargs"]["erode_above"](x, y)
    bed = 2000.0 + rough["kwargs"]["structure"](x, y)
    strip = (e1 > bed + 1e-3) & (e1 < bed + 10.0 - 1e-3)
    assert strip.sum() > 1000
    assert np.std((e1 - e0)[strip]) == pytest.approx(rough["meta"]["erosion_relief_m"], rel=1.0)
    relief = rough["meta"]["erosion_relief_m"]
    assert (e1 - e0).max() > relief and (e1 - e0).min() < -relief
    first = [np.argmax(e1[i] < bed[i] + 10.0 - 1e-3) for i in range(160)]      # the edge, row by row
    assert np.ptp(first) * dx > 300.0                                       # it wanders by hundreds of metres


@pytest.mark.parametrize("kind", ["pinchout", "truncation", "onlap"])
def test_rough_edges_with_tongues_close_as_the_fine_map_of_their_geometry_says(kind):
    """With tongues, a rough edge and relief the closure is whatever the geometry has, so it is checked against the
    fine analytic map of the top (no cells, no ACTNUM): over 6 seeds the closure of the cells is within 20 % of it
    in the median, and the largest trap of the fine map is never more than 40 m off (the cells' own thread of
    columns can join or split bodies that the fine map keeps apart). Edges to the cell's own scale are not asked of
    it: ``floor_m`` is two cells."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    layers = _layers(x_len, y_len, dx, [10.0], dz=2.0)
    errors = []
    for seed in range(20, 60):
        try:
            built = strat_trap(kind, x_len, y_len, 2000.0, [10.0], seed=seed, dip=1.2, taper_angle=0.3, area=2.0e6,
                               aspect=1.2, tongues=[(1.0e6, 0.5), (0.7e6, 0.4)], warp=0.15, wander=120.0,
                               range_m=1200.0, hurst=0.6, floor_m=2 * dx, relief_sd=1.0)
        except ValueError:
            continue
        trap = trap_report(layers, built)[0]
        fine = _fine_traps(built, 2000.0, x_len, y_len)[0]
        errors.append((abs(trap["height"] - fine["height"]), fine["height"]))
        if len(errors) == 6:
            break
    assert len(errors) == 6
    assert np.median([e / h for e, h in errors]) < 0.2 and max(e for e, _ in errors) < 40.0


def test_the_report_gives_the_trap_with_the_most_closure_first():
    """Two lenses of sand with no sand between them: the report lists the larger closure first (2,400 m of lens along
    dip against 800 m: 42 m against 14), though the other's crest is shallower; the cells' own report keeps the
    order of the crests."""
    nx, ny, dx = 60, 60, 100.0
    layer = Layer(nx, ny, 2, nx * dx, ny * dx, 10.0, 1500.0)
    big = st.taper(None, 200.0, outline=_disc(3000.0, 3600.0, 1200.0), x_len=nx * dx, y_len=ny * dx)
    small = st.taper(None, 100.0, outline=_disc(4500.0, 1200.0, 400.0), x_len=nx * dx, y_len=ny * dx)
    built = dict(kwargs=dict(structure=st.ramp(1.0, azimuth=0.0), isochore=[big + small]),
                 meta=dict(net_layers=1, barrier=False, column=None, x_len=nx * dx, y_len=ny * dx, top=1500.0,
                           thicknesses=[10.0]))
    first, second = trap_report([layer], built)
    assert first["height"] == pytest.approx(42.0, abs=2.5) and second["height"] == pytest.approx(14.0, abs=2.5)
    assert first["crest_depth"] > second["crest_depth"]
    _, _, zc, act = _build_geometry([layer], **built["kwargs"])
    assert [trap["crest_depth"] for trap in zone_trap(zc, act, dx, dx)] == [second["crest_depth"], first["crest_depth"]]


@pytest.mark.parametrize("kind,azimuth", [("pinchout", 0.0), ("pinchout", 90.0), ("facies_change", 0.0)])
def test_staggered_layers_end_each_farther_downdip_by_the_stagger_and_the_interval_keeps_its_thickness(kind, azimuth):
    """Sand layers that end at different places make the pinch-out interfinger in section: with three layers of 4 m and
    a stagger of 300 m the second layer first appears 300 m farther downdip than the first, and the third 600 m (to a
    cell and a half), along the line and on every tongue; with a barrier zone under them (a facies change) the
    interval is still 12 + 6 m thick where the sand is, the barrier taking what the three layers lose. No stagger
    leaves the layers ending together."""
    x_len, y_len, dx = 8000.0, 6000.0, 25.0
    barrier = kind == "facies_change"
    thick = [4.0, 4.0, 4.0] + ([6.0] if barrier else [])
    layers = _layers(x_len, y_len, dx, thick, dz=1.0)
    first = {}
    for stagger in (0.0, 300.0):
        built = strat_trap(kind, x_len, y_len, 2000.0, thick, seed=2, barrier=barrier, column=12.0 if barrier else None,
                           dip=1.5, taper_angle=0.5, area=1.5e6, aspect=1.0, warp=0.0, azimuth=azimuth,
                           stagger=stagger)
        _, _, zc, act = _build_geometry(layers, **built["kwargs"])
        # along dip the first active row of each layer in the middle column of the tongue (azimuth 0), or the middle row
        if azimuth == 0.0:
            column = act[int(round(0.5 * x_len / dx)) + int(round(built["meta"]["tongues"][0]["offset"] / dx))]
            first[stagger] = [int(np.argmax(column[:, 4 * k:4 * k + 4].any(axis=1))) * dx for k in range(3)]
        else:                                                           # dip along +x: rows are x
            row = act[:, int(round(0.5 * y_len / dx)) - int(round(built["meta"]["tongues"][0]["offset"] / dx))]
            first[stagger] = [int(np.argmax(row[:, 4 * k:4 * k + 4].any(axis=1))) * dx for k in range(3)]
        if barrier:
            sand = act[:, :, :12].any(axis=2)
            inside = np.repeat(np.repeat(ndimage.binary_erosion(sand, iterations=2), 2, axis=0), 2, axis=1)
            assert np.allclose((zc[:, :, -1] - zc[:, :, 0])[inside], 18.0, atol=1e-6)
            assert act[:, :, 12:].any(axis=2)[sand].all()
    assert np.ptp(first[0.0]) <= 1.5 * dx
    assert np.diff(first[300.0]) == pytest.approx([300.0, 300.0], abs=1.5 * dx)


def test_staggered_layers_end_farther_downdip_along_a_straight_line_too():
    x_len, y_len, dx = 8000.0, 6000.0, 25.0
    layers = _layers(x_len, y_len, dx, [4.0, 4.0], dz=1.0)
    built = strat_trap("pinchout", x_len, y_len, 2000.0, [4.0, 4.0], seed=2, dip=1.5, taper_angle=0.5, area=None,
                       stagger=250.0)
    _, _, _, act = _build_geometry(layers, **built["kwargs"])
    first = [int(np.argmax(act[100, :, 4 * k:4 * k + 4].any(axis=1))) * dx for k in range(2)]
    assert first[1] - first[0] == pytest.approx(250.0, abs=1.5 * dx)
    assert built["meta"]["tongues"] == [] and built["meta"]["stagger"] == 250.0


def test_a_stagger_belongs_to_the_depositional_edges_with_a_sand_of_more_than_one_layer():
    args = (8000.0, 6000.0, 2000.0, [4.0, 4.0], 1)
    for kind, kw in (("truncation", dict(dip=1.0, taper_angle=0.5)), ("onlap", {}), ("lens", {}),
                     ("pinchout_nose", dict(area=None, nose=NOSE))):
        with pytest.raises(ValueError, match="stagger"):
            strat_trap(kind, *args, stagger=100.0, **kw)
    with pytest.raises(ValueError, match="stagger"):
        strat_trap("pinchout", 8000.0, 6000.0, 2000.0, [8.0], 1, stagger=100.0)


# ---------------------------------------------------------------------------------------------------------------
# The grid (review F3): no octave of an edge or a top finer than two cells

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


# ---------------------------------------------------------------------------------------------------------------
# The edge wanders: what it may do to the trap (review F4)

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


# ---------------------------------------------------------------------------------------------------------------
# Inputs that are not a trap (review F10)

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


# ---------------------------------------------------------------------------------------------------------------
# The barrier is a rim round the sand, and the limit the report gives keeps it dry (review F1)

COLUMN = 12.0                                     # m: the oil column the barrier holds (Berg), a typical 8-20 m


def _barrier_model(kind="facies_change", mound=None, dx=50.0, thick=(10.0, 8.0), **kw):
    """A facies change (or a lens in a barrier zone) of the P50 trap on 8 x 6 km: the built geometry and its layers."""
    x_len, y_len = 8000.0, 6000.0
    args = dict(seed=5, barrier=True, column=COLUMN, dip=1.0, taper_angle=0.5, area=3.4e6, aspect=2.2, warp=0.0)
    args.update(kw)
    if mound is not None:
        args["mound"] = mound
    built = strat_trap(kind, x_len, y_len, 2000.0, list(thick), **args)
    layers = _layers(x_len, y_len, dx, list(thick))
    _, _, zc, act = _build_geometry(layers, **built["kwargs"])
    return built, layers, zc, act


@pytest.mark.parametrize("kind,mound", [("facies_change", None), ("facies_change", True), ("lens", True), ("lens", False)])
def test_the_barrier_is_a_rim_a_quarter_of_its_column_over_the_dip_wide_round_the_sand_and_walls_beyond(kind, mound):
    """The barrier zone lies under every column of sand and out to ``rim`` beyond its edge, a quarter [J] of the most
    that leaves the oil a column: column / tan(dip) is 687 m for 12 m at 1 degree, so the rim is 172 m, to a cell. Beyond
    it the zone has thinned to nothing: columns with no active cell, walls. Before, it was a slab under the whole
    model, updip of the trap too, and an initialisation by contacts put 84-100 % of the oil in it."""
    dx = 50.0
    built, layers, zc, act = _barrier_model(kind, mound, dx=dx)
    rim = 0.25 * COLUMN / np.tan(np.radians(1.0))
    assert built["meta"]["rim_m"] == pytest.approx(rim) and built["meta"]["column"] == COLUMN
    n_sand = layers[0].nz
    sand, barrier = act[:, :, :n_sand].any(axis=2), act[:, :, n_sand:].any(axis=2)
    assert (barrier | ~sand).all() and sand.any()                          # the barrier is under all the sand
    away = ndimage.distance_transform_edt(~sand) * dx                       # m from the sand, over the barrier's columns
    assert away[barrier].max() == pytest.approx(rim, abs=2.0 * dx) and away[barrier & ~sand].min() > 0.0
    assert (~act.any(axis=2)).sum() > 0.1 * sand.size and not barrier[away > rim + 2.0 * dx].any()   # walls beyond


def test_a_barrier_of_a_facies_change_takes_the_thickness_the_sand_loses_inside_the_rim():
    """The interval keeps its 10 + 8 m where the sand is (two cells in from its edge, where the rim's factor is 1); the
    rim thins to nothing over its width, and beyond it the interval has no thickness at all."""
    built, layers, zc, act = _barrier_model()
    thick = zc[:, :, -1] - zc[:, :, 0]
    sand = ndimage.binary_erosion(act[:, :, :5].any(axis=2), iterations=2)
    inside = np.repeat(np.repeat(sand, 2, axis=0), 2, axis=1)                # the doubled corner grid
    assert np.allclose(thick[inside], 18.0, atol=1e-6) and thick[~inside].max() <= 18.0 + 1e-6
    assert (thick[~inside] > 0.0).any() and (thick[~inside] == 0.0).any() and (thick[~inside] < 17.0).any()


def _shallowest(zc, act, k):
    """The top of the shallowest active cell of layers ``k`` (the mean of its four corners), from the geometry alone."""
    tops = 0.25 * (zc[0::2, 0::2, :-1] + zc[1::2, 0::2, :-1] + zc[0::2, 1::2, :-1] + zc[1::2, 1::2, :-1])
    return tops[:, :, k][act[:, :, k] > 0].min()


@pytest.mark.parametrize("kind,mound", [("facies_change", None), ("facies_change", True), ("lens", True), ("lens", False)])
def test_the_report_gives_a_limit_that_keeps_the_barrier_dry_and_the_rim_leaves_most_of_the_column(kind, mound):
    """The limit is the smaller of the spill and the shallowest top of the barrier plus its column (below the crest's
    own reach if the barrier's top is shallower than the crest), so that a contact (the sand's oil-water contact) at the
    limit puts oil in no barrier cell: the barrier's cells lie at or below contact - column, to its shallowest top,
    which the test finds from the geometry alone. The crest is no more than a quarter of the column (and a cell's rise)
    below the barrier's top: three quarters of the column are left."""
    dx = 50.0
    built, layers, zc, act = _barrier_model(kind, mound, dx=dx)
    (trap,) = trap_report(layers, built)
    n_sand = layers[0].nz
    top = _shallowest(zc, act, slice(n_sand, None))
    assert trap["barrier_top"] == pytest.approx(top)
    reach = min(trap["crest_depth"], top)
    assert trap["limit_depth"] == pytest.approx(min(reach + COLUMN, trap["spill_depth"] or np.inf)
                                                if trap["spill_depth"] is not None else reach + COLUMN)
    assert trap["limited_by"] == "barrier" and trap["height"] > 0.5 * COLUMN
    cell = np.tan(np.radians(1.0)) * dx
    assert trap["crest_depth"] - top <= 0.25 * COLUMN + 2.0 * cell
    wet = lambda contact: (_shallowest(zc, act, slice(n_sand, None)) < contact - COLUMN)   # the shallowest barrier cell
    assert not wet(trap["limit_depth"] - 1e-6) and wet(trap["limit_depth"] + 1.0)


def test_a_barrier_needs_the_column_it_holds_and_a_column_needs_a_barrier():
    args = dict(x_len=8000.0, y_len=6000.0, top=2000.0, seed=1, dip=1.0, taper_angle=0.5)
    with pytest.raises(ValueError, match="column"):
        strat_trap("facies_change", thicknesses=[10.0, 8.0], barrier=True, **args)
    with pytest.raises(ValueError, match="column"):
        strat_trap("pinchout", thicknesses=[10.0], column=10.0, **args)
    with pytest.raises(ValueError, match="column"):
        strat_trap("facies_change", thicknesses=[10.0, 8.0], barrier=True, column=0.0, **args)


# ---------------------------------------------------------------------------------------------------------------
# The top of a tongue: convex up on a flat base or hung from the bedding plane (the owner's ruling of 2026-10-02)

@pytest.mark.parametrize("kind", ["pinchout", "facies_change"])
def test_a_tongue_has_a_convex_up_top_on_a_flat_base_when_asked_and_hangs_from_the_bedding_plane_by_default(kind):
    """The geometry of the lens's mound, for the tongues of a pinch-out or a facies change: with ``mound=True`` the base of
    the sand is the plane of the beds 10 m down at every column, and the top sinks by T (1 - f) toward the edge (the
    thickness factor f), so it is deepest at the tip; by default, and with ``mound=False``, the top is the plane of the
    beds and the base is a bowl. With a barrier zone the interval keeps its 10 + 8 m under the bowl and the barrier is
    a flat slab of 8 m under the mound."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    thick, barrier = ([10.0, 8.0], True) if kind == "facies_change" else ([10.0], False)
    layers = _layers(x_len, y_len, dx, thick, dz=1.0)
    for mound in (None, False, True):
        built = strat_trap(kind, x_len, y_len, 2000.0, thick, seed=2, barrier=barrier, column=12.0 if barrier else None,
                           dip=1.1, taper_angle=0.4, area=3.4e6, aspect=2.2, warp=0.0, mound=mound)
        assert built["meta"]["mound"] is bool(mound)
        Xc, Yc, zc, act = _build_geometry(layers, **built["kwargs"])
        plane = 2000.0 + (Yc - 3000.0) * np.tan(np.radians(1.1))
        f = built["kwargs"]["isochore"][0](Xc, Yc)
        top, base = zc[:, :, 0], zc[:, :, 10]                              # the sand's top and base (10 layers of 1 m)
        if mound:
            assert np.allclose(top, plane + 10.0 * (1.0 - f), atol=1e-6) and np.allclose(base, plane + 10.0, atol=1e-6)
            assert (f == 0.0).any() and np.allclose((top - plane)[f == 0.0], 10.0)      # sunk by T where the sand has gone
            if barrier:
                inside = np.repeat(np.repeat(ndimage.binary_erosion(act[:, :, :10].any(axis=2), iterations=4), 2, axis=0),
                                   2, axis=1)
                assert np.allclose((zc[:, :, -1] - base)[inside], 8.0, atol=1e-6)
        else:
            assert np.allclose(top, plane, atol=1e-6) and np.allclose(base - plane, 10.0 * f, atol=1e-6)
            if barrier:
                inside = np.repeat(np.repeat(ndimage.binary_erosion(act[:, :, :10].any(axis=2), iterations=4), 2, axis=0),
                                   2, axis=1)
                assert np.allclose((zc[:, :, -1] - zc[:, :, 0])[inside], 18.0, atol=1e-6)


@pytest.mark.parametrize("kind", ["pinchout", "facies_change"])
@pytest.mark.parametrize("dip,taper,shift", [(1.1, 0.4, 10.0), (0.5, 1.2, 3.3)], ids=["no ridge", "ridge along the edge"])
def test_the_trap_measure_is_right_for_both_tops_and_the_convex_top_sinks_the_crest_and_not_the_closure(kind, dip, taper,
                                                                                                        shift):
    """Flat or convex, the cells' closure is that of the fine analytic map (1.5 cells' rise and 8 %). The convex top
    puts crest and spill down together, by up to the sand's thickness (10 m: the tip's top lies T below the plane), so
    that the closure stays that of the flat top, to 2 m; where the taper is steeper than the dip (2 tan(taper) >
    tan(dip)) the top of the edge is a ridge along it, and the spill (the ridge's low point) moves less (3 m)."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    thick, barrier = ([10.0, 8.0], True) if kind == "facies_change" else ([10.0], False)
    layers = _layers(x_len, y_len, dx, thick, dz=2.0)
    traps = {}
    for mound in (False, True):
        built = strat_trap(kind, x_len, y_len, 2000.0, thick, seed=2, barrier=barrier, column=12.0 if barrier else None,
                           dip=dip, taper_angle=taper, area=3.4e6, aspect=2.2, warp=0.2, wander=60.0, range_m=1000.0,
                           relief_sd=0.5, cell=dx, mound=mound)
        trap = trap_report(layers, built)[0]
        fine = _fine_traps(built, 2000.0, x_len, y_len)[0]
        assert trap["closure"] == pytest.approx(fine["height"], abs=1.5 * np.tan(np.radians(dip)) * dx + 0.08 * fine[
            "height"])
        traps[mound] = trap
    flat, convex = traps[False], traps[True]
    assert convex["closure"] == pytest.approx(flat["closure"], abs=2.0)
    assert 1.0 < convex["crest_depth"] - flat["crest_depth"] <= 10.0 + 1.0
    assert convex["spill_depth"] - flat["spill_depth"] == pytest.approx(shift, abs=1.0)
    if barrier:                       # the barrier logic keeps both dry; the mound's flat slab never reaches above its crest
        assert convex["limited_by"] == flat["limited_by"] == "barrier"
        assert flat["height"] - 1.5 <= convex["height"] <= 12.0 + 1e-9 and 0.5 * 12.0 < flat["height"] <= 12.0


@pytest.mark.parametrize("kind,extra", [("truncation", dict(taper_angle=0.3)), ("onlap", {}),
                                        ("pinchout_nose", dict(nose=NOSE, area=None)),
                                        ("truncation_nose", dict(nose=NOSE, area=None))])
def test_a_convex_up_top_is_not_for_an_erosion_surface_or_a_nose(kind, extra):
    with pytest.raises(ValueError, match="convex"):
        strat_trap(kind, 8000.0, 6000.0, 2000.0, [10.0], seed=1, dip=1.0, mound=True, **extra)
    strat_trap(kind, 8000.0, 6000.0, 2000.0, [10.0], seed=1, dip=1.0, mound=False, **extra)


# ---------------------------------------------------------------------------------------------------------------
# Every pool, and a contact that overfills none (review F5)

def _chain(values, exit=False):
    """A map of walls (infinite depth) with one row of cells of the given depths, 1 m across: its row is 1, its
    columns 1 to len(values). The map's edge is its first and last row and column, so the row has an exit only if
    ``exit``: the last cell is then on the edge."""
    depth = np.full((3, len(values) + (1 if exit else 2)), np.inf)
    depth[1, 1:len(values) + 1] = values
    return depth


def test_every_local_minimum_of_the_top_is_a_pool_with_the_level_where_it_spills_into_a_shallower_pool():
    """Depths 12 6 10 8 14 9 11 along a row of cells in walls, sealed all round: three pools, the elder rule. The crest of
    6 m is the body's: sealed, it fills to the deepest cell of the body (14 m). The pool of 8 m spills into it at the
    saddle of 10 m (its closure 2 m), that of 9 m at 14 m, the saddle on its left, into the pool of 6 m (the one on its
    right ends in a wall: 11 m, a dead end). Pools of less than ``min_height`` of closure are left out, the body's own
    never."""
    traps = _traps(_chain([12, 6, 10, 8, 14, 9, 11]), 1.0, 1.0, min_height=0.0)
    assert [t["crest"] for t in traps] == [(1, 2), (1, 4), (1, 6)] and [t["crest_depth"] for t in traps] == [6, 8, 9]
    assert [t["spill_depth"] for t in traps] == [None, 10.0, 14.0] and [t["closure"] for t in traps] == [8, 2, 5]
    assert [t["primary"] for t in traps] == [True, False, False] and [t["into"] for t in traps] == [None, (1, 2), (1, 2)]
    assert [t["spill_point"] for t in traps] == [None, (1, 3), (1, 5)] and [t["limited_by"] for t in traps] == [
        "sealed", "spill", "spill"]
    assert [t["area"] for t in traps] == [7.0, 1.0, 2.0]            # the pool of 6 m: the whole body, sealed
    assert [t["crest"] for t in _traps(_chain([12, 6, 10, 8, 14, 9, 11]), 1.0, 1.0, min_height=2.5)] == [(1, 2), (1, 6)]
    assert [t["crest"] for t in _traps(_chain([12, 6, 10, 8, 14, 9, 11]), 1.0, 1.0, min_height=9.0)] == [(1, 2)]


def test_a_pool_whose_water_reaches_the_edge_spills_there_and_a_pool_beyond_a_saddle_spills_into_it():
    """The same row ending in a cell of 5 m on the map's edge: that pool is open, with no closure (its crest is on the
    edge); the pool of 9 m spills into it at 11 m, the pool of 6 m (its water meets the open pool's at the saddle of 14
    m) at 14 m, and the pool of 8 m spills left, into the pool of 6 m, at 10 m."""
    traps = _traps(_chain([12, 6, 10, 8, 14, 9, 11, 5], exit=True), 1.0, 1.0, min_height=0.0)
    by = {t["crest_depth"]: t for t in traps}
    assert by[5.0]["spill_depth"] == 5.0 and by[5.0]["closure"] == 0.0 and by[5.0]["into"] is None
    assert by[9.0]["spill_depth"] == 11.0 and by[9.0]["into"] == by[5.0]["crest"]
    assert by[6.0]["spill_depth"] == 14.0 and by[6.0]["into"] == by[5.0]["crest"]
    assert by[8.0]["spill_depth"] == 10.0 and by[8.0]["into"] == by[6.0]["crest"]
    assert sum(t["primary"] for t in traps) == 1 and by[5.0]["primary"]


def _brute_pools(depth):
    """Each strict local minimum of ``depth`` and the level where it stops being a pool, by raising the water: the
    first level at which its connected water holds a shallower cell or a cell of the map's edge (4-connected, binary
    search over the levels)."""
    nx, ny = depth.shape
    out = {}
    levels = np.unique(depth[np.isfinite(depth)])

    def water(seed, level):
        seen, todo = {seed}, [seed]
        while todo:
            i, j = todo.pop()
            for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= a < nx and 0 <= b < ny and (a, b) not in seen and depth[a, b] <= level:
                    seen.add((a, b))
                    todo.append((a, b))
        return seen

    for i in range(nx):
        for j in range(ny):
            nbrs = [depth[a, b] for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                    if 0 <= a < nx and 0 <= b < ny]
            if not np.isfinite(depth[i, j]) or any(depth[i, j] > v for v in nbrs):
                continue

            def dies(level):
                cells = water((i, j), level)
                return any(depth[c] < depth[i, j] or c[0] in (0, nx - 1) or c[1] in (0, ny - 1) for c in cells)

            above = levels[levels >= depth[i, j]]                   # the water starts at the cell's own depth
            lo, hi = 0, len(above) - 1
            if not dies(above[hi]):
                out[(i, j)] = np.inf
                continue
            while lo < hi:
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if dies(above[mid]) else (mid + 1, hi)
            out[(i, j)] = float(above[lo])
    return out


def test_the_pools_of_random_maps_are_those_a_rising_water_level_finds():
    """Smooth random depth maps with walls (infinite depth) in patches: every strict local minimum is a pool, its spill the
    level at which its water first holds a shallower cell or a cell of the map's edge (found by raising the water, with
    nothing from ``strat_traps``), and ``zone_trap`` lists exactly those with a closure of at least ``min_height``
    (the body's own always)."""
    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(25):
        depth = 100.0 + 30.0 * ndimage.gaussian_filter(rng.standard_normal((22, 18)), 1.2)
        depth[ndimage.gaussian_filter(rng.standard_normal(depth.shape), 2.0) > 0.12] = np.inf
        expected = _brute_pools(depth)
        found = {t["crest"]: t for t in _traps(depth, 1.0, 1.0, min_height=0.0)}
        assert set(found) == set(expected)
        for crest, spill in expected.items():
            assert (np.inf if found[crest]["spill_depth"] is None else found[crest]["spill_depth"]) == pytest.approx(spill)
        listed = {t["crest"] for t in _traps(depth, 1.0, 1.0, min_height=3.0)}
        assert listed == {c for c, t in found.items() if t["primary"] or t["closure"] >= 3.0}
        checked += len(expected)
    assert checked > 100


def test_the_contact_limit_is_the_shallowest_column_of_the_edge_for_the_pool_that_holds_the_oil():
    """An initialisation by contacts puts oil in every cell above the contact, joined to the trap or not, and oil whose
    water meets the map's edge leaks: so no contact is admissible that is deeper than the shallowest column on the
    edge (5 m here: the pool of 6 m is then no trap to fill, its limit its crest). Without an exit (the row in
    walls, sealed) it is the trap's own limit, and an exit deeper than the spill does not bind."""
    open_row = {t["crest_depth"]: t for t in _traps(_chain([12, 6, 10, 8, 14, 9, 11, 5], exit=True), 1.0, 1.0,
                                                    min_height=0.0)}
    assert open_row[6.0]["contact_limit"] == 6.0 and open_row[6.0]["spill_depth"] == 14.0
    assert open_row[5.0]["contact_limit"] == 5.0
    sealed = {t["crest_depth"]: t for t in _traps(_chain([12, 6, 10, 8, 14, 9, 11]), 1.0, 1.0, min_height=0.0)}
    assert sealed[6.0]["contact_limit"] == 14.0 and sealed[8.0]["contact_limit"] == 10.0
    barred = _traps(_chain([12, 6, 10, 8, 14, 9, 11]), 1.0, 1.0, column=5.0, barrier_top=4.0, min_height=0.0)
    assert [t["limit_depth"] for t in barred] == [9.0, 9.0, 9.0] and [t["contact_limit"] for t in barred] == [9.0, 9.0,
                                                                                                           9.0]
    deep = _chain([12, 6, 10, 8, 14, 9, 11, 20], exit=True)           # an exit of 20 m, the primary pool's spill
    assert {t["crest_depth"]: t["contact_limit"] for t in _traps(deep, 1.0, 1.0, min_height=0.0)} == {6.0: 20.0, 8.0: 10.0,
                                                                                                        9.0: 14.0}


# ---------------------------------------------------------------------------------------------------------------
# The closure label is the realized one (review F6)

def test_meta_gives_the_nominal_closure_and_the_report_the_realized_one_a_mound_holding_half_as_much_again():
    """``meta`` has the tangent of the dip times the length of the drawn tongue under the name it deserves
    (``closure_nominal``, ``crest_nominal``, ``spill_nominal``), and the report's ``closure`` is what the cells hold:
    for a mound of 10 m on a gentle dip (0.3 degrees, 3.4 km2) the nominal 10.5 m, the realized 16 m (the mound's own
    relief, T (1 + x)^2 with x = 10.5 / 40), as the fine analytic map says; with a rough edge and a top with relief
    it is another number again. Nothing in ``meta`` is called ``expected``."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    layers = _layers(x_len, y_len, dx, [10.0], dz=1.0)
    mound = strat_trap("lens", x_len, y_len, 2000.0, [10.0], seed=2, dip=0.3, taper_angle=0.3, area=3.4e6, aspect=2.2,
                       warp=0.0)
    assert not [key for key in mound["meta"] if "expected" in key]
    nominal = mound["meta"]["closure_nominal"]
    assert nominal == pytest.approx(np.tan(np.radians(0.3)) * _length(3.4e6, 2.2))
    (trap,) = trap_report(layers, mound)
    assert trap["closure"] == pytest.approx(10.0 * (1.0 + nominal / 40.0) ** 2, rel=0.06) and trap["closure"] > 1.4 * nominal
    assert trap["closure"] == pytest.approx(_fine_traps(mound, 2000.0, x_len, y_len)[0]["height"], abs=1.5 * 0.0052 * dx
                                            + 0.05 * trap["closure"])
    rough = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=4, dip=1.2, taper_angle=0.4, area=3.4e6, aspect=2.2,
                       warp=0.2, wander=120.0, relief_sd=1.5, cell=dx)
    main = trap_report(layers, rough)[0]
    assert abs(main["closure"] - rough["meta"]["closure_nominal"]) > 1.0
    assert main["closure"] == pytest.approx(_fine_traps(rough, 2000.0, x_len, y_len)[0]["height"], abs=1.5 * 0.021 * dx +
                                            0.08 * main["closure"])
