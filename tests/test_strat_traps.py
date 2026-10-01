"""Stratigraphic traps: the trap of a zone that pinches out, Berg's barrier columns, and the trap builder."""
import numpy as np
import pytest

from resmill import structure as st
from resmill.export import _build_geometry
from resmill.layers.base import Layer
from resmill.strat_traps import barrier_column, effective_grain_size, strat_trap, trap_report, zone_trap

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
    assert trap["height"] == pytest.approx(np.tan(np.radians(1.0)) * 2000.0, abs=np.tan(np.radians(1.0)) * dx)
    assert trap["limit_depth"] == pytest.approx(trap["crest_depth"] + trap["height"])
    assert trap["area"] == pytest.approx(np.pi * (1000.0 + 0.5 * dx) ** 2, rel=0.05)   # the columns the disc touches
    assert np.array_equal(trap["mask"], act.any(axis=2))              # the whole lens, its deepest columns included


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
    assert built["meta"]["closure_expected"] == pytest.approx(closure, abs=0.5)
    assert built["meta"]["closure_expected"] == pytest.approx(np.tan(np.radians(dip)) * _length(area, aspect))
    (trap,) = trap_report(_layers(x_len, y_len, dx, [10.0]), built)
    cell = np.tan(np.radians(dip)) * dx                                   # the depth a cell's width is worth
    assert trap["limited_by"] == "spill"
    assert trap["height"] == pytest.approx(closure, abs=cell + 0.02 * closure)
    assert trap["crest_depth"] == pytest.approx(built["meta"]["crest_expected"], abs=cell)
    assert trap["spill_depth"] == pytest.approx(built["meta"]["spill_expected"], abs=cell)
    assert area * 0.97 < trap["area"] < area * 1.10


def test_a_straight_pinch_out_line_on_a_plane_monocline_has_no_closure():
    """Every point of the line lies at one depth, so the oil spills along it: the trap has no height or area."""
    built = strat_trap("pinchout", 8000.0, 6000.0, 2000.0, [10.0], seed=1, dip=1.0, taper_angle=0.5, area=None)
    (trap,) = trap_report(_layers(8000.0, 6000.0, 50.0, [10.0]), built)
    assert trap["height"] == 0.0 and trap["area"] == 0.0
    assert built["meta"]["closure_expected"] == 0.0


@pytest.mark.parametrize("dip,area,aspect,model,closure", EXAMPLES[1:], ids=["P50", "P90"])
def test_an_enclosed_lens_is_sealed_and_holds_the_dip_times_its_whole_length(dip, area, aspect, model, closure):
    x_len, y_len, dx = model
    built = strat_trap("lens", x_len, y_len, 2000.0, [10.0], seed=2, dip=dip, taper_angle=0.5, area=area,
                       aspect=aspect, warp=0.0)
    (trap,) = trap_report(_layers(x_len, y_len, dx, [10.0]), built)
    assert trap["limited_by"] == "sealed" and trap["spill_depth"] is None
    expected = np.tan(np.radians(dip)) * _length(area, aspect)
    assert built["meta"]["closure_expected"] == pytest.approx(expected)
    cell = np.tan(np.radians(dip)) * dx
    assert trap["height"] == pytest.approx(expected, abs=cell + 0.02 * expected)
    assert trap["crest_depth"] == pytest.approx(built["meta"]["crest_expected"], abs=cell)
    assert area * 0.97 < trap["area"] < area * 1.10


def test_a_facies_change_is_the_same_trap_with_a_barrier_zone_taking_the_sand_s_thickness():
    """The barrier zone thickens as the sand thins, so the interval keeps its thickness, the barrier's cells are active
    under every column, and the sand's trap is that of the pinch-out of the same sand."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    kw = dict(seed=5, dip=1.1, taper_angle=0.5, area=3.4e6, aspect=2.2, warp=0.3, wander=40.0, range_m=800.0)
    sand_only = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], **kw)
    both = strat_trap("facies_change", x_len, y_len, 2000.0, [10.0, 8.0], barrier=True, **kw)
    layers = _layers(x_len, y_len, dx, [10.0, 8.0])
    _, _, zc, act = _build_geometry(layers, **both["kwargs"])
    assert np.allclose(zc[:, :, -1] - zc[:, :, 0], 18.0, atol=1e-6)         # the interval keeps its 10 + 8 m
    assert act[:, :, 5:].all() and not act[:, :, :5].all()                  # barrier everywhere, sand not
    (net,) = trap_report(layers, both)
    (alone,) = trap_report(layers[:1], sand_only)
    assert net["crest"] == alone["crest"] and net["spill_point"] == alone["spill_point"]
    for key in ("crest_depth", "spill_depth", "area", "height"):
        assert net[key] == pytest.approx(alone[key])
    assert both["meta"]["net_layers"] == 1 and both["meta"]["barrier"] is True


def test_the_same_seed_builds_the_same_trap_and_another_builds_another():
    kw = dict(dip=1.0, taper_angle=0.4, area=2.0e6, aspect=2.0, warp=0.3, wander=50.0, range_m=700.0)
    a, b = (strat_trap("pinchout", 6000.0, 5000.0, 1800.0, [8.0], seed=s, **kw) for s in (7, 7))
    c = strat_trap("pinchout", 6000.0, 5000.0, 1800.0, [8.0], seed=8, **kw)
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
