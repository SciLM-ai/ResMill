"""The trap measure: pools, spill, Berg's barrier column and the contact limit (``resmill.trap_measure``)."""
import numpy as np
import pytest
from scipy import ndimage

from resmill import structure as st
from resmill.export import _build_geometry
from resmill.layers.base import Layer
from resmill.strat_traps import barrier_column, effective_grain_size, trap_report, zone_top, zone_trap
from resmill.trap_measure import _traps
from tests.strat_helpers import _disc


FT = 0.3048


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
    grain size (the first is the height for oil to migrate through the same sand, which no barrier is: zero here)."""
    args = dict(sigma=0.035)
    assert barrier_column(100.0, 0.2e-3, 0.05e-3, **args) / FT == pytest.approx(55.0, abs=1.0)
    assert barrier_column(1000.0, 0.2e-3, 0.05e-3, **args) / FT == pytest.approx(5.0, abs=0.7)
    assert barrier_column(100.0, 0.2e-3, 0.01e-3, **args) / FT == pytest.approx(300.0, abs=3.0)
    assert barrier_column(1000.0, 0.2e-3, 0.01e-3, **args) / FT == pytest.approx(30.0, abs=0.5)
    assert barrier_column(100.0, 0.2e-3, 0.1e-3, **args) == pytest.approx(7.6, rel=0.01)
    assert barrier_column(100.0, 0.2e-3, 0.2e-3, **args) == 0.0     # his 300 cm stringer is the height for oil to migrate


def test_barrier_column_scales_as_the_equation_says_and_is_never_negative():
    base = barrier_column(300.0, 1.0e-4, 3.0e-5)
    assert barrier_column(600.0, 1.0e-4, 3.0e-5) == pytest.approx(base / 2.0)           # 1 / (density contrast)
    assert barrier_column(300.0, 1.0e-4, 3.0e-5, sigma=0.060) == pytest.approx(2.0 * base)   # interfacial tension
    assert barrier_column(300.0, 1.0e-4, 1.0e-4) == 0.0                                 # a barrier no finer: none
    assert barrier_column(300.0, 1.0e-4, 0.99e-4) > 0.0 and barrier_column(300.0, 3.0e-5, 1.0e-4) == 0.0


@pytest.mark.parametrize("k,phi,d", [(900, 0.32, 6.0e-5), (533, 0.21, 13.5e-5), (65, 0.18, 6.9e-5),
                                     (77, 0.24, 3.6e-5), (75, 0.26, 2.9e-5), (170, 0.29, 3.3e-5),
                                     (25, 0.24, 2.1e-5), (5, 0.20, 1.5e-5)])
def test_effective_grain_size_is_berg_empirical_equation_of_permeability_and_porosity(k, phi, d):
    """D = (1.89 k n^-5.1)^0.5 cm (k in mD, the porosity n in percent), against the grain sizes of his Table 1 that
    follow from their own porosity and permeability (two entries of the Milbur barrier do not: 5 to 8 % off)."""
    assert effective_grain_size(k, phi) == pytest.approx(d, rel=0.03)


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
