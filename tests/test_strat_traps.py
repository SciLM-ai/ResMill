"""Stratigraphic traps: the trap of a zone that pinches out, Berg's barrier columns, and the trap builder."""
import numpy as np
import pytest

from resmill import structure as st
from resmill.export import _build_geometry
from resmill.layers.base import Layer
from resmill.strat_traps import zone_trap


def _disc(cx, cy, radius):
    """An outline of a disc: negative inside, as ``closure`` is (a distance, so its footprint is exact)."""
    return st.Structure(lambda x, y: np.hypot(np.asarray(x) - cx, np.asarray(y) - cy) - radius)


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
