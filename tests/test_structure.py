import numpy as np
from scipy import ndimage
import pytest

from resmill import structure as st
from resmill.layers.base import Layer


def test_algebra_with_scalars_callables_and_structures():
    a = st.Structure(lambda x, y: np.asarray(x, float) + 0 * np.asarray(y, float))
    assert (a + 5)(2, 3) == 7
    assert (5 + a)(2, 3) == 7
    assert (a + (lambda x, y: y))(2, 3) == 5
    assert (a + st.Structure(lambda x, y: y))(2, 3) == 5
    assert (a - 1)(2, 3) == 1
    assert (-a)(2, 3) == -2
    assert (2 * a)(2, 3) == 4
    assert (a * 2)(2, 3) == 4


def test_call_broadcasts():
    f = st.Structure(lambda x, y: 0 * x + 1.5)
    out = f(np.zeros((3, 4)), 0.0)
    assert out.shape == (3, 4)
    assert np.all(out == 1.5)


def test_anticline_analytic_points():
    f = st.anticline(amplitude=60, wavelength=1000, azimuth=0, center=(0, 0))
    assert f(0, 0) == pytest.approx(-60)          # crest lifted 60 m
    assert f(123, 0) == pytest.approx(-60)        # hinge runs along x
    assert f(0, 500) == pytest.approx(0)          # flank back at datum
    assert f(0, 250) == pytest.approx(-30)
    assert f(0, 1000) == pytest.approx(-60)       # next crest one wavelength away
    g = st.anticline(amplitude=60, wavelength=1000, azimuth=90, center=(0, 0))
    assert g(0, 777) == pytest.approx(-60)        # hinge now along y
    assert g(500, 0) == pytest.approx(0)


def test_syncline_is_negated_anticline():
    a = st.anticline(40, 800, center=(0, 0))
    s = st.syncline(40, 800, center=(0, 0))
    assert s(0, 0) == pytest.approx(-a(0, 0)) == pytest.approx(40)


def test_dome_analytic_points():
    f = st.dome(amplitude=50, radius=300, center=(0, 0))
    assert f(0, 0) == pytest.approx(-50)
    assert f(300, 0) == pytest.approx(-50 / np.e)
    assert f(0, -300) == pytest.approx(-50 / np.e)


def test_dome_elliptical_four_way_closure():
    # aspect stretches the closure along the azimuth axis (here +x).
    f = st.dome(amplitude=50, radius=100, aspect=2.0, azimuth=0, center=(0, 0))
    assert f(0, 0) == pytest.approx(-50)
    assert f(200, 0) == pytest.approx(-50 / np.e)    # along-axis: aspect * radius
    assert f(0, 100) == pytest.approx(-50 / np.e)    # cross-axis: radius
    # It closes in every direction: uplift decays along the long axis too.
    assert abs(f(400, 0)) < abs(f(200, 0)) < abs(f(0, 0))
    # azimuth rotates the long axis (90 -> along y).
    g = st.dome(amplitude=50, radius=100, aspect=2.0, azimuth=90, center=(0, 0))
    assert g(0, 200) == pytest.approx(-50 / np.e)
    assert g(100, 0) == pytest.approx(-50 / np.e)


def test_ramp_matches_layer_dip_plane():
    layer = Layer(4, 3, 2, 40, 30, 4, top_depth=1000, dip=7.5)
    f = st.ramp(7.5)
    assert np.allclose(f(layer.X, layer.Y), layer.z1 - 1000)


def test_fault_step_and_azimuth_inference():
    f = st.fault(throw=25, x0=100)      # trace of constant x
    assert f(150, 7) == 25
    assert f(50, -3) == 0
    assert f(100, 0) == 0               # exactly on the trace: upthrown side
    g = st.fault(throw=25, y0=40)       # trace of constant y
    assert g(0, 50) == 25
    assert g(0, 30) == 0


def test_surface_reproduces_grid_and_interpolates():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(5, 4))
    f = st.surface(arr, x_len=40, y_len=30)
    xs = np.linspace(0, 40, 5)
    ys = np.linspace(0, 30, 4)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    assert np.allclose(f(X, Y), arr)
    mid = f(0.5 * (xs[0] + xs[1]), ys[0])
    assert mid == pytest.approx(0.5 * (arr[0, 0] + arr[1, 0]))


def test_lazy_center_defaults_to_grid_middle():
    f = st.anticline(amplitude=60, wavelength=4000, azimuth=0)
    X, Y = np.meshgrid(np.linspace(0, 1000, 21), np.linspace(0, 800, 17), indexing='ij')
    vals = f(X, Y)
    assert vals.min() == pytest.approx(-60)
    assert np.argmin(vals[0]) == 8        # crest at the middle y row


def _depth_map(fld, half=9000.0, n=361):
    """Depth shift of ``fld`` on a regular map grid centred on the origin: (x, y, depth, spacing)."""
    x = np.linspace(-half, half, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return X, Y, fld(X, Y), x[1] - x[0]


def test_closure_stats_measures_a_paraboloid_dome():
    """A paraboloid of radius R and relief H on a flat surround closes over pi R^2 with height H."""
    x = np.linspace(-3000.0, 3000.0, 301)
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.hypot(X, Y)
    depth = 2000.0 - np.where(r < 2000.0, 80.0 * (1.0 - (r / 2000.0) ** 2), 0.0)
    stats = st.closure_stats(depth, x[1] - x[0], x[1] - x[0])
    assert stats["area"] == pytest.approx(np.pi * 2000.0 ** 2, rel=0.02)
    assert stats["height"] == pytest.approx(80.0, rel=0.01)
    assert stats["spill_depth"] == pytest.approx(2000.0)
    assert stats["mask"][150, 150] and not stats["mask"][0, 0]


@pytest.mark.parametrize("kw", [dict(), dict(aspect=3.0, azimuth=30.0), dict(limb_ratio=2.5, tilt=0.4),
                                dict(aspect=2.0, satellites=2, seed=4, tilt=0.2)])
def test_closure_has_the_requested_area_and_height(kw):
    """``closure`` scales its shape so the trap it makes closes over ``area`` with relief ``height``,
    whatever its aspect, asymmetry, tilt and satellite culminations."""
    fld = st.closure(area=8e6, height=120.0, center=(0.0, 0.0), **kw)
    X, Y, depth, d = _depth_map(fld)
    crest = np.unravel_index(np.argmin(np.hypot(X - fld.crest_offset[0], Y - fld.crest_offset[1])), X.shape)
    stats = st.closure_stats(depth, d, d, crest=crest)
    assert depth[crest] == pytest.approx(depth[np.hypot(X, Y) < 1500.0].min(), abs=0.5)   # it is the crest
    assert stats["area"] == pytest.approx(8e6, rel=0.03)
    assert stats["height"] == pytest.approx(120.0, rel=0.02)


def test_closure_limbs_follow_the_limb_ratio():
    """The forelimb (the side the azimuth normal points to) is ``limb_ratio`` times steeper than the backlimb."""
    fld = st.closure(area=8e6, height=120.0, center=(0.0, 0.0), limb_ratio=2.5)
    t = np.linspace(0.0, 3000.0, 601)
    fore, back = fld(0.0 * t, t), fld(0.0 * t, -t)             # azimuth 0: the dip direction is +y
    slope = lambda prof: np.abs(np.diff(prof)).max()
    assert slope(fore) / slope(back) == pytest.approx(2.5, rel=0.1)


def test_closure_is_reproducible_and_lifts_its_crest():
    a = st.closure(area=4e6, height=60.0, satellites=2, seed=7, center=(0.0, 0.0))
    b = st.closure(area=4e6, height=60.0, satellites=2, seed=7, center=(0.0, 0.0))
    X, Y, depth, _ = _depth_map(a)
    assert np.array_equal(depth, b(X, Y))
    assert depth.min() < 0.0                                     # the crest is lifted (positive down)


def test_roughness_has_its_sd_and_range_and_is_a_function_of_position():
    """``roughness`` is a Gaussian-covariance random surface, exp(-3 (r/R)^2): its SD is ``sd``, and
    at lag R the correlation has fallen to about 5 %. It is fixed by its seed and its footprint, so
    evaluating it at corners and at nodes gives one surface."""
    rough = st.roughness(sd=9.0, range_m=1500.0, x_len=30000.0, y_len=30000.0, seed=3)
    x = np.arange(0.0, 30000.0, 100.0)
    X, Y = np.meshgrid(x, x, indexing="ij")
    z = rough(X, Y)
    assert np.std(z) == pytest.approx(9.0, rel=0.15)
    lag = 15                                                     # 1,500 m
    corr = np.corrcoef(z[:-lag].ravel(), z[lag:].ravel())[0, 1]
    assert corr < 0.15
    corr_short = np.corrcoef(z[:-2].ravel(), z[2:].ravel())[0, 1]   # 200 m: exp(-3 (200/1500)^2) = 0.95
    assert corr_short == pytest.approx(0.95, abs=0.04)
    again = st.roughness(sd=9.0, range_m=1500.0, x_len=30000.0, y_len=30000.0, seed=3)
    assert np.array_equal(again(X, Y), z)
    assert rough(np.array([1234.5]), np.array([678.9]))[0] == pytest.approx(
        again(np.array([[1234.5]]), np.array([[678.9]]))[0, 0])


def test_closure_warp_makes_the_outline_lobate():
    """``warp`` distorts the culmination's coordinates with smooth random waves, so the trap's outline
    departs from an ellipse (a longer perimeter for its area) while its area and relief stay as asked."""
    def outline_ratio(fld):
        X, Y, depth, d = _depth_map(fld)
        crest = np.unravel_index(np.argmin(np.hypot(X - fld.crest_offset[0], Y - fld.crest_offset[1])), X.shape)
        stats = st.closure_stats(depth, d, d, crest=crest)
        mask = stats["mask"]
        edge = mask & ~ndimage.binary_erosion(mask)
        return edge.sum() * d / np.sqrt(4.0 * np.pi * stats["area"]), stats
    plain, _ = outline_ratio(st.closure(area=8e6, height=120.0, aspect=1.5, center=(0.0, 0.0)))
    lobate, stats = outline_ratio(st.closure(area=8e6, height=120.0, aspect=1.5, warp=0.4, seed=2,
                                             center=(0.0, 0.0)))
    assert lobate > 1.1 * plain
    assert stats["area"] == pytest.approx(8e6, rel=0.03) and stats["height"] == pytest.approx(120.0, rel=0.02)


def _walled_bowl(gate):
    """A 5 x 5 map, depth 20 on its edge and 10 inside, every edge into the inner 3 x 3 sealed but one (between the
    edge cell (0, 2) and the inner cell (1, 2)), which passes at ``gate``: ``(depth, wall_x, wall_y)``."""
    depth = np.full((5, 5), 10.0)
    depth[[0, -1], :] = depth[:, [0, -1]] = 20.0
    wall_x, wall_y = np.zeros((4, 5)) - np.inf, np.zeros((5, 4)) - np.inf
    wall_x[[0, 3], 1:4] = np.inf
    wall_y[1:4, [0, 3]] = np.inf
    wall_x[0, 2] = gate
    return depth, wall_x, wall_y


@pytest.mark.parametrize("gate,level", [(50.0, 50.0), (5.0, 20.0), (-np.inf, 20.0), (np.inf, np.inf)])
def test_the_spill_flood_crosses_an_edge_at_its_pass_level(gate, level):
    """A path crosses a cell edge at the highest of the level so far, the next cell's depth and the edge's pass level:
    sealed (+inf) it reaches nothing, free (-inf) or below the cells' depth it changes nothing."""
    depth, wall_x, wall_y = _walled_bowl(gate)
    spill = st._spill_levels(depth, wall_x, wall_y)
    assert np.all(spill[1:4, 1:4] == level) and np.all(spill[[0, -1], :] == 20.0)


def test_the_spill_flood_takes_the_cheaper_way_in_when_it_comes_later():
    """A cell next to a low edge cell over a high pass level and to a higher edge cell over a free edge takes the
    cheaper of the two, though the low neighbour is reached first: 30, not 100."""
    depth = np.full((3, 3), 30.0)
    depth[0, 1], depth[1, 1] = 10.0, 5.0
    wall_x = np.full((2, 3), -np.inf)
    wall_x[0, 1] = 100.0
    assert st._spill_levels(depth, wall_x)[1, 1] == 30.0


def test_the_spill_flood_is_the_shallowest_path_for_any_pass_levels():
    """Against plain relaxation to the fixed point: the level of every cell is the least, over paths from the edge, of
    the highest depth or pass level along the path, whatever the pass levels (free, sealed or between)."""
    def relax(depth, wall_x, wall_y):
        level = np.full(depth.shape, np.inf)
        edge = np.zeros(depth.shape, dtype=bool)
        edge[[0, -1], :] = edge[:, [0, -1]] = True
        level[edge] = depth[edge]
        while True:
            new = level.copy()
            for wall, lo, hi in ((wall_x, np.s_[:-1], np.s_[1:]), (wall_y, np.s_[:, :-1], np.s_[:, 1:])):
                new[hi] = np.minimum(new[hi], np.maximum(np.maximum(depth[hi], level[lo]), wall))
                new[lo] = np.minimum(new[lo], np.maximum(np.maximum(depth[lo], level[hi]), wall))
            if np.array_equal(new, level):
                return level
            level = new

    rng = np.random.default_rng(8)
    for _ in range(6):
        depth = ndimage.gaussian_filter(rng.normal(size=(18, 14)), 1.5) * 40.0 + 2000.0
        walls = [np.where(rng.random(shape) < 0.3, np.inf, 2000.0 + 20.0 * rng.normal(size=shape))
                 for shape in ((17, 14), (18, 13))]
        assert np.array_equal(st._spill_levels(depth, *walls), relax(depth, *walls))


def test_the_spill_flood_reads_a_boolean_wall_as_sealed_and_open():
    """A boolean wall array is the same as pass levels of +inf (True) and -inf (False)."""
    rng = np.random.default_rng(3)
    depth = ndimage.gaussian_filter(rng.normal(size=(30, 25)), 2.0) * 30.0 + 2000.0
    wall_x, wall_y = rng.random((29, 25)) < 0.3, rng.random((30, 24)) < 0.3
    levels = lambda w: np.where(w, np.inf, -np.inf)
    assert np.array_equal(st._spill_levels(depth, wall_x, wall_y),
                          st._spill_levels(depth, levels(wall_x), levels(wall_y)))
    assert np.array_equal(st._spill_levels(depth, wall_x), st._spill_levels(depth, levels(wall_x), None))
    assert np.array_equal(st._spill_levels(depth), st._spill_levels(depth, np.zeros((29, 25), bool)))


def test_the_spill_flood_ends_on_columns_of_infinite_depth():
    """Dead columns (infinite depth) neither hang the flood nor let oil through: the pocket a ring of them closes off
    keeps an infinite level, as behind a wall."""
    i, j = np.indices((9, 9))
    ring = np.maximum(abs(i - 4), abs(j - 4))
    depth = np.where(ring == 2, np.inf, 10.0)
    spill = st._spill_levels(depth)
    assert np.all(spill[ring <= 2] == np.inf) and np.all(spill[ring > 2] == 10.0)


def test_taper_is_a_linear_ramp_across_the_pinch_out_line():
    """The factor is 0 at and beyond the line, 1 once ``taper_m`` in from it, and linear between: with the line at
    1,000 m along the dip direction (+y, azimuth 0) and 400 m of taper, 0.25 / 0.5 / 1 at 1,100 / 1,200 / 1,400 m,
    whatever x is."""
    f = st.taper(position=1000.0, taper_m=400.0)
    y = np.array([0.0, 999.0, 1000.0, 1100.0, 1200.0, 1399.0, 1400.0, 5000.0])
    expected = [0.0, 0.0, 0.0, 0.25, 0.5, 399.0 / 400.0, 1.0, 1.0]
    for x in (0.0, 123.4, 7000.0):
        assert f(x + 0.0 * y, y) == pytest.approx(expected)


def test_taper_follows_the_azimuth_normal():
    """The line lies across the azimuth normal ``(sin az, cos az)``, the dip direction of ``ramp``: at azimuth 90 the
    factor depends on x alone, and at 30 degrees it is (x sin 30 + y cos 30 - position) / taper_m."""
    f = st.taper(position=200.0, taper_m=1000.0, azimuth=90.0)
    assert f(700.0, 0.0) == pytest.approx(0.5) and f(700.0, 4321.0) == pytest.approx(0.5)
    g = st.taper(position=200.0, taper_m=1000.0, azimuth=30.0)
    x, y = 500.0, 600.0
    assert g(x, y) == pytest.approx((x * 0.5 + y * np.sqrt(3.0) / 2.0 - 200.0) / 1000.0) == pytest.approx(0.5696, abs=1e-4)


def test_taper_gradient_is_the_drawn_angle():
    """Thickness T times the factor changes by tan(angle) per metre across the taper: its steepest slope on a map is
    T / taper_m, 10 m over 573 m for 1 degree, and the factor never decreases downdip."""
    t_m, angle = 10.0, 1.0
    f = st.taper(position=1500.0, taper_m=t_m / np.tan(np.radians(angle)))
    y = np.linspace(0.0, 4000.0, 8001)
    thickness = t_m * f(0.0 * y, y)
    assert np.max(np.diff(thickness) / np.diff(y)) == pytest.approx(np.tan(np.radians(angle)), rel=1e-9)
    assert np.all(np.diff(thickness) >= 0.0) and thickness.min() == 0.0 and thickness.max() == t_m


def test_taper_line_wanders_as_roughness_does():
    """The line lies at ``position`` plus a ``roughness`` surface of SD ``wander`` and range ``range_m`` (the same
    seed gives the same surface), so the factor is the straight-line one with that surface subtracted from the
    distance in from the line."""
    kw = dict(position=1500.0, taper_m=500.0)
    f = st.taper(**kw, wander=60.0, range_m=1200.0, x_len=6000.0, y_len=4000.0, seed=5)
    rough = st.roughness(60.0, 1200.0, 6000.0, 4000.0, seed=5)
    X, Y = np.meshgrid(np.linspace(0.0, 6000.0, 61), np.linspace(0.0, 4000.0, 41), indexing="ij")
    assert np.array_equal(f(X, Y), np.clip((Y - 1500.0 - rough(X, Y)) / 500.0, 0.0, 1.0))
    edge = (f(X, Y) > 0.0).argmax(axis=1)                                 # the first row of sand in each column
    assert edge.max() > edge.min()                                        # the line really is not straight


def _disc(cx, cy, radius):
    """An outline of a disc: negative inside, as ``closure`` is (a distance, so its footprint is exact)."""
    return st.Structure(lambda x, y: np.hypot(np.asarray(x) - cx, np.asarray(y) - cy) - radius)


def test_taper_outline_tapers_from_the_footprint_edge():
    """With an ``outline`` the sand is its footprint (where the outline is below ``level``) and the factor rises with
    the distance in from its edge: a disc of radius 1,000 m with 400 m of taper has f = (1000 - r) / 400 up to 1 and
    is 0 outside it, to within the raster step (about 8 m here)."""
    f = st.taper(None, 400.0, outline=_disc(3000.0, 2000.0, 1000.0), x_len=6000.0, y_len=4000.0)
    r = np.array([0.0, 300.0, 599.0, 700.0, 900.0, 990.0, 1010.0, 1500.0])
    assert f(3000.0 + r, 2000.0 + 0.0 * r) == pytest.approx(np.clip((1000.0 - r) / 400.0, 0.0, 1.0), abs=0.03)
    assert f(3000.0, 2000.0 + r[0]) == 1.0 and f(0.0, 0.0) == 0.0 and f(5500.0, 3900.0) == 0.0


def test_taper_outline_unites_with_the_line():
    """A line and an outline together are the sand of both: a disc centred on the line is a tongue protruding updip
    from the sheet (downdip of the line the sheet's ramp, updip of it the disc's)."""
    f = st.taper(2000.0, 400.0, outline=_disc(3000.0, 2000.0, 1000.0), x_len=6000.0, y_len=4000.0)
    assert f(500.0, 2100.0) == pytest.approx(0.25)                          # the sheet, 100 m down from the line
    assert f(500.0, 1900.0) == 0.0                                          # updip of the line, outside the disc
    assert f(3000.0, 1000.0) == pytest.approx(0.0, abs=0.03)                # the disc's tip, 1,000 m updip of its centre
    assert f(3000.0, 1200.0) == pytest.approx(0.5, abs=0.03)                # 200 m in from it
    assert f(3000.0, 1500.0) == pytest.approx(1.0, abs=0.03)                # 500 m in from it: full thickness


@pytest.mark.parametrize("kw,message", [
    (dict(position=1000.0, taper_m=0.0), "taper_m"),
    (dict(position=None, taper_m=100.0), "position"),
    (dict(position=1000.0, taper_m=100.0, wander=10.0, range_m=500.0), "x_len"),
    (dict(position=1000.0, taper_m=100.0, wander=10.0, x_len=4000.0, y_len=4000.0), "range_m"),
    (dict(position=None, taper_m=100.0, outline=_disc(0.0, 0.0, 5.0)), "x_len"),
])
def test_taper_refuses_what_it_cannot_draw(kw, message):
    with pytest.raises(ValueError, match=message):
        st.taper(**kw)


def test_taper_with_isochore_gives_thickness_times_factor_and_collapses_where_it_is_zero():
    """As an isochore the factor scales a zone: at every node the zone is T f thick, and a column is inactive exactly
    where all four of its corners hold less than the 5 mm that ``to_grdecl`` counts as a cell."""
    from resmill.export import _build_geometry
    nx, ny, nz, dx = 30, 24, 4, 50.0
    layer = Layer(nx, ny, nz, nx * dx, ny * dx, 20.0, top_depth=1500.0)
    f = st.taper(position=500.0, taper_m=300.0, azimuth=0.0)
    Xc, Yc, zc, act = _build_geometry([layer], isochore=[f])
    thick = zc[:, :, -1] - zc[:, :, 0]
    assert np.allclose(thick, 20.0 * f(Xc, Yc), atol=1e-6) and thick.min() == 0.0
    layer_thick = (thick / nz).reshape(nx, 2, ny, 2).max(axis=(1, 3))
    assert np.array_equal(act.any(axis=2), layer_thick > 5e-3)
    assert not act[:, :10].any() and act[:, 12:].all()
