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


def test_spill_levels_treat_cells_without_rock_as_walls():
    """A cell of infinite depth is one a fault collapsed (no rock): no path crosses it and its own level stays infinite.
    Two of them side by side once kept pushing each other onto the flood's queue for ever. A band of them across a map
    splits it into two maps (each spills at its own edge), and a pit ringed by them has no way out: its level is infinite."""
    band = np.full((5, 8), 7.0)
    band[:, 3:5] = np.inf                                                # a collapsed band, two cells wide, across the map
    band[2, 6] = 1.0                                                     # a pit east of it
    band[2, 1] = 2.0                                                     # and one west
    spill = st._spill_levels(band)
    assert np.isinf(spill[:, 3:5]).all()
    assert (spill[2, 6], spill[2, 1]) == (7.0, 7.0)                      # each spills over its own side's 7 m rim
    ringed = np.full((7, 7), 10.0)
    ringed[2:5, 2:5] = np.inf
    ringed[3, 3] = 2.0                                                   # a pit inside a ring of collapsed cells
    spill = st._spill_levels(ringed)
    assert np.isinf(spill[2:5, 2:5]).all()
    assert (spill[np.isfinite(ringed)][ringed[np.isfinite(ringed)] == 10.0] == 10.0).all()


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


# ----- growth strata: the thickness factor of a zone laid down while a fault moved -----

def growth_fault(**kw):
    from resmill.faults import Fault
    return Fault(**{**dict(center=(1000.0, 1000.0), strike=90.0, length=10000.0, throw=50.0, dip=60.0, z_center=2000.0,
                           hw_share=1.0, drag=(0.0, 0.0), name="F1"), **kw})


def test_growth_is_one_at_the_footwall_cutoff_and_the_expansion_a_width_beyond_it():
    """A 60 degree fault whose plane at 2,000 m is at x = 1,000 m, hanging wall east: at the zone's depth, 2,100 m, its trace
    is 100 / tan(60) = 57.74 m east. The factor is 1 there and west of it, EI a width on, and a smooth step between
    (3 t^2 - 2 t^3: 0.156 of the way up a quarter of the way along, half way up at the middle), on the fault's centre line."""
    from resmill.faults import _plane
    g = st.growth(growth_fault(), expansion=1.8, width=300.0, depth=2100.0)
    x0 = 1000.0 + 100.0 / np.tan(np.radians(60.0))
    x = np.array([x0 - 500.0, x0, x0 + 75.0, x0 + 150.0, x0 + 300.0, x0 + 4000.0])
    assert g(x, np.full(6, 1000.0)) == pytest.approx([1.0, 1.0, 1.0 + 0.8 * 0.15625, 1.4, 1.8, 1.8], abs=1e-9)
    f = growth_fault(flatten=2500.0, dip=40.0)                             # listric: the trace is the curve's at 2,100 m
    h0 = float(_plane(f, 2000.0)[1](2100.0))
    xl = np.array([1000.0 + h0 - 1.0, 1000.0 + h0 + 300.0])
    assert st.growth(f, 1.8, 300.0, 2100.0)(xl, np.full(2, 1000.0)) == pytest.approx([1.0, 1.8], abs=1e-9)


def test_growth_dies_along_strike_with_the_throw_profile():
    """Half a half-length along the strike the throw of a fault is 0.559 of its centre's (Walsh and Watterson: (1 - r)^1.5
    sqrt(1 + 3 r) at r = 0.5), and nothing beyond the tip: so the excess expansion is 0.559 of the centre's there and
    zero past the tip, 1 in the footwall."""
    g = st.growth(growth_fault(), expansion=2.0, width=100.0, depth=2000.0)       # the zone at the tip ellipse's centre
    far = 1000.0 + 2000.0                                                          # clear of the transition
    trace = lambda s: (far, 1000.0 - s)                                           # the trace runs along -y at strike 90
    assert g(*trace(0.0)) == pytest.approx(2.0)
    assert g(*trace(2500.0)) == pytest.approx(1.0 + 0.559017, abs=1e-5)           # r = 2500 / 5000 = 0.5
    assert g(*trace(5000.0)) == pytest.approx(1.0) and g(*trace(7000.0)) == pytest.approx(1.0)
    assert g(far - 2500.0, 1000.0) == pytest.approx(1.0)                          # footwall


@pytest.mark.parametrize("flatten", [None, 2500.0])
def test_growth_without_a_width_steps_up_over_the_horizons_heave(flatten):
    """``width=None``: the zone thickens over the horizon's heave at the fault's centre line (the distance between its
    footwall and hanging-wall cutoffs), so the hanging wall begins at the full expansion. The heave of 50 m of throw is
    50 / tan(60) = 28.9 m for a 60 degree plane, and (L / tan(dip)) (exp(50 / L) - 1) for a 40 degree fault whose tan(dip)
    falls by 1/e every L = 2.5 km below its bend (the horizon is at the bend); half way across it the factor is half way up."""
    dip = 60.0 if flatten is None else 40.0
    f = growth_fault(flatten=flatten, dip=dip, z_center=2000.0)                # throw 50 m at the centre line, trace at x = 1,000
    if flatten is None:
        heave = 50.0 / np.tan(np.radians(dip))
    else:
        heave = flatten / np.tan(np.radians(dip)) * (np.exp(50.0 / flatten) - 1.0)
    x = 1000.0 + heave * np.array([-0.2, 0.0, 0.5, 1.0, 1.5])
    assert st.growth(f, 2.0, None, 2000.0)(x, np.full(5, 1000.0)) == pytest.approx([1.0, 1.0, 1.5, 2.0, 2.0], abs=1e-6)
    assert st.growth(f, 2.0, None, 9000.0)(np.array([5000.0]), np.array([1000.0])) == pytest.approx(1.0)   # no throw there


@pytest.mark.parametrize("flatten", [None, 2500.0])
def test_growth_thickens_the_zone_by_the_expansion_in_the_hanging_wall_only(flatten):
    """As the isochore of a 40 m zone cut by a fault whose hanging wall moves rigidly (a plane) or by vertical shear (listric,
    toward its flattening plane): 5 m cells are EI x 5 m in the hanging wall far from the fault and 5 m in the footwall, none
    negative."""
    from resmill.export import _build_geometry
    nx, ny, nz, dx = 160, 6, 8, 50.0
    layer = Layer(nx, ny, nz, nx * dx, ny * dx, 40.0, top_depth=2000.0, kzkx=0.1)
    f = growth_fault(center=(1000.0, 150.0), flatten=flatten, dip=40.0 if flatten else 60.0, throw=30.0, z_center=2020.0)
    g = st.growth(f, 2.2, 200.0, 2020.0)
    _, _, zc, _ = _build_geometry([layer], isochore=[g], faults=[f])
    thick = np.diff(zc[:, 6, :], axis=1)
    assert thick.min() >= 0.0                                                        # the cut-out beside the plane is thin, never negative
    assert thick[20:30].max() == pytest.approx(5.0, abs=1e-6)                        # x 0.5-0.75 km: footwall, clear of the fault
    far = thick[-20:]                                       # plus the throw's own change over the thickened zone: a few cm
    assert far.max() == pytest.approx(11.0, abs=0.05) and far.min() == pytest.approx(11.0, abs=0.05)


def test_growth_refuses_what_has_no_meaning():
    """It needs the fault's own depth scale (z_center), a width and an expansion above 0."""
    from resmill.faults import Fault
    with pytest.raises(ValueError, match="z_center"):
        st.growth(Fault(center=(0.0, 0.0), strike=0.0, length=1000.0, throw=10.0), 1.5, 100.0, 2000.0)
    for kw in (dict(width=0.0), dict(width=-5.0), dict(expansion=0.0)):
        with pytest.raises(ValueError):
            st.growth(growth_fault(), **{**dict(expansion=1.5, width=100.0, depth=2000.0), **kw})
