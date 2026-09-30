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
