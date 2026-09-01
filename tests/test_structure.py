import numpy as np
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
