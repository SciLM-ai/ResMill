import numpy as np
import pytest
from resmill.layers.lobe import LobeLayer, _compensation_weights


def test_lobe_shapes():
    layer = LobeLayer(nx=10, ny=10, nz=5, x_len=100, y_len=100, z_len=10, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.0, poro_std=0.03, perm_std=0.3, ntg=0.7)
    assert layer.poro_mat.shape == (10, 10, 5)
    assert layer.perm_mat.shape == (10, 10, 5)
    assert layer.active.shape == (10, 10, 5)


def test_lobe_has_facies():
    layer = LobeLayer(nx=10, ny=10, nz=5, x_len=100, y_len=100, z_len=10, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.0, poro_std=0.03, perm_std=0.3, ntg=0.7)
    assert hasattr(layer, 'facies')
    assert layer.facies.shape == (10, 10, 5)


def test_lobe_poro_non_negative():
    layer = LobeLayer(nx=10, ny=10, nz=5, x_len=100, y_len=100, z_len=10, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.0, poro_std=0.03, perm_std=0.3, ntg=0.7)
    assert np.all(layer.poro_mat >= 0)
    assert np.all(layer.perm_mat >= 0)


def test_lobe_active_is_binary():
    layer = LobeLayer(nx=10, ny=10, nz=5, x_len=100, y_len=100, z_len=10, top_depth=1000)
    layer.create_geology(poro_ave=0.2, perm_ave=1.0, poro_std=0.03, perm_std=0.3, ntg=0.7)
    assert set(np.unique(layer.active)).issubset({0, 1})


# --------------------------------------------------------------------------
# opt-in compensation strength: the weight of a stamp centre in stamp thicknesses
# --------------------------------------------------------------------------

STAMPS = dict(poro_ave=0.25, perm_ave=2.5, poro_std=0.03, perm_std=0.2, ntg=0.5, dh_ave=3.0, dh_std=0.6,
              r_ave=300.0, r_std=60.0, asp=1.7, azimuth=0.0, upthinning=False)


def _stamped(seed=1, nx=60, ny=40, nz=20, **extra):
    np.random.seed(seed)
    layer = LobeLayer(nx=nx, ny=ny, nz=nz, x_len=nx * 50.0, y_len=ny * 50.0, z_len=nz * 1.5, top_depth=1000)
    layer.create_geology(**{**STAMPS, **extra})
    return layer


def _compensation_index(surfaces):
    """Straub et al. 2009: sigma_ss(T) is the spatial standard deviation of the thickness gained per event
    over T events and falls as T^-kappa (0.5 for random stacking, 1 for perfect compensation); kappa is
    minus the slope of log sigma_ss against log T."""
    s = np.asarray(surfaces)
    steps = np.unique(np.round(np.logspace(0, np.log10(len(s) // 2), 10)).astype(int))
    sigma = [np.mean([w.std() for w in (s[t:] - s[:-t])[::max(1, (len(s) - t) // 12)] / t]) for t in steps]
    return -np.polyfit(np.log(steps), np.log(sigma), 1)[0]


def test_compensation_weights_fall_with_the_relief_in_stamp_thicknesses():
    """Columns 0, 1, 2 and 0.5 cells of 2 m high, stamps 4 m thick, scale 0.5: the deficits are 0, 2, 4 and
    1 m and the weights exp(-deficit / 2 m) = 1, 0.3679, 0.1353 and 0.6065 (sum 2.1097)."""
    surface = np.array([[0.0, 1.0], [2.0, 0.5]]) + 1e-6
    p = _compensation_weights(surface, dz=2.0, dh_ave=4.0, scale=0.5)
    np.testing.assert_allclose(p, [[0.4740, 0.1744], [0.0641, 0.2875]], atol=5e-5)
    assert p.sum() == pytest.approx(1.0)


def test_a_vanishing_compensation_scale_always_picks_the_lowest_column():
    """The deterministic limit is what ResMill's weight reaches at m = 50: the same stack."""
    old, new = _stamped(m=50), _stamped(compensation_scale=1e-9)
    assert np.array_equal(old.lobe_id, new.lobe_id) and np.array_equal(old.perm_mat, new.perm_mat)


def test_a_larger_compensation_scale_stacks_the_lobes_more_randomly():
    """Random stacking has kappa 0.5 (0.55-0.6 in this estimator), a lobe always on the lowest column about
    0.8: a scale of 3 stamp thicknesses is nearly random, 0.05 nearly deterministic."""
    random_, ordered = [], []
    for seed in (1, 2, 3):
        random_.append(_compensation_index(_stamped(seed, compensation_scale=3.0).allsurface))
        ordered.append(_compensation_index(_stamped(seed, compensation_scale=0.05).allsurface))
    assert np.mean(random_) < 0.7 and np.mean(ordered) > np.mean(random_) + 0.12


def test_without_a_compensation_scale_the_exponent_m_saturates_from_two():
    """ResMill's own weight (the default, kept as it was): m = 0 is random stacking and m = 2 already puts every
    stamp on the lowest column, so m = 50 changes nothing (kappa 0.56 -> 0.83 -> 0.83 on 3 seeds in the
    research runs)."""
    kappa = {m: np.mean([_compensation_index(_stamped(seed, m=m).allsurface) for seed in (1, 2, 3)])
             for m in (0, 2, 50)}
    assert kappa[0] < 0.7 and kappa[2] > kappa[0] + 0.12 and abs(kappa[50] - kappa[2]) < 0.06
