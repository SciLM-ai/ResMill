import numpy as np
import pytest
from resmill.layers.lobe import LobeLayer, _compensation_weights, _faded_sand_fraction


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


# --------------------------------------------------------------------------
# opt-in rock by facies: calibrated sand, mud rock, kv/kh, and sand fading towards the margins
# --------------------------------------------------------------------------

MUD = {"poro": 0.10, "log10_perm": -3.0}


def _rocky(facies_props, seed=2, **extra):
    return _stamped(seed, nx=40, ny=30, nz=16, r_ave=300.0, facies_props=facies_props, **extra)


def test_the_faded_sand_fraction_is_an_exponential_of_the_cells_rank():
    """floor 0.2, crest 1, e-folding 0.3 of the ranks: the poorest cells hold 0.2 + 0.8 exp(-1/0.3) = 0.2285,
    the best ones 1, the mean 0.2 + 0.8 x 0.3 (1 - exp(-1/0.3)) = 0.43144; the mean is asked for, the e-folding found."""
    structure = np.random.default_rng(0).normal(size=(100, 50, 2))
    s = _faded_sand_fraction(structure, ntg=0.431438, floor=0.2, crest=1.0)
    assert s.mean() == pytest.approx(0.431438, abs=1e-5)
    assert s.min() == pytest.approx(0.2285, abs=2e-3) and s.max() == pytest.approx(1.0, abs=2e-3)
    ranked = s.ravel()[np.argsort(structure.ravel())]
    assert np.all(np.diff(ranked) >= 0)                                    # richer structure, more sand
    assert ranked[len(ranked) // 2] == pytest.approx(0.2 + 0.8 * np.exp(-0.5 / 0.3), abs=2e-3)   # the median cell


def test_a_faded_sand_fraction_needs_a_mean_between_its_floor_and_crest():
    structure = np.random.default_rng(1).normal(size=(10, 10, 2))
    for ntg in (0.15, 0.2, 1.0):
        with pytest.raises(ValueError, match="between"):
            _faded_sand_fraction(structure, ntg=ntg, floor=0.2, crest=1.0)


def test_calibrated_rock_keeps_the_sand_averages_at_any_net_to_gross():
    """Without facies_props ResMill standardizes over all cells before it zeroes the mud, so the sand
    averages 0.292, 0.274 and 0.256 for a requested 0.25 at ntg 0.25, 0.5 and 0.85 (research runs); with them the
    sand keeps exactly the requested porosity and log permeability and its spread, and the mud its own rock."""
    for ntg in (0.25, 0.5, 0.85):
        layer = _rocky({-1: dict(MUD)}, ntg=ntg)
        sand = np.asarray(layer.active) == 1
        assert sand.mean() == pytest.approx(ntg, abs=0.01)
        assert np.asarray(layer.poro_mat)[sand].mean() == pytest.approx(0.25, abs=1e-6)
        assert np.log10(np.asarray(layer.perm_mat)[sand]).mean() == pytest.approx(2.5, abs=1e-6)
        assert np.asarray(layer.poro_mat)[sand].std() == pytest.approx(0.03, rel=0.05)
        assert np.allclose(np.asarray(layer.poro_mat)[~sand], 0.10)
        assert np.allclose(np.log10(np.asarray(layer.perm_mat)[~sand]), -3.0)


def test_mud_spreads_are_opt_in_log_permeability_and_relative_porosity():
    layer = _rocky({-1: dict(MUD, log10_perm_sd=0.5, poro_sd=0.2)})
    mud = np.asarray(layer.active) == 0
    assert np.log10(np.asarray(layer.perm_mat)[mud]).std() == pytest.approx(0.5, rel=0.15)
    assert np.log(np.asarray(layer.poro_mat)[mud] / 0.10).std() == pytest.approx(0.2, rel=0.15)


def test_kvkh_per_facies_is_written_as_the_layers_vertical_ratio(tmp_path):
    """Sand 0.6 and mud 0.02 as kv/kh per cell; the export writes PERMZ = kvkh x PERMX. Without any kvkh the
    layer has none and PERMZ stays kzkx x PERMX."""
    from resmill.export import to_grdecl
    layer = _rocky({-1: dict(MUD, kvkh=0.02), 3: {"kvkh": 0.6}})
    sand = np.asarray(layer.active) == 1
    kvkh = np.asarray(layer.kvkh_mat)
    assert np.allclose(kvkh[sand], 0.6) and np.allclose(kvkh[~sand], 0.02)
    to_grdecl(layer, tmp_path / "m.grdecl")
    text = (tmp_path / "m.grdecl").read_text().split()
    values = {key: np.array([float(w) for w in text[text.index(key) + 1:text.index("/", text.index(key))]])
              for key in ("PERMX", "PERMZ")}
    assert np.allclose(values["PERMZ"] / values["PERMX"], kvkh[:, :, ::-1].ravel(order="F"), rtol=1e-4)
    assert _rocky({-1: dict(MUD)}).kvkh_mat is None


FRINGE = {2: {"ntg_floor": 0.3, "kvkh": 0.002}, 3: {"kvkh": 0.6}}


def test_sand_fading_has_no_mud_and_keeps_the_mean_sand_fraction():
    """Binary lobes leave a mud lattice (cells near 1e-3 mD); fading each cell's sand between 0.3 and 1 leaves none:
    a cell holds at least 0.3 of the sand rock, 0.3 x 10^1.5 = 9.5 mD at the least, and the zone's mean sand
    fraction is the requested ntg."""
    binary = _rocky({-1: dict(MUD)}, ntg=0.55)
    faded = _rocky({-1: dict(MUD), **FRINGE}, ntg=0.55)
    assert np.asarray(binary.perm_mat).min() < 1e-2
    s = np.asarray(faded.sand_fraction)
    assert s.mean() == pytest.approx(0.55, abs=1e-3) and s.min() >= 0.3 and s.max() <= 1.0
    assert np.asarray(faded.perm_mat).min() > 0.3 * 10 ** 1.5


def test_a_faded_cell_mixes_its_sand_and_mud_arithmetically():
    """With no spread in the sand (porosity 0.25, 10^2.5 mD) a cell of sand fraction s holds
    s x 0.25 + (1 - s) x 0.10 porosity and s x 316.2 + (1 - s) x 1e-3 mD."""
    layer = _rocky({-1: dict(MUD), **FRINGE}, ntg=0.55, poro_std=0.0, perm_std=0.0)
    s = np.asarray(layer.sand_fraction, dtype=float)
    assert np.allclose(layer.poro_mat, s * 0.25 + (1 - s) * 0.10, atol=1e-6)
    assert np.allclose(layer.perm_mat, s * 10 ** 2.5 + (1 - s) * 10 ** -3.0, rtol=1e-5)


def test_faded_kvkh_runs_log_linearly_from_the_fringe_to_the_sand():
    """kv/kh 0.002 at the floor (0.3) and 0.6 in pure sand: sand fraction 0.65 is half way, 0.002 x 300^0.5."""
    layer = _rocky({-1: dict(MUD), **FRINGE}, ntg=0.55)
    s, kvkh = np.asarray(layer.sand_fraction, dtype=float), np.asarray(layer.kvkh_mat, dtype=float)
    expected = 0.002 * 300.0 ** ((s - 0.3) / 0.7)
    assert np.allclose(kvkh, expected, rtol=1e-3)
    assert kvkh.min() >= 0.002 * 0.999 and kvkh.max() <= 0.6 * 1.001


def test_faded_lobes_are_labelled_by_their_sand_fraction():
    layer = _rocky({-1: dict(MUD), **FRINGE}, ntg=0.55)
    s, facies = np.asarray(layer.sand_fraction), np.asarray(layer.facies)
    assert np.array_equal(facies == 3, s >= 0.5) and set(np.unique(facies)) == {2, 3}
    assert np.array_equal(np.asarray(layer.active) == 1, s >= 0.5)


def test_the_default_lobe_has_no_mud_rock_no_kvkh_and_no_sand_fraction():
    layer = _stamped(1, nx=40, ny=30, nz=16)
    assert np.all(np.asarray(layer.perm_mat)[np.asarray(layer.active) == 0] == 0.0)
    assert layer.kvkh_mat is None and not hasattr(layer, "sand_fraction")


def test_facies_props_refuses_what_a_lobe_does_not_use():
    for bad in ({0: {"poro": 0.1}}, {3: {"poro": 0.25}}, {-1: {"porosity": 0.1}}):
        with pytest.raises(ValueError, match="facies_props"):
            _rocky({-1: dict(MUD), **bad}, ntg=0.5)
    with pytest.raises(ValueError, match="between"):                       # a floor above the mean sand fraction
        _rocky({-1: dict(MUD), 2: {"ntg_floor": 0.6}}, ntg=0.5)
