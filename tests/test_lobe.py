import numpy as np
import pytest
from resmill.layers.lobe import LobeLayer, _cap_cells, _compensation_weights, _stamp_mud


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
# opt-in rock by facies: calibrated sand, mud rock and kv/kh of each facies
# --------------------------------------------------------------------------

MUD = {"poro": 0.10, "log10_perm": -3.0}


def _rocky(facies_props, seed=2, **extra):
    return _stamped(seed, nx=40, ny=30, nz=16, r_ave=300.0, facies_props=facies_props, **extra)


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


def test_the_default_lobe_has_no_mud_rock_no_kvkh_no_sand_fraction_and_no_barrier():
    layer = _stamped(1, nx=40, ny=30, nz=16)
    assert np.all(np.asarray(layer.perm_mat)[np.asarray(layer.active) == 0] == 0.0)
    assert layer.kvkh_mat is None and layer.mult_z is None and not hasattr(layer, "sand_fraction")


def test_facies_props_refuses_what_a_lobe_does_not_use():
    for bad in ({0: {"poro": 0.1}}, {3: {"poro": 0.25}}, {-1: {"porosity": 0.1}}, {2: {"ntg_crest": 0.9}}):
        with pytest.raises(ValueError, match="facies_props"):
            _rocky({-1: dict(MUD), **bad}, ntg=0.5)


# --------------------------------------------------------------------------
# opt-in interlobe mud: a cap on every stamp, thick at the margin, eroded where the next stamp is thick
# --------------------------------------------------------------------------

def _stack(*surfaces):
    """A stack of cumulative surfaces (cells), ``(ny=1, nx)`` each, from the empty floor."""
    return [np.zeros((1, len(surfaces[0])))] + [np.array([s], dtype=float) for s in surfaces]


def test_a_stamp_is_capped_with_mud_and_the_next_stamp_cuts_the_cap():
    """dz 1 m, mud cap M = 1 m, the next stamp cuts a fifth of its own thickness. Stamp 1 is 4 m thick on both columns (cap [3, 4]);
    stamp 2 adds 3 m to column 0 only: it cuts 0.6 m off the top of that cap (mud [3, 3.4] stays: 0.4 m in cell 3) and lays a cap
    of its own [6, 7]; column 1 keeps its whole cap, 1 m in cell 3."""
    mud, moment, amalgamated = _stamp_mud(_stack([4.0, 4.0], [7.0, 4.0]), nz=8, dz=1.0, mud_cap=1.0, erosion=0.2, floor=0.0)
    expected = np.zeros((8, 1, 2))
    expected[3, 0, :] = [0.4, 1.0]
    expected[6, 0, 0] = 1.0
    np.testing.assert_allclose(mud, expected, atol=1e-12)
    assert amalgamated == 0.0                   # one contact (column 0), the cap survived


def test_the_moment_of_the_mud_is_where_it_sits_in_the_cell():
    """The first moment about the cell's base is the mud's thickness times the height of its middle in cell heights: the
    0.4 m left of the cap [3, 4] (mud [3, 3.4], middle 3.2) is 0.4 x 0.2 in cell 3, the whole cap [3, 4] of column 1 1.0 x 0.5."""
    mud, moment, _ = _stamp_mud(_stack([4.0, 4.0], [7.0, 4.0]), nz=8, dz=1.0, mud_cap=1.0, erosion=0.2, floor=0.0)
    assert moment[3, 0, 0] == pytest.approx(0.4 * 0.2) and moment[3, 0, 1] == pytest.approx(0.5) and moment[6, 0, 0] == pytest.approx(0.5)
    mud, moment, _ = _stamp_mud(_stack([4.0]), nz=8, dz=2.0, mud_cap=1.5, erosion=0.0, floor=0.0)    # cells of 2 m: the stamp is 8 m thick, its cap [6.5, 8] m is in cell 3 [6, 8]
    assert mud[3, 0, 0] == pytest.approx(1.5) and moment[3, 0, 0] == pytest.approx(1.5 * ((6.5 + 8.0) / 2 - 6.0) / 2.0)


def test_a_cap_cut_through_is_amalgamation():
    """Erosion 0.5: the 3 m stamp cuts 1.5 m, more than the 1 m cap, so the sand of the two stamps touches: the cap of
    stamp 1 is gone from column 0 (only stamp 2's own cap, 1 m in cell 6, is left) and the one contact is amalgamated."""
    mud, _, amalgamated = _stamp_mud(_stack([4.0, 4.0], [7.0, 4.0]), nz=8, dz=1.0, mud_cap=1.0, erosion=0.5, floor=0.0)
    assert mud[3, 0, 0] == 0.0 and mud[3, 0, 1] == 1.0 and mud[6, 0, 0] == 1.0
    assert amalgamated == 1.0


def test_a_thin_deposit_between_two_stamps_does_not_shield_the_cap_under_it():
    """The scour of a younger stamp reaches down through whatever lies under it. Column 0 has a 0.01 m sliver of a third
    stamp between stamp 1 (cap [3, 4]) and a 3 m stamp that scours 1.5 m (erosion 0.5): the reach, 4.01 - 1.5 = 2.51, passes
    the cap, so it is gone exactly as in column 1, where nothing lies between (reach 4 - 1.5 = 2.5). The edge of a stamp is
    such a sliver, so a cap that survived it would end in a straight line along every stamp's footprint."""
    mud, _, amalgamated = _stamp_mud(_stack([4.0, 4.0], [4.01, 4.0], [7.01, 7.0]), nz=8, dz=1.0, mud_cap=1.0, erosion=0.5, floor=0.0)
    assert mud[3, 0, 0] == 0.0 and mud[3, 0, 1] == 0.0                 # the caps of stamp 1 are gone in both columns
    assert mud[6, 0, 0] == pytest.approx(0.99) and mud[7, 0, 0] == pytest.approx(0.01) and mud[6, 0, 1] == pytest.approx(1.0)   # the top cap, 0.01 m higher in column 0
    assert amalgamated == 1.0                  # two caps of at least a tenth of a cell met a later stamp, and both were cut


def test_the_cap_of_a_thin_stamp_keeps_the_fringes_sand_fraction():
    """With a fringe sand fraction of 0.3 a stamp is never more than 0.7 mud: 4 m thick it is capped at M = 1 m, 1 m thin
    (a margin) 0.7 m of it is mud and 0.3 m sand, whatever M is: here M = 5 m, so the 4 m stamp is 2.8 m mud."""
    mud, _, _ = _stamp_mud(_stack([1.0, 4.0]), nz=4, dz=1.0, mud_cap=5.0, erosion=0.0, floor=0.3)
    assert mud[:, 0, 0].sum() == pytest.approx(0.7) and mud[:, 0, 1].sum() == pytest.approx(2.8)
    assert mud[0, 0, 0] == pytest.approx(0.7) and mud[1, 0, 1] == pytest.approx(0.8) and mud[2, 0, 1] == pytest.approx(1.0)


def test_mud_above_the_top_of_the_layer_is_cut_off():
    mud, _, _ = _stamp_mud(_stack([6.0]), nz=4, dz=1.0, mud_cap=2.0, erosion=0.0, floor=0.0)   # cap [4, 6]: above the layer
    assert mud.sum() == 0.0


def _column(mud_by_cell, moment_by_cell=None):
    """``mud`` and ``moment`` of one column of cells of 1 m (``(1, 1, nz)``, k up) from the mud in each cell; the moment puts
    the mud in the middle of the cell unless given (cell heights)."""
    mud = np.array(mud_by_cell, dtype=float)[None, None, :]
    moment = mud * (0.5 if moment_by_cell is None else np.array(moment_by_cell, dtype=float)[None, None, :])
    return mud, moment


def test_every_cell_is_one_facies_by_the_share_of_it_that_is_sand():
    """Sand from half sand up, the heterolithic fringe from a fifth to a half, mud below, in cells of 1 m: mud of 0.1, 0.5,
    0.55, 0.79, 0.81 and 1 m is sand, sand (exactly half), fringe, fringe, mud and mud."""
    mud, moment = _column([0.1, 0.5, 0.55, 0.79, 0.81, 1.0])
    s, facies, _ = _cap_cells(mud, moment, 1.0)
    np.testing.assert_allclose(s[0, 0], [0.9, 0.5, 0.45, 0.21, 0.19, 0.0])
    assert facies[0, 0].tolist() == [3, 3, 2, 2, -1, -1]


def test_a_cap_of_one_and_a_half_cells_gives_mud_cells():
    """A cap [2.5, 4] in cells of 1 m is 0.5 m in cell 2 and the whole of cell 3: cell 3 is mud, and the 0.5 m left in the
    sand cell 2 (not a cell) is a thin cap on the face next to the mud cell."""
    mud, moment = _column([0, 0, 0.5, 1.0, 0], [0, 0, 0.75, 0.5, 0])
    _, facies, thin = _cap_cells(mud, moment, 1.0)
    assert facies[0, 0].tolist() == [3, 3, 3, -1, 3]
    assert thin[0, 0].tolist() == [0, 0, 0, 0.5, 0]       # cell 2's mud sits in its upper half: the face between cells 2 and 3, the lower face of cell 3


def test_a_cap_thinner_than_half_a_cell_is_a_barrier_on_the_face_nearest_its_middle():
    """In cells of 1 m a cap [3.7, 4.0] (0.3 m, middle at 0.85 of cell 3) is on the upper face of cell 3, the lower face of
    cell 4; a cap [5.05, 5.35] (middle at 0.2 of cell 5) on the lower face of cell 5; a cap on the layer's base or top is dropped.
    Neither is a cell."""
    mud, moment = _column([0.0, 0.1, 0, 0.3, 0, 0.3, 0.0, 0.2], [0, 0.1, 0, 0.85, 0, 0.2, 0, 0.9])
    _, facies, thin = _cap_cells(mud, moment, 1.0)
    assert (facies == 3).all()
    # cell 1: middle 0.1, lower face of cell 1; cell 3: 0.85, upper face (lower face of cell 4); cell 5: 0.2, lower face of cell 5;
    # cell 7: middle 0.9, upper face of the layer's top cell, dropped
    np.testing.assert_allclose(thin[0, 0], [0, 0.1, 0, 0, 0.3, 0.3, 0, 0])


def test_dust_of_the_overlap_arithmetic_is_not_a_cap():
    mud, moment = _column([0, 1e-9, 0, 0])
    assert not _cap_cells(mud, moment, 1.0)[2].any()


FRINGE = {"ntg_floor": 0.1, "poro": 0.18, "log10_perm": 1.0, "kvkh": 0.002}


def _interlobe(seed=2, ntg=0.65, **extra):
    props = {-1: dict(MUD, kvkh=0.1), 2: dict(FRINGE), 3: {"kvkh": 0.6}}
    return _rocky(props, seed=seed, ntg=ntg, interlobe_erosion=0.3, **extra)


def test_the_interlobe_net_share_is_the_asked_net_to_gross_and_its_mud_thickens_as_it_falls():
    """The cap thickness M is found so that the share of net cells is the asked net-to-gross; a sandier layer has a thinner
    mud cap, more of its contacts amalgamated (the erosion is 0.2 of the younger stamp's thickness whatever the layer's
    sand) and more of its cells sand-dominated."""
    layer = _stamped(3, nx=60, ny=40, nz=24, r_ave=400.0, upthinning=False)
    out = {}
    for ntg in (0.4, 0.55, 0.7, 0.9):
        mud, moment, outcome = layer._interlobe_cells(ntg=ntg, erosion=0.2, floor=0.1, fringe_net=False)
        s = 1.0 - mud / layer.dz
        assert (s >= 0.5).mean() == pytest.approx(ntg, abs=3e-3) and outcome["net_cells"] == pytest.approx(ntg, abs=3e-3)
        assert mud.shape == moment.shape == (60, 40, 24) and not outcome["aim_missed"]
        assert outcome["sand_fraction"] == pytest.approx(1.0 - mud.mean() / layer.dz)
        out[ntg] = outcome
    order = (0.4, 0.55, 0.7, 0.9)
    mud = [out[n]["mud_thickness_m"] for n in order]
    amalgamated = [out[n]["amalgamated"] for n in order]
    assert mud == sorted(mud, reverse=True) and mud[0] > 2.0 * mud[-1]
    assert all(b >= a - 0.01 for a, b in zip(amalgamated, amalgamated[1:])) and amalgamated[-1] > amalgamated[0]


def test_a_fringe_above_1_mD_is_net_rock_and_counts_in_the_net_share():
    """With the fringe net (its permeability above 1 mD) the cells down to a fifth sand are net, so the same cap holds a
    larger net share than with sand alone counted, and a net share asked for is met with a thicker cap."""
    layer = _stamped(3, nx=60, ny=40, nz=24, r_ave=400.0, upthinning=False)
    sand_only = layer._interlobe_cells(ntg=0.7, erosion=0.2, floor=0.1, fringe_net=False)[2]
    with_fringe = layer._interlobe_cells(ntg=0.7, erosion=0.2, floor=0.1, fringe_net=True)[2]
    assert with_fringe["net_cells"] == pytest.approx(0.7, abs=3e-3) and sand_only["net_cells"] == pytest.approx(0.7, abs=3e-3)
    assert with_fringe["mud_thickness_m"] > 1.5 * sand_only["mud_thickness_m"]


def test_an_interlobe_net_to_gross_below_what_the_stack_can_hold_is_warned_of_and_the_least_is_built():
    """A fringe sand fraction of 0.3 holds sand in every stamp, so an asked 0.001 is out of reach: the thickest mud cap the
    stack takes is built and the net share it leaves reported, with the miss flagged; a net-to-gross outside 0-1 is an error."""
    layer = _stamped(3, nx=60, ny=40, nz=24, r_ave=400.0)
    with pytest.warns(UserWarning, match="net-to-gross"):
        mud, moment, outcome = layer._interlobe_cells(ntg=0.001, erosion=0.0, floor=0.3, fringe_net=False)
    assert outcome["aim_missed"] and outcome["net_cells"] > 0.1
    with pytest.raises(ValueError, match="between 0 and 1"):
        layer._interlobe_cells(ntg=1.0, erosion=0.0, floor=0.3, fringe_net=False)


def test_interlobe_cells_are_one_facies_each_with_the_rock_of_it_and_nothing_mixed():
    """With no spread inside a facies every cell of one facies has the same rock, and there are no other values: sand
    10^2.5 mD and 0.25, the fringe 10 mD and 0.18, the mud 1e-3 mD and 0.10; kv/kh 0.6, 0.002 and 0.1. The fringe is
    net (above 1 mD), so the cells above 1 mD are the sand and the fringe cells and nothing else."""
    layer = _interlobe(ntg=0.8, poro_std=0.0, perm_std=0.0)
    facies, perm, poro = np.asarray(layer.facies), np.asarray(layer.perm_mat), np.asarray(layer.poro_mat)
    assert set(np.unique(facies)) == {-1, 2, 3}
    for code, (k, phi, kvkh) in {3: (10 ** 2.5, 0.25, 0.6), 2: (10.0, 0.18, 0.002), -1: (1e-3, 0.10, 0.1)}.items():
        mine = facies == code
        assert np.allclose(perm[mine], k, rtol=1e-4) and np.allclose(poro[mine], phi, atol=1e-5)
        assert np.allclose(np.asarray(layer.kvkh_mat)[mine], kvkh, rtol=1e-5)
    assert np.array_equal(perm > 1.0, facies >= 2) and np.array_equal(np.asarray(layer.active) == 1, facies == 3)
    s = np.asarray(layer.sand_fraction)
    assert np.array_equal(facies == 3, s >= 0.5) and np.array_equal(facies == 2, (s >= 0.2) & (s < 0.5))


def _lobes_at(dz, ntg=0.65, floor=0.15, z_len=24.0):
    np.random.seed(5)
    layer = LobeLayer(nx=60, ny=40, nz=int(round(z_len / dz)), x_len=3000.0, y_len=2000.0, z_len=z_len, top_depth=1000)
    layer.create_geology(**{**STAMPS, "ntg": ntg, "dh_ave": 4.0, "dh_std": 0.8, "r_ave": 400.0, "r_std": 80.0, "upthinning": False,
                            "interlobe_erosion": 0.2, "facies_props": {-1: dict(MUD, kvkh=0.1), 2: dict(FRINGE, ntg_floor=floor, log10_perm_sd=0.1),
                                                                      3: {"kvkh": 0.6}}})
    return layer


def test_interlobe_net_to_gross_is_the_share_of_cells_above_1_mD_where_the_layers_resolve_the_lobes():
    """The share of cells above 1 mD is the net-to-gross asked, with 0.5 m or 1 m cells on lobes 4 m thick at the axis
    (0.65 asked: the bisection's, and the fringe's spread, are a few thousandths), so the label is what the deck has."""
    for dz in (0.5, 1.0):
        layer = _lobes_at(dz)
        net = float((np.asarray(layer.perm_mat) > 1.0).mean())
        assert net == pytest.approx(0.65, abs=0.02), (dz, net)
        assert layer.interlobe["net_cells"] == pytest.approx(0.65, abs=0.01) and not layer.interlobe["aim_missed"]


def test_interlobe_net_to_gross_the_grid_cannot_resolve_is_flagged_and_the_label_is_what_the_deck_has():
    """Cells 3 m thick on lobes 4 m thick at the axis (2 m on average) hold a sand and mud of each lobe in one cell, a
    heterolithic fringe cell where the sand share is a fifth to a half, and the fringe is net (above 1 mD): 0.25 asked is out of
    reach, the least cap is built with a warning, the miss is flagged and the cells above 1 mD are the share the deck has."""
    with pytest.warns(UserWarning, match="net-to-gross"):
        layer = _lobes_at(3.0, ntg=0.25, floor=0.1)
    assert layer.interlobe["aim_missed"]
    net = float((np.asarray(layer.perm_mat) > 1.0).mean())
    assert net > 0.3 and net == pytest.approx(layer.interlobe["net_cells"], abs=0.03)


def test_a_thin_cap_is_a_multz_face_equal_to_the_drape_formula(tmp_path, monkeypatch):
    """A stack with one cap 0.3 cells thick, [3.7, 4.0] in cells of 1.5 m ... here cells of 1 m: the sand cells 3 and 4 above
    it and below it, a MULTZ of the drape's thin barrier on the face between them: [1 + (t / h)(ks / kd - 1)]^-1 with
    t = 0.3 m, h = 1 m, kd the mud's 1e-3 mD and ks the harmonic mean of the PERMZ of the two cells; MULTZ is 1 on every other
    face, in the layer's array (k up) and in the GRDECL (K down, the lower face of a cell is its + z face)."""
    from resmill.export import to_grdecl
    from resmill.layers.drapes import thin_barrier

    mud = np.zeros((3, 3, 8))
    mud[:, :, 3] = 0.3
    moment = np.zeros_like(mud)
    moment[:, :, 3] = 0.3 * 0.85
    outcome = dict(mud_thickness_m=0.3, amalgamated=0.0, net_cells=1.0, sand_fraction=0.96, aim_missed=False)
    monkeypatch.setattr(LobeLayer, "_interlobe_cells", lambda self, *a: (mud, moment, outcome))
    np.random.seed(1)
    layer = LobeLayer(nx=3, ny=3, nz=8, x_len=150.0, y_len=150.0, z_len=8.0, top_depth=1000)
    layer.create_geology(**{**STAMPS, "ntg": 0.9, "interlobe_erosion": 0.0,
                            "facies_props": {-1: dict(MUD, kvkh=0.1), 2: dict(FRINGE), 3: {"kvkh": 0.6}}})
    kz = np.asarray(layer.perm_mat) * np.asarray(layer.kvkh_mat)
    ks = 2.0 * kz[:, :, 3] * kz[:, :, 4] / (kz[:, :, 3] + kz[:, :, 4])
    expected = thin_barrier(0.3, 1.0, ks, 10 ** -3.0)
    mult = np.asarray(layer.mult_z)
    assert mult.shape == (3, 3, 8) and np.allclose(mult[:, :, 4], expected, rtol=1e-5) and ((expected > 1e-9) & (expected < 1e-3)).all()
    assert (np.delete(mult, 4, axis=2) == 1.0).all()
    to_grdecl(layer, tmp_path / "m.grdecl")
    text = (tmp_path / "m.grdecl").read_text()
    assert "MULTX" in text and "MULTY" in text and "MULTZ" in text
    words = text.split()
    start = words.index("MULTZ") + 1
    values = []
    for w in words[start:words.index("/", start)]:
        n, star, v = w.partition("*")
        values += [float(v)] * int(n) if star else [float(n)]
    assert len(values) == 72
    k_down = np.array(values).reshape((3, 3, 8), order="F")                              # I fastest, then J, then K down
    assert np.allclose(k_down[:, :, 8 - 1 - 4], expected, rtol=1e-5)                     # k up 4 is K-down index 3
    assert (k_down != 1.0).sum() == 9


def test_a_stack_without_thin_caps_has_no_multz_and_one_with_them_has_a_barrier_share():
    """Cells of 1.5 m on stamps 3 m thick: the thin caps of the sand cells are faces with a multiplier under 1 (none above), and
    ``interlobe['barrier_faces']`` is the share of faces under a half."""
    layer = _interlobe(ntg=0.9)
    mult = np.asarray(layer.mult_z)
    assert mult.max() == 1.0 and 0.0 < (mult < 1.0).mean() < 0.3 and (mult[..., 0] == 1.0).all()
    assert layer.interlobe["barrier_faces"] == pytest.approx((mult[..., 1:] < 0.5).mean())


def test_interlobe_needs_the_rock_by_facies_and_the_fringes_rock():
    with pytest.raises(ValueError, match="facies_props"):
        _stamped(1, nx=40, ny=30, nz=16, interlobe_erosion=0.3)
    with pytest.raises(ValueError, match="fringe"):
        _rocky({-1: dict(MUD), 2: {"ntg_floor": 0.3, "kvkh": 0.002}, 3: {"kvkh": 0.6}}, ntg=0.65, interlobe_erosion=0.3)


def test_the_decay_of_a_thin_stamp_is_clipped_when_asked():
    """A stamp thinner than a couple of cells has cells whose lower face is below its base, and the porosity decay
    (surface top - lower face) / thickness exceeds 1 there: a ring of porosity above the design maximum of 0.35, in
    plan view a small bright ring. ``clip_decay`` holds it to 1."""
    peaks = {}
    for clip in (False, True):
        np.random.seed(2)
        layer = LobeLayer(nx=40, ny=30, nz=16, x_len=2000.0, y_len=1500.0, z_len=24.0, top_depth=1000)
        _, allporo, _ = layer._lobemodeling(dh_ave=3.0, dh_std=0.6, r_ave=300.0, r_std=60.0, asp=1.7, azimuth=0.0,
                                            azimuth_std=10.0, upthinning=False, compensation_scale=0.1, clip_decay=clip)
        peaks[clip] = float(allporo[-1].max())
    assert peaks[False] > 0.36 and peaks[True] <= 0.35 + 1e-9
