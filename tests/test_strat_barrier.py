"""``strat_trap``: the barrier zone of a facies change or a lens, its rim and the limit the report gives."""
import numpy as np
import pytest
from scipy import ndimage

from resmill.export import _build_geometry
from resmill.strat_traps import strat_trap, trap_report, zone_trap
from tests.strat_helpers import _layers


def test_a_facies_change_is_the_same_trap_with_a_barrier_zone_taking_the_sand_s_thickness():
    """The barrier zone thickens as the sand thins, so the interval keeps its thickness where the sand is, and the sand's
    trap is that of the pinch-out of the same sand; the barrier is under all of it and a rim beyond (200 m in from the
    rough edge the interval is whole)."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    kw = dict(seed=5, dip=1.1, taper_angle=0.5, area=3.4e6, aspect=2.2, warp=0.3, wander=40.0, range_m=800.0, cell=dx)
    sand_only = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], **kw)
    both = strat_trap("facies_change", x_len, y_len, 2000.0, [10.0, 8.0], barrier=True, column=12.0, **kw)
    layers = _layers(x_len, y_len, dx, [10.0, 8.0])
    _, _, zc, act = _build_geometry(layers, **both["kwargs"])
    sand = ndimage.binary_erosion(act[:, :, :5].any(axis=2), iterations=4)   # 200 m in from the sand's rough edge
    thick = zc[:, :, -1] - zc[:, :, 0]
    assert np.allclose(thick[np.repeat(np.repeat(sand, 2, axis=0), 2, axis=1)], 18.0, atol=1e-6)
    assert act[:, :, 5:].any(axis=2)[act[:, :, :5].any(axis=2)].all()      # barrier under the sand
    (net,) = trap_report(layers, both)
    (alone,) = trap_report(layers[:1], sand_only)
    assert net["crest"] == alone["crest"] and net["spill_point"] == alone["spill_point"]
    for key in ("crest_depth", "spill_depth", "closure"):
        assert net[key] == pytest.approx(alone[key])
    assert both["meta"]["net_layers"] == 1 and both["meta"]["barrier"] is True and alone["barrier_top"] is None
    assert net["barrier_top"] < net["crest_depth"] and net["limited_by"] == "barrier"
    assert net["height"] == pytest.approx(net["barrier_top"] + 12.0 - net["crest_depth"])


def test_a_barrier_zone_under_a_mound_stays_a_flat_slab_and_under_a_hung_lens_takes_the_sand_s_thickness():
    """The barrier is the last zone: under a mound it keeps its thickness (8 m) and its base is the plane of the beds, 8 m
    under the mound's base; with the lens hung from a flat top it takes what the sand loses, so that the interval keeps
    its 10 + 8 m: both where the sand is (two cells in from its edge: the barrier is a rim round it, thinning to nothing
    beyond)."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    layers = _layers(x_len, y_len, dx, [10.0, 8.0], dz=2.0)
    for mound in (True, False):
        built = strat_trap("lens", x_len, y_len, 2000.0, [10.0, 8.0], seed=2, dip=1.1, taper_angle=0.3, area=3.4e6,
                           aspect=2.2, warp=0.0, barrier=True, column=12.0, mound=mound)
        Xc, Yc, zc, act = _build_geometry(layers, **built["kwargs"])
        plane = 2000.0 + (Yc - 3000.0) * np.tan(np.radians(1.1))
        sand = act[:, :, :5].any(axis=2)
        assert act[:, :, 5:].any(axis=2)[sand].all() and (act[:, :, 5:].any(axis=2) & ~sand).any()
        inside = np.repeat(np.repeat(ndimage.binary_erosion(sand, iterations=2), 2, axis=0), 2, axis=1)
        if mound:
            assert np.allclose((zc[:, :, -1] - zc[:, :, 5])[inside], 8.0, atol=1e-6)
            assert np.allclose((zc[:, :, -1] - plane)[inside], 18.0, atol=1e-6)
        else:
            assert np.allclose((zc[:, :, -1] - zc[:, :, 0])[inside], 18.0, atol=1e-6)


def test_a_facies_change_with_tongues_a_rough_edge_and_relief_keeps_the_thickness_of_its_interval_where_the_sand_is():
    """The barrier zone is the complement of the sand wherever the sand is, whatever the edge: the interval is 10 + 8 m
    thick there, the barrier active under it and a rim beyond, and the sand has gone where the barrier alone is."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    built = strat_trap("facies_change", x_len, y_len, 2000.0, [10.0, 8.0], seed=5, barrier=True, column=15.0, dip=1.1,
                       taper_angle=0.5, area=2.0e6, aspect=1.5, tongues=[(0.8e6, 0.5)], warp=0.2, wander=150.0,
                       range_m=1200.0, hurst=0.5, floor_m=100.0, relief_sd=1.5)
    layers = _layers(x_len, y_len, dx, [10.0, 8.0])
    _, _, zc, act = _build_geometry(layers, **built["kwargs"])
    sand = act[:, :, :5].any(axis=2)
    inside = np.repeat(np.repeat(ndimage.binary_erosion(sand, iterations=10), 2, axis=0), 2, axis=1)   # 500 m in
    assert np.allclose((zc[:, :, -1] - zc[:, :, 0])[inside], 18.0, atol=1e-6) and inside.any()
    assert act[:, :, 5:].any(axis=2)[sand].all() and (act[:, :, 5:].any(axis=2) & ~sand).any() and sand.any()
    assert built["meta"]["net_layers"] == 1 and len(built["meta"]["tongues"]) == 2


COLUMN = 12.0                                     # m: the oil column the barrier holds (Berg), a typical 8-20 m


def _barrier_model(kind="facies_change", mound=None, dx=50.0, thick=(10.0, 8.0), **kw):
    """A facies change (or a lens in a barrier zone) of the P50 trap on 8 x 6 km: the built geometry and its layers."""
    x_len, y_len = 8000.0, 6000.0
    args = dict(seed=5, barrier=True, column=COLUMN, dip=1.0, taper_angle=0.5, area=3.4e6, aspect=2.2, warp=0.0)
    args.update(kw)
    if mound is not None:
        args["mound"] = mound
    built = strat_trap(kind, x_len, y_len, 2000.0, list(thick), **args)
    layers = _layers(x_len, y_len, dx, list(thick))
    _, _, zc, act = _build_geometry(layers, **built["kwargs"])
    return built, layers, zc, act


@pytest.mark.parametrize("kind,mound", [("facies_change", None), ("facies_change", True), ("lens", True), ("lens", False)])
def test_the_barrier_is_a_rim_a_quarter_of_its_column_over_the_dip_wide_round_the_sand_and_walls_beyond(kind, mound):
    """The barrier zone lies under every column of sand and out to ``rim`` beyond its edge, a quarter [J] of the most
    that leaves the oil a column: column / tan(dip) is 687 m for 12 m at 1 degree, so the rim is 172 m, to a cell. Beyond
    it the zone has thinned to nothing: columns with no active cell, walls. Before, it was a slab under the whole
    model, updip of the trap too, and an initialisation by contacts put 84-100 % of the oil in it."""
    dx = 50.0
    built, layers, zc, act = _barrier_model(kind, mound, dx=dx)
    rim = 0.25 * COLUMN / np.tan(np.radians(1.0))
    assert built["meta"]["rim_m"] == pytest.approx(rim) and built["meta"]["column"] == COLUMN
    n_sand = layers[0].nz
    sand, barrier = act[:, :, :n_sand].any(axis=2), act[:, :, n_sand:].any(axis=2)
    assert (barrier | ~sand).all() and sand.any()                          # the barrier is under all the sand
    away = ndimage.distance_transform_edt(~sand) * dx                       # m from the sand, over the barrier's columns
    assert away[barrier].max() == pytest.approx(rim, abs=2.0 * dx) and away[barrier & ~sand].min() > 0.0
    assert (~act.any(axis=2)).sum() > 0.1 * sand.size and not barrier[away > rim + 2.0 * dx].any()   # walls beyond


def test_a_barrier_of_a_facies_change_takes_the_thickness_the_sand_loses_inside_the_rim():
    """The interval keeps its 10 + 8 m where the sand is (two cells in from its edge, where the rim's factor is 1); the
    rim thins to nothing over its width, and beyond it the interval has no thickness at all."""
    built, layers, zc, act = _barrier_model()
    thick = zc[:, :, -1] - zc[:, :, 0]
    sand = ndimage.binary_erosion(act[:, :, :5].any(axis=2), iterations=2)
    inside = np.repeat(np.repeat(sand, 2, axis=0), 2, axis=1)                # the doubled corner grid
    assert np.allclose(thick[inside], 18.0, atol=1e-6) and thick[~inside].max() <= 18.0 + 1e-6
    assert (thick[~inside] > 0.0).any() and (thick[~inside] == 0.0).any() and (thick[~inside] < 17.0).any()


def _shallowest(zc, act, k):
    """The top of the shallowest active cell of layers ``k`` (the mean of its four corners), from the geometry alone."""
    tops = 0.25 * (zc[0::2, 0::2, :-1] + zc[1::2, 0::2, :-1] + zc[0::2, 1::2, :-1] + zc[1::2, 1::2, :-1])
    return tops[:, :, k][act[:, :, k] > 0].min()


@pytest.mark.parametrize("kind,mound", [("facies_change", None), ("facies_change", True), ("lens", True), ("lens", False)])
def test_the_report_gives_a_limit_that_keeps_the_barrier_dry_and_the_rim_leaves_most_of_the_column(kind, mound):
    """The limit is the smaller of the spill and the shallowest top of the barrier plus its column (below the crest's
    own reach if the barrier's top is shallower than the crest), so that a contact (the sand's oil-water contact) at the
    limit puts oil in no barrier cell: the barrier's cells lie at or below contact - column, to its shallowest top,
    which the test finds from the geometry alone. The crest is no more than a quarter of the column (and a cell's rise)
    below the barrier's top: three quarters of the column are left."""
    dx = 50.0
    built, layers, zc, act = _barrier_model(kind, mound, dx=dx)
    (trap,) = trap_report(layers, built)
    n_sand = layers[0].nz
    top = _shallowest(zc, act, slice(n_sand, None))
    assert trap["barrier_top"] == pytest.approx(top)
    reach = min(trap["crest_depth"], top)
    assert trap["limit_depth"] == pytest.approx(min(reach + COLUMN, trap["spill_depth"] or np.inf)
                                                if trap["spill_depth"] is not None else reach + COLUMN)
    assert trap["limited_by"] == "barrier" and trap["height"] > 0.5 * COLUMN
    cell = np.tan(np.radians(1.0)) * dx
    assert trap["crest_depth"] - top <= 0.25 * COLUMN + 2.0 * cell
    wet = lambda contact: (_shallowest(zc, act, slice(n_sand, None)) < contact - COLUMN)   # the shallowest barrier cell
    assert not wet(trap["limit_depth"] - 1e-6) and wet(trap["limit_depth"] + 1.0)
    free = [t for t in zone_trap(zc, act, dx, dx, k=slice(0, n_sand)) if t["crest"] == trap["crest"]][0]
    assert trap["height"] < free["height"] and free["limited_by"] != "barrier"             # the barrier lets it fill less:
    assert 0.0 < trap["area"] < free["area"] and not (trap["mask"] & ~free["mask"]).any()   # a smaller part of the same trap


def test_a_barrier_needs_the_column_it_holds_and_a_column_needs_a_barrier():
    args = dict(x_len=8000.0, y_len=6000.0, top=2000.0, seed=1, dip=1.0, taper_angle=0.5)
    with pytest.raises(ValueError, match="column"):
        strat_trap("facies_change", thicknesses=[10.0, 8.0], barrier=True, **args)
    with pytest.raises(ValueError, match="column"):
        strat_trap("pinchout", thicknesses=[10.0], column=10.0, **args)
    with pytest.raises(ValueError, match="column"):
        strat_trap("facies_change", thicknesses=[10.0, 8.0], barrier=True, column=0.0, **args)
