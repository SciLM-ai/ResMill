"""``strat_trap``: irregular edges, tongues, tops (mound or bedded), relief, stagger and the realized closure."""
import numpy as np
import pytest
from scipy import ndimage

from resmill.export import _build_geometry
from resmill.layers.base import Layer
from resmill.strat_traps import strat_trap, trap_report
from tests.strat_helpers import NOSE, _fine_traps, _layers, _length


def _sand(kind, built, nz=6, x_len=8000.0, y_len=6000.0, dx=25.0, thick=6.0, top=2000.0):
    """The exported geometry of a single sand layer: the columns that are sand, their centres and the interfaces."""
    layer = Layer(int(round(x_len / dx)), int(round(y_len / dx)), nz, x_len, y_len, thick, top)
    Xc, Yc, zc, act = _build_geometry([layer], **built["kwargs"])
    xc, yc = np.meshgrid((np.arange(layer.nx) + 0.5) * dx, (np.arange(layer.ny) + 0.5) * dx, indexing="ij")
    return act.any(axis=2), xc, yc, zc, layer


TONGUES = [(1.6e6, 1.0), (0.9e6, 0.5), (0.6e6, 0.4), (0.8e6, 0.45), (0.5e6, 0.35)]


@pytest.mark.parametrize("kind", ["pinchout", "truncation", "onlap"])
@pytest.mark.parametrize("n", [1, 3, 5])
def test_the_tongues_are_as_many_as_drawn_and_each_has_the_length_and_width_it_was_drawn_with(kind, n):
    """Every tongue is a separate lobe of sand updip of the line: counted on the exported grid (the connected bodies of
    active columns more than two cells updip of the line, azimuth 0) there are as many as were drawn (the main one and
    the ``tongues`` of the caller), they stand along the line in the order of their offsets, each reaches updip of the
    line by its drawn length (two cells and 6 %) and is as wide at its root as aspect times that."""
    x_len, y_len, dx = 8000.0, 6000.0, 25.0
    built = strat_trap(kind, x_len, y_len, 2000.0, [6.0], seed=4, dip=1.5, taper_angle=0.5, area=TONGUES[0][0],
                       aspect=TONGUES[0][1], tongues=TONGUES[1:n], warp=0.0)
    sand, xc, yc, _, _ = _sand(kind, built)
    line, drawn = built["meta"]["line"], built["meta"]["tongues"]
    labels, count = ndimage.label(sand & (yc < line - 2.0 * dx))
    assert len(drawn) == n and count == n
    found = sorted(range(1, n + 1), key=lambda c: xc[labels == c].mean())
    for tongue, c in zip(sorted(drawn, key=lambda d: d["offset"]), found):
        body = labels == c
        length = line - yc[body].min() + 0.5 * dx
        assert length == pytest.approx(tongue["length"], abs=2 * dx + 0.06 * tongue["length"])
        row = body[:, np.argmin(abs(yc[0] - (line - 3.0 * dx)))]                 # the row three cells updip
        width = row.sum() * dx
        assert width == pytest.approx(tongue["aspect"] * tongue["length"], rel=0.12, abs=2 * dx)
        assert xc[body].mean() == pytest.approx(0.5 * x_len + tongue["offset"], abs=0.12 * width + 2 * dx)


def test_tongues_that_do_not_fit_are_refused_and_a_lens_a_nose_or_a_straight_edge_take_none():
    args = (8000.0, 6000.0, 2000.0, [10.0], 1)
    with pytest.raises(ValueError, match="fit"):
        strat_trap("pinchout", *args, area=3.4e6, aspect=2.2, tongues=[(3.4e6, 2.2)] * 4)
    for kind, kw in (("lens", {}), ("pinchout", dict(area=None)), ("pinchout_nose", dict(area=None, nose=NOSE))):
        with pytest.raises(ValueError, match="tongue"):
            strat_trap(kind, *args, tongues=[(1.0e6, 1.0)], **kw)


@pytest.mark.parametrize("hurst", [0.2, 0.4, 0.6])
def test_the_edge_wanders_as_rough_as_the_hurst_exponent_drawn(hurst):
    """With no tongue the edge is the line moved by a ``relief`` surface, so the first active row of each column of the
    exported grid is a profile of that surface along the line: its structure function S(lag) = mean (y(x + lag) -
    y(x))^2 rises as lag^(2 hurst) over 100-1,200 m (the exponent within 0.4 of 2 hurst, over 6 seeds: the cells and
    the Gaussian covariance of the octaves flatten it at the high end) and its rms is the ``wander`` asked for (to
    35 %). With no wander the edge is straight."""
    x_len, y_len, dx = 8000.0, 6000.0, 25.0
    lags, slopes, rms = np.array([4, 6, 8, 12, 16, 24, 32, 48]), [], []
    for seed in range(6):
        built = strat_trap("pinchout", x_len, y_len, 2000.0, [4.0], seed=seed, dip=1.0, taper_angle=0.5, area=None,
                           wander=200.0, range_m=3000.0, hurst=hurst, floor_m=dx)
        sand, _, yc, _, _ = _sand("pinchout", built, nz=1, thick=4.0)
        edge = yc[0][sand.argmax(axis=1)]
        s = [np.mean((edge[lag:] - edge[:-lag]) ** 2) for lag in lags]
        slopes.append(np.polyfit(np.log(lags * dx), np.log(s), 1)[0])
        rms.append(edge.std())
    assert np.mean(slopes) == pytest.approx(2.0 * hurst, abs=0.4)
    assert np.mean(rms) == pytest.approx(200.0, rel=0.35)
    flat = strat_trap("pinchout", x_len, y_len, 2000.0, [4.0], seed=0, dip=1.0, taper_angle=0.5, area=None)
    sand, _, yc, _, _ = _sand("pinchout", flat, nz=1, thick=4.0)
    assert np.ptp(sand.argmax(axis=1)) == 0


def test_a_rougher_hurst_exponent_gives_a_rougher_edge_whatever_the_seed():
    """Over 6 seeds the structure function slope of the edge rises with the drawn Hurst exponent: 0.2 < 0.4 < 0.6 <
    0.8, each step."""
    x_len, y_len, dx = 8000.0, 6000.0, 25.0
    lags, mean = np.array([4, 8, 16, 32]), []
    for hurst in (0.2, 0.4, 0.6, 0.8):
        slopes = []
        for seed in range(6):
            built = strat_trap("pinchout", x_len, y_len, 2000.0, [4.0], seed=seed, dip=1.0, taper_angle=0.5, area=None,
                               wander=200.0, range_m=3000.0, hurst=hurst, floor_m=dx)
            sand, _, yc, _, _ = _sand("pinchout", built, nz=1, thick=4.0)
            edge = yc[0][sand.argmax(axis=1)]
            s = [np.mean((edge[lag:] - edge[:-lag]) ** 2) for lag in lags]
            slopes.append(np.polyfit(np.log(lags * dx), np.log(s), 1)[0])
        mean.append(np.mean(slopes))
    assert np.all(np.diff(mean) > 0.05)


def test_a_lens_is_a_mound_on_a_flat_base_and_the_fill_of_a_channel_hangs_from_a_flat_top():
    """By default the base of the sand is the plane of the beds, at every column of the exported grid (the cells of
    absent sand collapse on it), and the top is a mound: as thick as the sand in the middle (10 m, to 2 %) and concave
    down along the dip axis where it is thick. With ``mound=False`` the top is the plane and the base sags below it as
    a bowl, the lens hanging from a flat top as the fill of a channel does."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    layers = _layers(x_len, y_len, dx, [10.0], dz=1.0)
    for mound in (True, False):
        built = strat_trap("lens", x_len, y_len, 2000.0, [10.0], seed=2, dip=1.1, taper_angle=0.3, area=3.4e6,
                           aspect=2.2, warp=0.0, mound=mound)
        assert built["meta"]["mound"] is mound
        Xc, Yc, zc, act = _build_geometry(layers, **built["kwargs"])
        plane = 2000.0 + (Yc - 3000.0) * np.tan(np.radians(1.1))
        thick = zc[:, :, -1] - zc[:, :, 0]
        centre = thick[len(thick) // 2, 0::2]                            # along dip through the middle, at nodes
        inner = (centre[1:-1] > 0.3 * 10.0) & (centre[:-2] > 0.3 * 10.0) & (centre[2:] > 0.3 * 10.0)
        if mound:
            assert np.allclose(zc[:, :, -1], plane + 10.0, atol=1e-6)
            assert thick.max() == pytest.approx(10.0, rel=0.02)
            assert (np.diff(centre, 2)[inner] < 1e-9).all() and inner.sum() > 5
        else:
            assert np.allclose(zc[:, :, 0], plane, atol=1e-6)
            assert 0.0 < thick.max() < 10.0 and (np.diff(centre, 2)[inner] < 1e-9).all()
            assert np.allclose(zc[:, :, -1] - plane, thick, atol=1e-6)         # the bowl is what the sand fills


def test_the_taper_thins_without_a_step_or_a_hinge_to_the_edge():
    """Across a straight edge the sand's thickness along dip, read at the nodes of the exported grid, rises from 0 to
    its full 10 m over T / tan(angle) = 1,146 m (the wedge's width, to a node), its mean slope the drawn angle; it
    leaves full thickness with no slope (under 10 % of the mean there, where a linear ramp has all of it) and ends at
    the edge at twice the mean."""
    x_len, y_len, dx = 4000.0, 6000.0, 25.0
    angle = 0.5
    built = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=1, dip=1.0, taper_angle=angle, area=None)
    _, _, _, zc, _ = _sand("pinchout", built, nz=10, x_len=x_len, y_len=y_len, thick=10.0)
    h = (zc[:, :, -1] - zc[:, :, 0])[len(zc) // 4, 0::2]                     # along dip, one column, at the nodes
    mean = np.tan(np.radians(angle))
    taper = (h > 1e-9) & (h < 10.0 - 1e-6)
    assert taper.sum() * dx == pytest.approx(10.0 / mean, abs=2 * dx)
    slope = np.diff(h) / dx
    first, last = np.nonzero(taper)[0][[0, -1]]
    assert slope[first] == pytest.approx(2.0 * mean, rel=0.1)
    assert slope[last] < 0.1 * mean and np.all(np.diff(slope[first:last + 1]) <= 1e-12)
    assert (h[last + 1] - h[first - 1]) / ((last + 2 - first) * dx) == pytest.approx(mean, rel=0.05)


def test_relief_makes_the_top_and_the_base_of_the_zone_uneven_together_and_none_leaves_them_planes():
    """``relief_sd`` is the rms of the relief of the zone's top, 2 m here (to 40 %: one surface of a few correlation
    lengths), and the base follows it, the full sand being 10 m thick still; with none the top is exactly the plane."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    flat = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=3, dip=1.0, taper_angle=0.5, area=None)
    rough = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=3, dip=1.0, taper_angle=0.5, area=None,
                       relief_sd=2.0, relief_range=2000.0, cell=dx)
    layers = _layers(x_len, y_len, dx, [10.0], dz=1.0)
    Xc, Yc, zc0, _ = _build_geometry(layers, **flat["kwargs"])
    _, _, zc, act = _build_geometry(layers, **rough["kwargs"])
    plane = 2000.0 + (Yc - 3000.0) * np.tan(np.radians(1.0))
    sheet = Yc > 4200.0                                                    # full sand, far from the line
    top, base = zc[:, :, 0][sheet] - plane[sheet], zc[:, :, -1][sheet] - plane[sheet] - 10.0
    assert np.allclose(zc0[:, :, 0][sheet], plane[sheet], atol=1e-9)
    assert top.std() == pytest.approx(2.0, rel=0.4) and top.std() > 0.5
    assert np.allclose(base, top, atol=1e-9)                              # the zone keeps its thickness
    assert np.allclose(zc[:, :, -1][sheet] - zc[:, :, 0][sheet], 10.0, atol=1e-9)


def _same_fields(a, b):
    """Two builds that hold the same fields: the structure, the surfaces and every isochore agree on a grid."""
    x, y = np.meshgrid(np.linspace(0.0, 8000.0, 41), np.linspace(0.0, 6000.0, 31), indexing="ij")
    keys = [k for k in a["kwargs"] if k not in ("isochore", "min_thickness")]
    return a["meta"] == b["meta"] and all(np.array_equal(a["kwargs"][k](x, y), b["kwargs"][k](x, y)) for k in keys) \
        and all(np.array_equal(f(x, y), g(x, y)) for f, g in zip(a["kwargs"].get("isochore", []),
                                                                  b["kwargs"].get("isochore", [])))


@pytest.mark.parametrize("kind", ["pinchout", "facies_change", "lens", "truncation", "onlap"])
def test_everything_drawn_follows_the_seed(kind):
    """The same seed gives the same tongues, edge and relief, to the bit, and another seed another one (the seeds whose
    lobes fit and whose edge leaves a trap)."""
    thick, barrier = ([10.0, 6.0], True) if kind == "facies_change" else ([10.0], False)
    kw = dict(barrier=barrier, column=15.0 if barrier else None, dip=1.4, taper_angle=0.4, area=2.0e6, aspect=1.5,
              warp=0.2, wander=120.0, range_m=1200.0, hurst=0.5, floor_m=60.0, relief_sd=1.5,
              **({} if kind == "lens" else dict(tongues=[(0.8e6, 0.5)])))
    builds = []
    for seed in range(11, 60):
        try:
            builds.append(strat_trap(kind, 8000.0, 6000.0, 2000.0, thick, seed=seed, **kw))
        except ValueError:
            continue
        if len(builds) == 2:
            break
    a, c = builds
    b = strat_trap(kind, 8000.0, 6000.0, 2000.0, thick, seed=a["meta"]["seed"], **kw)
    assert _same_fields(a, b) and not _same_fields(a, c)


def test_the_erosion_surface_of_a_truncation_has_the_relief_the_wander_maps_to():
    """A truncation's edge is where the beds meet an erosion surface with a relief of its own: the lateral wander of
    the edge R is the vertical relief tan(discordance) R of the surface (``erosion_relief_m``: 0.9 m for a wander of
    300 m at 0.3 degrees), so that the surface moves from the smooth one by that much in the strip (twice it at the
    edge, nothing where it meets the beds: the rms over the strip is within a factor 2 of it), and the sand reaches
    updip in the valleys and is cut back on the ridges by R."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    kw = dict(dip=1.2, taper_angle=0.3, area=None, range_m=1500.0, hurst=0.5, floor_m=100.0)
    smooth = strat_trap("truncation", x_len, y_len, 2000.0, [10.0], seed=7, **kw)
    rough = strat_trap("truncation", x_len, y_len, 2000.0, [10.0], seed=7, wander=300.0, **kw)
    assert smooth["meta"]["erosion_relief_m"] == 0.0
    assert strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=7, area=None, wander=300.0, cell=dx)["meta"][
        "erosion_relief_m"] is None
    assert rough["meta"]["erosion_relief_m"] == pytest.approx(300.0 * np.tan(np.radians(0.3)))
    x, y = np.meshgrid((np.arange(160) + 0.5) * dx, (np.arange(120) + 0.5) * dx, indexing="ij")
    e0, e1 = smooth["kwargs"]["erode_above"](x, y), rough["kwargs"]["erode_above"](x, y)
    bed = 2000.0 + rough["kwargs"]["structure"](x, y)
    strip = (e1 > bed + 1e-3) & (e1 < bed + 10.0 - 1e-3)
    assert strip.sum() > 1000
    assert np.std((e1 - e0)[strip]) == pytest.approx(rough["meta"]["erosion_relief_m"], rel=1.0)
    relief = rough["meta"]["erosion_relief_m"]
    assert (e1 - e0).max() > relief and (e1 - e0).min() < -relief
    first = [np.argmax(e1[i] < bed[i] + 10.0 - 1e-3) for i in range(160)]      # the edge, row by row
    assert np.ptp(first) * dx > 300.0                                       # it wanders by hundreds of metres


@pytest.mark.parametrize("kind", ["pinchout", "truncation", "onlap"])
def test_rough_edges_with_tongues_close_as_the_fine_map_of_their_geometry_says(kind):
    """With tongues, a rough edge and relief the closure is whatever the geometry has, so it is checked against the
    fine analytic map of the top (no cells, no ACTNUM): over 6 seeds the closure of the cells is within 20 % of it
    in the median, and the largest trap of the fine map is never more than 40 m off (the cells' own thread of
    columns can join or split bodies that the fine map keeps apart). Edges to the cell's own scale are not asked of
    it: ``floor_m`` is two cells."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    layers = _layers(x_len, y_len, dx, [10.0], dz=2.0)
    errors = []
    for seed in range(20, 60):
        try:
            built = strat_trap(kind, x_len, y_len, 2000.0, [10.0], seed=seed, dip=1.2, taper_angle=0.3, area=2.0e6,
                               aspect=1.2, tongues=[(1.0e6, 0.5), (0.7e6, 0.4)], warp=0.15, wander=120.0,
                               range_m=1200.0, hurst=0.6, floor_m=2 * dx, relief_sd=1.0)
        except ValueError:
            continue
        trap = trap_report(layers, built)[0]
        fine = _fine_traps(built, 2000.0, x_len, y_len)[0]
        errors.append((abs(trap["height"] - fine["height"]), fine["height"]))
        if len(errors) == 6:
            break
    assert len(errors) == 6
    assert np.median([e / h for e, h in errors]) < 0.2 and max(e for e, _ in errors) < 40.0


@pytest.mark.parametrize("kind,azimuth", [("pinchout", 0.0), ("pinchout", 90.0), ("facies_change", 0.0)])
def test_staggered_layers_end_each_farther_downdip_by_the_stagger_and_the_interval_keeps_its_thickness(kind, azimuth):
    """Sand layers that end at different places make the pinch-out interfinger in section: with three layers of 4 m and
    a stagger of 300 m the second layer first appears 300 m farther downdip than the first, and the third 600 m (to a
    cell and a half), along the line and on every tongue; with a barrier zone under them (a facies change) the
    interval is still 12 + 6 m thick where the sand is, the barrier taking what the three layers lose. No stagger
    leaves the layers ending together."""
    x_len, y_len, dx = 8000.0, 6000.0, 25.0
    barrier = kind == "facies_change"
    thick = [4.0, 4.0, 4.0] + ([6.0] if barrier else [])
    layers = _layers(x_len, y_len, dx, thick, dz=1.0)
    first = {}
    for stagger in (0.0, 300.0):
        built = strat_trap(kind, x_len, y_len, 2000.0, thick, seed=2, barrier=barrier, column=12.0 if barrier else None,
                           dip=1.5, taper_angle=0.5, area=1.5e6, aspect=1.0, warp=0.0, azimuth=azimuth,
                           stagger=stagger)
        _, _, zc, act = _build_geometry(layers, **built["kwargs"])
        # along dip the first active row of each layer in the middle column of the tongue (azimuth 0), or the middle row
        if azimuth == 0.0:
            column = act[int(round(0.5 * x_len / dx)) + int(round(built["meta"]["tongues"][0]["offset"] / dx))]
            first[stagger] = [int(np.argmax(column[:, 4 * k:4 * k + 4].any(axis=1))) * dx for k in range(3)]
        else:                                                           # dip along +x: rows are x
            row = act[:, int(round(0.5 * y_len / dx)) - int(round(built["meta"]["tongues"][0]["offset"] / dx))]
            first[stagger] = [int(np.argmax(row[:, 4 * k:4 * k + 4].any(axis=1))) * dx for k in range(3)]
        if barrier:
            sand = act[:, :, :12].any(axis=2)
            inside = np.repeat(np.repeat(ndimage.binary_erosion(sand, iterations=2), 2, axis=0), 2, axis=1)
            assert np.allclose((zc[:, :, -1] - zc[:, :, 0])[inside], 18.0, atol=1e-6)
            assert act[:, :, 12:].any(axis=2)[sand].all()
    assert np.ptp(first[0.0]) <= 1.5 * dx
    assert np.diff(first[300.0]) == pytest.approx([300.0, 300.0], abs=1.5 * dx)


def test_staggered_layers_end_farther_downdip_along_a_straight_line_too():
    x_len, y_len, dx = 8000.0, 6000.0, 25.0
    layers = _layers(x_len, y_len, dx, [4.0, 4.0], dz=1.0)
    built = strat_trap("pinchout", x_len, y_len, 2000.0, [4.0, 4.0], seed=2, dip=1.5, taper_angle=0.5, area=None,
                       stagger=250.0)
    _, _, _, act = _build_geometry(layers, **built["kwargs"])
    first = [int(np.argmax(act[100, :, 4 * k:4 * k + 4].any(axis=1))) * dx for k in range(2)]
    assert first[1] - first[0] == pytest.approx(250.0, abs=1.5 * dx)
    assert built["meta"]["tongues"] == [] and built["meta"]["stagger"] == 250.0


def test_a_stagger_belongs_to_the_depositional_edges_with_a_sand_of_more_than_one_layer():
    args = (8000.0, 6000.0, 2000.0, [4.0, 4.0], 1)
    for kind, kw in (("truncation", dict(dip=1.0, taper_angle=0.5)), ("onlap", {}), ("lens", {}),
                     ("pinchout_nose", dict(area=None, nose=NOSE))):
        with pytest.raises(ValueError, match="stagger"):
            strat_trap(kind, *args, stagger=100.0, **kw)
    with pytest.raises(ValueError, match="stagger"):
        strat_trap("pinchout", 8000.0, 6000.0, 2000.0, [8.0], 1, stagger=100.0)


@pytest.mark.parametrize("kind", ["pinchout", "facies_change"])
def test_a_tongue_has_a_convex_up_top_on_a_flat_base_when_asked_and_hangs_from_the_bedding_plane_by_default(kind):
    """The geometry of the lens's mound, for the tongues of a pinch-out or a facies change: with ``mound=True`` the base of
    the sand is the plane of the beds 10 m down at every column, and the top sinks by T (1 - f) toward the edge (the
    thickness factor f), so it is deepest at the tip; by default, and with ``mound=False``, the top is the plane of the
    beds and the base is a bowl. With a barrier zone the interval keeps its 10 + 8 m under the bowl and the barrier is
    a flat slab of 8 m under the mound."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    thick, barrier = ([10.0, 8.0], True) if kind == "facies_change" else ([10.0], False)
    layers = _layers(x_len, y_len, dx, thick, dz=1.0)
    for mound in (None, False, True):
        built = strat_trap(kind, x_len, y_len, 2000.0, thick, seed=2, barrier=barrier, column=12.0 if barrier else None,
                           dip=1.1, taper_angle=0.4, area=3.4e6, aspect=2.2, warp=0.0, mound=mound)
        assert built["meta"]["mound"] is bool(mound)
        Xc, Yc, zc, act = _build_geometry(layers, **built["kwargs"])
        plane = 2000.0 + (Yc - 3000.0) * np.tan(np.radians(1.1))
        f = built["kwargs"]["isochore"][0](Xc, Yc)
        top, base = zc[:, :, 0], zc[:, :, 10]                              # the sand's top and base (10 layers of 1 m)
        if mound:
            assert np.allclose(top, plane + 10.0 * (1.0 - f), atol=1e-6) and np.allclose(base, plane + 10.0, atol=1e-6)
            assert (f == 0.0).any() and np.allclose((top - plane)[f == 0.0], 10.0)      # sunk by T where the sand has gone
            if barrier:
                inside = np.repeat(np.repeat(ndimage.binary_erosion(act[:, :, :10].any(axis=2), iterations=4), 2, axis=0),
                                   2, axis=1)
                assert np.allclose((zc[:, :, -1] - base)[inside], 8.0, atol=1e-6)
        else:
            assert np.allclose(top, plane, atol=1e-6) and np.allclose(base - plane, 10.0 * f, atol=1e-6)
            if barrier:
                inside = np.repeat(np.repeat(ndimage.binary_erosion(act[:, :, :10].any(axis=2), iterations=4), 2, axis=0),
                                   2, axis=1)
                assert np.allclose((zc[:, :, -1] - zc[:, :, 0])[inside], 18.0, atol=1e-6)


@pytest.mark.parametrize("kind", ["pinchout", "facies_change"])
@pytest.mark.parametrize("dip,taper,shift", [(1.1, 0.4, 10.0), (0.5, 1.2, 3.3)], ids=["no ridge", "ridge along the edge"])
def test_the_trap_measure_is_right_for_both_tops_and_the_convex_top_sinks_the_crest_and_not_the_closure(kind, dip, taper,
                                                                                                        shift):
    """Flat or convex, the cells' closure is that of the fine analytic map (1.5 cells' rise and 8 %). The convex top
    puts crest and spill down together, by up to the sand's thickness (10 m: the tip's top lies T below the plane), so
    that the closure stays that of the flat top, to 2 m; where the taper is steeper than the dip (2 tan(taper) >
    tan(dip)) the top of the edge is a ridge along it, and the spill (the ridge's low point) moves less (3 m)."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    thick, barrier = ([10.0, 8.0], True) if kind == "facies_change" else ([10.0], False)
    layers = _layers(x_len, y_len, dx, thick, dz=2.0)
    traps = {}
    for mound in (False, True):
        built = strat_trap(kind, x_len, y_len, 2000.0, thick, seed=2, barrier=barrier, column=12.0 if barrier else None,
                           dip=dip, taper_angle=taper, area=3.4e6, aspect=2.2, warp=0.2, wander=60.0, range_m=1000.0,
                           relief_sd=0.5, cell=dx, mound=mound)
        trap = trap_report(layers, built)[0]
        fine = _fine_traps(built, 2000.0, x_len, y_len)[0]
        assert trap["closure"] == pytest.approx(fine["height"], abs=1.5 * np.tan(np.radians(dip)) * dx + 0.08 * fine[
            "height"])
        traps[mound] = trap
    flat, convex = traps[False], traps[True]
    assert convex["closure"] == pytest.approx(flat["closure"], abs=2.0)
    assert 1.0 < convex["crest_depth"] - flat["crest_depth"] <= 10.0 + 1.0
    assert convex["spill_depth"] - flat["spill_depth"] == pytest.approx(shift, abs=1.0)
    if barrier:                       # the barrier logic keeps both dry; the mound's flat slab never reaches above its crest
        assert convex["limited_by"] == flat["limited_by"] == "barrier"
        assert flat["height"] - 1.5 <= convex["height"] <= 12.0 + 1e-9 and 0.5 * 12.0 < flat["height"] <= 12.0


@pytest.mark.parametrize("kind,extra", [("truncation", dict(taper_angle=0.3)), ("onlap", {}),
                                        ("pinchout_nose", dict(nose=NOSE, area=None)),
                                        ("truncation_nose", dict(nose=NOSE, area=None))])
def test_a_convex_up_top_is_not_for_an_erosion_surface_or_a_nose(kind, extra):
    with pytest.raises(ValueError, match="convex"):
        strat_trap(kind, 8000.0, 6000.0, 2000.0, [10.0], seed=1, dip=1.0, mound=True, **extra)
    strat_trap(kind, 8000.0, 6000.0, 2000.0, [10.0], seed=1, dip=1.0, mound=False, **extra)


def test_meta_gives_the_nominal_closure_and_the_report_the_realized_one_a_mound_holding_half_as_much_again():
    """``meta`` has the tangent of the dip times the length of the drawn tongue under the name it deserves
    (``closure_nominal``, ``crest_nominal``, ``spill_nominal``), and the report's ``closure`` is what the cells hold:
    for a mound of 10 m on a gentle dip (0.3 degrees, 3.4 km2) the nominal 10.5 m, the realized 16 m (the mound's own
    relief, T (1 + x)^2 with x = 10.5 / 40), as the fine analytic map says; with a rough edge and a top with relief
    it is another number again. Nothing in ``meta`` is called ``expected``."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    layers = _layers(x_len, y_len, dx, [10.0], dz=1.0)
    mound = strat_trap("lens", x_len, y_len, 2000.0, [10.0], seed=2, dip=0.3, taper_angle=0.3, area=3.4e6, aspect=2.2,
                       warp=0.0)
    assert not [key for key in mound["meta"] if "expected" in key]
    nominal = mound["meta"]["closure_nominal"]
    assert nominal == pytest.approx(np.tan(np.radians(0.3)) * _length(3.4e6, 2.2))
    (trap,) = trap_report(layers, mound)
    assert trap["closure"] == pytest.approx(10.0 * (1.0 + nominal / 40.0) ** 2, rel=0.06) and trap["closure"] > 1.4 * nominal
    assert trap["closure"] == pytest.approx(_fine_traps(mound, 2000.0, x_len, y_len)[0]["height"], abs=1.5 * 0.0052 * dx
                                            + 0.05 * trap["closure"])
    rough = strat_trap("pinchout", x_len, y_len, 2000.0, [10.0], seed=4, dip=1.2, taper_angle=0.4, area=3.4e6, aspect=2.2,
                       warp=0.2, wander=120.0, relief_sd=1.5, cell=dx)
    main = trap_report(layers, rough)[0]
    assert abs(main["closure"] - rough["meta"]["closure_nominal"]) > 1.0
    assert main["closure"] == pytest.approx(_fine_traps(rough, 2000.0, x_len, y_len)[0]["height"], abs=1.5 * 0.021 * dx +
                                            0.08 * main["closure"])


@pytest.mark.parametrize("aspect", [0.5, 2.2])
def test_a_mound_is_whole_in_the_middle_whatever_its_aspect_so_the_taper_is_limited_by_its_half_width(aspect):
    """A lens of 3.4 km2 as a mound tapers over the least of the taper that the angle draws (1.9 km for 10 m at 0.3
    degrees) and its half-length along dip or across it, so that it is as thick as the sand in its middle (10 m, to 2 %)
    whether it is long and narrow (aspect 0.5: half-width 0.69 km, the limit) or short and wide (2.2: half-length
    0.7 km)."""
    x_len, y_len, dx = 8000.0, 6000.0, 50.0
    built = strat_trap("lens", x_len, y_len, 2000.0, [10.0], seed=2, dip=1.1, taper_angle=0.3, area=3.4e6, aspect=aspect,
                       warp=0.0)
    length = _length(3.4e6, aspect)
    assert built["meta"]["taper_m"] == pytest.approx(0.5 * length * min(1.0, aspect))
    assert built["meta"]["taper_m"] < 10.0 / np.tan(np.radians(0.3))
    layers = _layers(x_len, y_len, dx, [10.0], dz=1.0)
    _, _, zc, _ = _build_geometry(layers, **built["kwargs"])
    assert (zc[:, :, -1] - zc[:, :, 0]).max() == pytest.approx(10.0, rel=0.02)
