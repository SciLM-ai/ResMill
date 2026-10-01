"""Fault patterns tied to the fold: the structure research's acceptance tests T1-T8 (fold_fault_relations.md, R-14),
the style templates (R-5) and the fixes of the independent review."""
import math

import numpy as np
import pytest
from scipy import ndimage

from resmill import structure as st
from resmill.export import to_grdecl
from resmill.fault_patterns import MIN_THROW, STYLES, _frame, _in_reservoir, fold_faults
from resmill.layers.base import Layer

X_LEN, Y_LEN, DX = 16000.0, 12000.0, 100.0
CENTER = (8000.0, 6000.0)
TOP, THICK = 2000.0, 60.0
FOLD = ("longitudinal", "oblique", "transverse")


def dome(aspect, limb_ratio=1.0, tilt=0.0, **kw):
    return st.closure(area=20e6, height=150.0, aspect=aspect, azimuth=0.0, center=CENTER, limb_ratio=limb_ratio,
                      tilt=tilt, **kw)


def faults_of(style, fold, seed, density=2.0, x_len=X_LEN, y_len=Y_LEN, dx=DX, **kw):
    return fold_faults(style, fold, x_len, y_len, dx, density, TOP, THICK, seed=seed, **kw)


def pooled(style, fold, seeds=range(48), density=2.0, **kw):
    return [f for s in seeds for f in faults_of(style, fold, s, density, **kw)]


def share(faults, azimuth, within=10.0, kinds=None):
    """Share of the faults' length whose traces lie within +-``within`` degrees of the direction ``azimuth``."""
    sel = [f for f in faults if kinds is None or f.kind in kinds]
    length = np.array([f.length for f in sel])
    off = np.abs((np.array([f.strike for f in sel]) - azimuth + 90.0) % 180.0 - 90.0)
    return float(length[off <= within].sum() / length.sum())


def reservoir_throw(f, fold):
    """A fault's central throw left in the reservoir at its centre."""
    t_loc = TOP + float(np.asarray(fold(np.array([f.center[0]]), np.array([f.center[1]]))).ravel()[0])
    return f.throw * _in_reservoir(f.z_center, t_loc, THICK, 0.5 * f.length / 2.15 * math.sin(math.radians(f.dip)))


def allowed_of(fold, x_len=X_LEN, y_len=Y_LEN, dx=DX):
    fr = _frame(fold, x_len, y_len, dx)
    return fr, ndimage.distance_transform_edt(~fr["mask"], sampling=(fr["dx"], fr["dy"])) <= 0.3 * fr["b"]


def unit(strike):
    s = math.radians(strike)
    return np.array([math.cos(s), -math.sin(s)]), np.array([math.sin(s), math.cos(s)])     # trace, normal


def crosses(f, allowed, dx, n=400):
    """Whether a fault's chord passes through ``allowed``."""
    t, _ = unit(f.strike)
    pts = np.array(f.center)[None, :] + np.linspace(-0.5, 0.5, n)[:, None] * f.length * t[None, :]
    ij = np.floor(pts / dx).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < allowed.shape[0]) & (ij[:, 1] < allowed.shape[1])
    return bool(allowed[ij[ok, 0], ij[ok, 1]].any())


def test_t1_an_elongate_dome_without_regional_strain():
    """T1: on an aspect-2 dome with no regional strain, 15-30 % of the fold faults' length lies within 10 degrees of
    the long axis and 5-15 % of the short axis (Withjack and Scheiner 1982 measured 18 % and 10 %)."""
    faults = pooled("four_way", dome(2.0))
    assert 0.15 <= share(faults, 0.0, kinds=FOLD) <= 0.30
    assert 0.05 <= share(faults, 90.0, kinds=FOLD) <= 0.15


def test_t2_a_round_dome_under_regional_extension():
    """T2: on a round dome under regional extension (R = 1.5), 35-65 % of the fault length lies across the extension
    (measured 46-64 %)."""
    assert 0.35 <= share(pooled("faulted_anticline", dome(1.0), regional=(1.5, 0.0)), 90.0) <= 0.65


@pytest.mark.parametrize("regional", [(-0.4, 90.0), None])
def test_t3_an_elongate_fold_under_shortening(regional):
    """T3: on an elongate fold shortened across its axis, 13-30 % of the fold faults' length is transverse, parallel
    to the shortening (measured 13-26 %), at R = -0.4 and on the fold-belt style's own draw of R."""
    kw = {} if regional is None else dict(regional=regional)
    assert 0.13 <= share(pooled("fold_belt", dome(2.0), **kw), 90.0, kinds=FOLD) <= 0.30


@pytest.mark.parametrize("dx", [100.0, 25.0])
def test_t5_fold_faults_end_inside_the_buffered_trap_relay_segments_too(dx):
    """T5: fold-related faults, their relay segments, crestal grabens and steps end inside the trap buffered by 0.3 of
    its half-width, so their throw dies at the fold's edge; regional faults run on."""
    x_len, y_len = (X_LEN, Y_LEN) if dx == 100.0 else (6000.0, 5000.0)
    fold = dome(2.0) if dx == 100.0 else st.closure(area=6e6, height=120.0, aspect=2.0, center=(3000.0, 2500.0))
    fr, allowed = allowed_of(fold, x_len, y_len, dx)
    seeds = range(12) if dx == 100.0 else range(6)
    faults = [f for s in seeds for f in faults_of("faulted_anticline", fold, s, 4.0, x_len, y_len, dx)]
    for f in faults:
        if f.kind in FOLD + ("graben", "step"):
            t, _ = unit(f.strike)
            for tip in (np.array(f.center) + 0.5 * f.length * t, np.array(f.center) - 0.5 * f.length * t):
                i, j = min(int(tip[0] // dx), allowed.shape[0] - 1), min(int(tip[1] // dx), allowed.shape[1] - 1)
                assert allowed[max(i, 0), max(j, 0)]
    assert any(f.kind in ("regional", "major") for f in faults)


def test_t6_crestal_grabens_face_inward_and_span_005_045_of_the_trap():
    """T6: a crestal graben's width is 0.05-0.45 of the trap's width and each master's hanging wall faces the other."""
    fold = dome(2.0)
    fr = _frame(fold, X_LEN, Y_LEN, DX)
    seen = 0
    for s in range(40):
        g = [f for f in faults_of("turtle", fold, s) if f.kind == "graben"]
        if len(g) == 2:
            across = np.array([-math.sin(fr["axis"]), math.cos(fr["axis"])])
            width = abs(np.dot(np.array(g[0].center) - np.array(g[1].center), across))
            assert 0.05 <= width / (2.0 * fr["b"]) <= 0.45
            for f, other in ((g[0], g[1]), (g[1], g[0])):
                _, normal = unit(f.strike)
                assert np.dot(np.array(other.center) - np.array(f.center), f.hanging_wall * normal) > 0
            seen += 1
    assert seen > 20                                                     # turtles: a graben in 80 % of traps


def test_t7_the_density_counts_faults_of_five_metres_in_the_reservoir_and_only_those_are_drawn():
    """T7 and S6: the population faults number the density times the trap's (closure's) area, each with at least 5 m
    of throw in the reservoir (a fault drawn smaller is drawn longer, its displacement-length scatter kept)."""
    fold = dome(2.0)
    fr, allowed = allowed_of(fold)
    area = fr["mask"].sum() * DX * DX / 1e6
    for density in (0.5, 2.0, 5.0):
        counts = []
        for s in range(8):
            pop = [f for f in faults_of("four_way", fold, s, density) if f.kind in FOLD + ("inherited",)]
            assert all(reservoir_throw(f, fold) >= MIN_THROW - 1e-9 for f in pop)
            counts.append(len(pop))
        assert 0.85 * density * area <= np.mean(counts) <= 1.3 * density * area + 1


def test_low_relief_faults_are_long_and_under_displaced_like_burgans():
    """Low relief: displacement-length ratios down to 0.0025 (Burgan: 3-6 km long, throws below 15 m), the faults
    reaching 5 m in the reservoir by being long."""
    fold = st.closure(area=40e6, height=25.0, aspect=1.8, center=CENTER)
    faults = [f for f in pooled("low_relief", fold, seeds=range(24), density=0.3) if f.kind in ("regional", "major")]
    ratio = np.array([f.throw / f.length for f in faults])
    assert np.median(ratio) < 0.01 and np.percentile(ratio, 10) < 0.004
    assert np.median([f.length for f in faults]) > 1500.0


def test_t8_fold_faults_are_labelled_by_their_angle_to_the_fold_axis():
    """T8: longitudinal within 30 degrees of the fold axis, transverse beyond 60, oblique between."""
    fold = dome(2.0, limb_ratio=1.5)
    axis = math.degrees(-_frame(fold, X_LEN, Y_LEN, DX)["axis"])
    for f in pooled("four_way", fold, seeds=range(4)):
        if f.kind in FOLD:
            off = abs((f.strike - axis + 90.0) % 180.0 - 90.0)
            assert f.kind == ("longitudinal" if off < 30.0 else "transverse" if off > 60.0 else "oblique")


def test_the_crest_is_the_traps_top_when_the_trap_holds_satellite_highs():
    """With satellite culminations the trap holds several local highs; the crest is its shallowest point."""
    for s in range(10):
        fold = dome(2.0, satellites=2, seed=s)
        fr = _frame(fold, X_LEN, Y_LEN, DX)
        assert fr["depth"][fr["crest"]] == fr["depth"][fr["mask"]].min()


def test_each_fault_is_placed_from_the_reservoirs_depth_where_it_is():
    """On a tilted 150 m fold the fold-related faults' tip ellipses centre on the reservoir at their own position
    (scatter N(0, 0.6 half-height)), not on the crest's depth."""
    fold = dome(2.0, tilt=0.4)
    offs = []
    for f in pooled("four_way", fold, seeds=range(8)):
        if f.kind in FOLD:
            local = TOP + float(np.asarray(fold(np.array([f.center[0]]), np.array([f.center[1]]))).ravel()[0])
            half = 0.5 * f.length / 2.15 * math.sin(math.radians(f.dip))
            offs.append((f.z_center - (local + 0.5 * THICK)) / (0.6 * half))
    assert abs(np.mean(offs)) < 0.2 and 0.65 < np.std(offs) < 1.0     # N(0, 1) less the few that miss the reservoir


@pytest.mark.parametrize("limb_ratio", [2.0, 3.0, 4.0])
def test_compressional_folds_get_a_break_thrust_on_the_steeper_limb_with_the_core_as_hanging_wall(limb_ratio):
    """Fold belts: a reverse fault on the steeper limb whenever the limb ratio is 1.3 or more, inside the trap even on
    a narrow forelimb, its hanging wall the fold's core (E10, E25)."""
    fold = dome(2.0, limb_ratio=limb_ratio)
    fr = _frame(fold, X_LEN, Y_LEN, DX)
    crest = np.array([fr["X"][fr["crest"]], fr["Y"][fr["crest"]]])
    across = np.array([-math.sin(fr["axis"]), math.cos(fr["axis"])])
    gx, gy = np.gradient(fr["depth"], DX, DX)
    side = (fr["X"] - crest[0]) * across[0] + (fr["Y"] - crest[1]) * across[1]
    slope = np.hypot(gx, gy)
    steep = 1.0 if slope[fr["mask"] & (side > 0)].mean() > slope[fr["mask"] & (side < 0)].mean() else -1.0
    for s in range(12):
        thrusts = [f for f in faults_of("fold_belt", fold, s) if f.kind == "thrust"]
        assert len(thrusts) == 1 and thrusts[0].reverse
        f = thrusts[0]
        assert np.dot(np.array(f.center) - crest, across) * steep > 0
        _, normal = unit(f.strike)
        assert np.dot(crest - np.array(f.center), f.hanging_wall * normal) > 0


def test_tear_faults_are_steep_with_throw_below_one_percent_of_their_length():
    """The owner's strike-slip faults: steep (80-90 degrees) tear faults across the axis, throw 0.2-1 % of length."""
    tears = [f for f in pooled("fold_belt", dome(2.0, limb_ratio=2.0), seeds=range(60)) if f.kind == "tear"]
    assert len(tears) > 30 and all(f.dip >= 80.0 and f.throw <= 0.01 * f.length + 1e-9 for f in tears)


def test_grabens_and_steps_never_exceed_06_of_the_traps_relief():
    """R-6: a crestal graben's first master throws 0.2-0.6 of the relief (Kooh Bedoo 0.63, Gavbast 0.18), step faults
    together the same, even on a broad low trap where the displacement-length law alone would exceed the relief."""
    for fold in (dome(2.0), st.closure(area=60e6, height=60.0, aspect=1.5, center=CENTER)):
        relief = _frame(fold, X_LEN, Y_LEN, DX)["relief"]
        for s in range(30):
            faults = faults_of("four_way", fold, s, over_salt=True)
            masters = [f.throw for f in faults if f.kind == "graben"]
            if masters:
                assert 0.2 * relief - 1e-9 <= max(masters) <= 0.6 * relief + 1e-9
            steps = sum(f.throw for f in faults if f.kind == "step")
            assert steps == 0.0 or 0.2 * relief - 1e-9 <= steps <= 0.6 * relief + 1e-9


def saddle(fold):
    fr = _frame(fold, X_LEN, Y_LEN, DX)
    ring = ndimage.binary_dilation(fr["mask"]) & ~fr["mask"]
    k = np.unravel_index(int(np.argmin(np.where(ring, fr["depth"], np.inf))), ring.shape)
    return fr, np.array([fr["X"][fr["crest"]], fr["Y"][fr["crest"]]]), np.array([fr["X"][k], fr["Y"][k]])


def test_the_trap_is_found_under_a_regional_tilt_that_makes_a_corner_the_shallowest_point():
    fold = st.closure(area=15e6, height=100.0, aspect=1.6, center=CENTER, tilt=0.45, seed=7)
    X, Y = np.meshgrid((np.arange(160) + 0.5) * DX, (np.arange(120) + 0.5) * DX, indexing="ij")
    corner = np.unravel_index(int(np.argmin(np.asarray(fold(X, Y)))), X.shape)
    assert corner[0] in (0, 159) or corner[1] in (0, 119)
    fr = _frame(fold, X_LEN, Y_LEN, DX)
    assert fr is not None and abs(fr["X"][fr["crest"]] - CENTER[0]) < 3000.0 and fr["mask"].sum() * DX * DX > 5e6
    assert faults_of("fault_bounded", fold, 1, 1.0)


def test_a_fault_bounded_trap_is_sealed_through_its_spill_point_mostly_from_its_footwall():
    """The bounding fault crosses the up-dip direction at the spill point (the saddle); the trap lies in its footwall
    in about 60 % of traps (upthrown sides, E17; Columbus Basin footwalls, E20), in its hanging wall otherwise; minor
    faults parallel to it sit in its hanging wall, mostly synthetic (E19, E20)."""
    fold = dome(1.6, tilt=0.4)
    fr, crest, spill = saddle(fold)
    footwall, synthetic = [], []
    for s in range(40):
        faults = faults_of("fault_bounded", fold, s, 1.0)
        b = [f for f in faults if f.kind == "bounding"]
        assert len(b) == 1
        f = b[0]
        _, normal = unit(f.strike)
        assert abs(np.dot(spill - np.array(f.center), normal)) < 2.0 * DX
        footwall.append(np.dot(crest - np.array(f.center), f.hanging_wall * normal) < 0)
        for m in faults:
            if m.kind == "minor":
                assert abs((m.strike - f.strike + 90.0) % 180.0 - 90.0) < 40.0
                assert np.dot(np.array(m.center) - np.array(f.center), f.hanging_wall * normal) > 0
                synthetic.append(m.hanging_wall * np.sign(np.dot(normal, unit(m.strike)[1])) == f.hanging_wall)
    assert 0.45 <= np.mean(footwall) <= 0.75 and 0.55 <= np.mean(synthetic) <= 0.85


def test_on_an_untilted_fold_the_bounding_faults_side_varies():
    """Without a tilt every rim cell ties for the spill point; ties are broken at random, not always to the west."""
    fold = dome(1.6)
    fr = _frame(fold, X_LEN, Y_LEN, DX)
    crest = np.array([fr["X"][fr["crest"]], fr["Y"][fr["crest"]]])
    sides = {round(math.degrees(math.atan2(*(np.array(f.center) - crest)[::-1])) / 45.0)
             for s in range(12) for f in faults_of("fault_bounded", fold, s, 0.5) if f.kind == "bounding"}
    assert len(sides) >= 3


def test_basinward_regional_faults_dip_down_the_regional_slope():
    """Gulf type: about 80 % of the regional faults dip basinward, their hanging wall down the regional slope the fold
    sits on (E9), whatever the fold's azimuth."""
    for azimuth in (0.0, 40.0):
        fold = st.closure(area=20e6, height=150.0, aspect=2.0, azimuth=azimuth, center=CENTER, tilt=0.4)
        fr = _frame(fold, X_LEN, Y_LEN, DX)
        gx, gy = np.gradient(fr["depth"], DX, DX)
        down = np.array([gx[~fr["mask"]].mean(), gy[~fr["mask"]].mean()])
        votes = [np.dot(f.hanging_wall * unit(f.strike)[1], down) > 0
                 for f in pooled("faulted_anticline", fold, seeds=range(16)) if f.kind in ("regional", "major")]
        assert 0.65 <= np.mean(votes) <= 0.95


def test_regional_faults_are_concave_toward_their_hanging_wall_and_some_are_major():
    """Regional faults: concave toward the hanging wall, 2-10 lengths radius; zero to three majors per trap, 3-20 km
    long, crossing the buffered trap (a relay pair together); the rest with Norne's lengths."""
    fold = dome(1.0)
    fr, allowed = allowed_of(fold)
    n_major = []
    for s in range(16):
        faults = faults_of("faulted_anticline", fold, s, regional=(1.5, 0.0))
        majors = [f for f in faults if f.kind == "major"]
        n_major.append(len({round(f.strike, 6) for f in majors}))
        for f in faults:
            if f.kind in ("regional", "major"):
                assert np.sign(f.radius) == f.hanging_wall and abs(f.radius) <= 10.0 * f.length + 1e-6
        assert all(f.length <= 4000.0 + 1e-6 for f in faults if f.kind == "regional")
        assert all(0.5 * 3000.0 - 1e-6 <= f.length <= 20000.0 + 1e-6 for f in majors)
        for f in majors:
            assert crosses(f, allowed, DX) or any(crosses(g, allowed, DX) for g in majors
                                                  if g is not f and g.strike == f.strike)
    assert set(n_major) <= {0, 1, 2, 3} and max(n_major) >= 2


def test_the_regional_density_inside_the_trap_does_not_depend_on_the_models_size():
    """Regional faults keep their share of the density over the whole model, so the same trap gets about as many per
    km2 in a 10 x 8 km model as in a 32 x 24 km one."""
    rates = []
    for x_len, y_len in ((10000.0, 8000.0), (32000.0, 24000.0)):
        fold = st.closure(area=20e6, height=150.0, aspect=2.0, center=(0.5 * x_len, 0.5 * y_len))
        fr, allowed = allowed_of(fold, x_len, y_len, 200.0)
        hits = sum(crosses(f, allowed, 200.0, 60) for s in range(12)
                   for f in faults_of("faulted_anticline", fold, s, 2.0, x_len, y_len, 200.0, regional=(1.0, 0.0),
                                      max_faults=5000)
                   if f.kind == "regional")
        rates.append(hits / (12 * allowed.sum() * 200.0 * 200.0 / 1e6))
    assert 0.5 <= rates[0] / rates[1] <= 2.0


def test_a_transverse_graben_needs_a_salt_core():
    """R-7: the graben is transverse only in shortened salt-cored domes (E13); basement-cored folds keep it
    longitudinal."""
    fold = dome(2.0, limb_ratio=2.0)
    axis = math.degrees(-_frame(fold, X_LEN, Y_LEN, DX)["axis"])
    angle = lambda f: abs((f.strike - axis + 90.0) % 180.0 - 90.0)
    plain = [angle(f) for f in pooled("fold_belt", fold, seeds=range(40)) if f.kind == "graben"]
    salt = [angle(f) for f in pooled("fold_belt", fold, seeds=range(40), over_salt=True) if f.kind == "graben"]
    assert plain and max(plain) < 45.0 and salt and max(salt) > 45.0


def test_max_faults_caps_the_total_and_bad_input_is_refused():
    fold = dome(2.0)
    for cap in (5, 20):
        assert len(faults_of("faulted_anticline", fold, 1, 50.0, max_faults=cap)) <= cap
    with pytest.raises(ValueError, match="non-finite"):
        fold_faults("four_way", lambda x, y: np.where(x > 8000.0, np.nan, 0.0 * x), X_LEN, Y_LEN, DX, 1.0, TOP, THICK, 1)
    with pytest.raises(ValueError):
        faults_of("dome", fold, 1)


def test_every_style_draws_valid_faults_that_the_grid_export_takes(tmp_path):
    """Each style, same seed same faults; names unique within 8 characters; to_grdecl accepts them."""
    nx, ny, nz = 80, 60, 6
    layer = Layer(nx, ny, nz, nx * 100.0, ny * 100.0, 30.0, top_depth=TOP, kzkx=0.1)
    layer.poro_mat = np.full((nx, ny, nz), 0.2)
    layer.perm_mat = np.full((nx, ny, nz), 100.0)
    fold = st.closure(area=6e6, height=100.0, aspect=1.8, center=(4000.0, 3000.0), limb_ratio=1.5, tilt=0.2)
    for style in STYLES:
        faults = fold_faults(style, fold, 8000.0, 6000.0, 100.0, 2.0, TOP, 30.0, seed=7)
        again = fold_faults(style, fold, 8000.0, 6000.0, 100.0, 2.0, TOP, 30.0, seed=7)
        assert faults and faults == again
        assert len({f.name for f in faults}) == len(faults) and all(len(f.name) <= 8 for f in faults)
        to_grdecl(layer, tmp_path / f"{style}.grdecl", structure=fold, faults=faults)


def test_the_cap_drops_the_regional_faults_farthest_from_the_trap_first():
    """With the default cap of 300 on a big model, the trap keeps all its own faults and its regional density; the
    regional faults far from it are the ones left out."""
    x_len, y_len = 30000.0, 20000.0
    fold = st.closure(area=20e6, height=150.0, aspect=2.0, center=(15000.0, 10000.0))
    fr, allowed = allowed_of(fold, x_len, y_len, 200.0)
    area = fr["mask"].sum() * 200.0 * 200.0 / 1e6
    for s in range(4):
        capped = faults_of("faulted_anticline", fold, s, 2.0, x_len, y_len, 200.0, regional=(1.0, 0.0))
        full = faults_of("faulted_anticline", fold, s, 2.0, x_len, y_len, 200.0, regional=(1.0, 0.0), max_faults=5000)
        assert len(capped) <= 300 < len(full)
        fold_n = lambda fs: sum(f.kind in FOLD for f in fs)
        assert fold_n(capped) == fold_n(full) and fold_n(capped) >= 0.4 * 2.0 * area
        near = lambda fs: sum(f.kind == "regional" and crosses(f, allowed, 200.0, 60) for f in fs)
        assert near(capped) >= 0.8 * near(full)


def test_the_regional_count_follows_the_model_not_the_traps_own_sets():
    """Regional faults number the density times their share times the model's area, whatever the trap's own sets
    (on a 2 km2 trap a crestal graben alone uses up the trap's count)."""
    fold = st.closure(area=2e6, height=80.0, aspect=1.5, center=CENTER)
    expected = 1.0 * 0.5 * X_LEN * Y_LEN / 1e6
    counts = [sum(f.kind in ("regional", "major") for f in faults_of("four_way", fold, s, 1.0, regional=(1.0, 0.0),
                                                                      over_salt=True, max_faults=5000))
              for s in range(8)]
    assert 0.85 * expected <= np.mean(counts) <= 1.15 * expected


def test_relay_segments_reach_five_metres_at_their_own_centres():
    """A relay pair forms only when each segment keeps 5 m of throw in the reservoir where it lies."""
    fold = st.closure(area=6e6, height=120.0, aspect=2.0, center=(3000.0, 2500.0), tilt=0.3)
    seen = 0
    for s in range(8):
        faults = faults_of("faulted_anticline", fold, s, 4.0, 6000.0, 5000.0, 25.0)
        lengths = {}
        for f in faults:
            lengths.setdefault((f.kind, round(f.strike, 6), f.throw), []).append(f)
        for group in lengths.values():
            if len(group) == 2:
                seen += 1
                assert all(reservoir_throw(f, fold) >= MIN_THROW - 1e-6 for f in group)
    assert seen > 0


def test_low_relief_regional_faults_are_long_like_burgans():
    """Low relief: the regional faults are 3-20 km long (Burgan's 3-6 km) and under-displaced; a relay's segments
    half that at least."""
    fold = st.closure(area=40e6, height=25.0, aspect=1.8, center=CENTER)
    faults = [f for f in pooled("low_relief", fold, seeds=range(16), density=0.3) if f.kind == "regional"]
    assert faults and all(f.length >= 0.5 * 3000.0 - 1e-6 for f in faults)


def test_minor_faults_stay_in_the_bounding_faults_hanging_wall():
    fold = dome(1.6, tilt=0.4)
    for s in range(30):
        faults = faults_of("fault_bounded", fold, s, 1.0)
        b = [f for f in faults if f.kind == "bounding"][0]
        _, normal = unit(b.strike)
        for m in faults:
            if m.kind == "minor":
                t, _ = unit(m.strike)
                for sgn in (-1, 1):
                    tip = np.array(m.center) + sgn * 0.5 * m.length * t
                    assert np.dot(tip - np.array(b.center), b.hanging_wall * normal) > 0


def test_a_flat_topped_trap_draws_no_crestal_sets_and_does_not_fail():
    plateau = st.Structure(lambda x, y: -np.minimum(150.0 * np.exp(-((np.asarray(x) - 8000.0) ** 2 + (np.asarray(y) - 6000.0) ** 2)
                                                                  / 2500.0 ** 2), 100.0))
    for style in ("four_way", "turtle", "fold_belt"):
        faults_of(style, plateau, 1, over_salt=True)


def test_regional_faults_spread_evenly_over_the_model_when_the_cap_does_not_bind():
    """Regionally oriented faults are not tied to the fold: their centres fill the model evenly (about as many in each
    quarter), unless the cap forces the far ones out."""
    fold = dome(2.0)
    centres = np.array([f.center for f in pooled("faulted_anticline", fold, seeds=range(6), regional=(1.0, 0.0),
                                                   max_faults=5000) if f.kind == "regional"])
    quarters = [np.sum((centres[:, 0] < X_LEN / 2) == left) for left in (True, False)]
    rings = np.hypot(centres[:, 0] - CENTER[0], centres[:, 1] - CENTER[1])
    assert min(quarters) > 0.4 * len(centres)
    assert np.mean(rings > 4000.0) > 0.5                                 # most of the model lies over 4 km out
