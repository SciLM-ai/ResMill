"""Salt traps: salt_flank_trap and subsalt_trap, composed from ResMill's own pieces (ResSimMill step 7, task 2).

Every expected number is worked out here from the formulas of the pieces (a circular closure's radius, the contact's
distance from the crest, a hook's relief and its truncation angle, the overhang's width above the neck), not read back
from the composers."""
import math

import numpy as np
import pytest

from resmill import structure as st
from resmill.export import _build_geometry, to_grdecl
from resmill.fault_patterns import fold_faults
from resmill.fault_seal import Capillary, blocks_at
from resmill.fold_traps import fold_trap
from resmill.layers.base import Layer
from resmill.salt import salt_upturn, sequence_dip, truncation_cut
from resmill.salt_traps import salt_flank_trap, subsalt_trap

NX, NY, NZ, DX, TOP = 80, 60, 12, 50.0, 2000.0
X_LEN, Y_LEN, THICK = NX * DX, NY * DX, NZ * 5.0
MIDDLE = (0.5 * X_LEN, 0.5 * Y_LEN)
RADIUS = 800.0
# a circular closure of 800 m radius, no warp, tilt or satellites: its trap is a disc about the model's middle
CLOSURE = dict(area=math.pi * RADIUS ** 2, height=80.0, seed=1)
BODY = dict(axes=(1000.0, 1000.0))
FLANK = dict(normal=0.0, wall_at=0.4, width=300.0, dip=40.0)
BASE = dict(kind="truncated", cut=0.5, base_dip=0.0, base_azimuth=0.0, rugosity_sd=0.0, rugosity_range=1500.0, sheet=900.0,
            seed=3)


def flank(**override):
    args = dict(closure=CLOSURE, body=BODY, **FLANK)
    args.update(override)
    return salt_flank_trap(X_LEN, Y_LEN, DX, TOP, THICK, **args)


def subsalt(**override):
    args = dict(closure=CLOSURE, **BASE)
    args.update(override)
    return subsalt_trap(X_LEN, Y_LEN, DX, TOP, THICK, **args)


def layer(nz=NZ):
    model = Layer(NX, NY, nz, X_LEN, Y_LEN, THICK, top_depth=TOP, kzkx=0.1)
    model.poro_mat = np.full((NX, NY, nz), 0.2)
    model.perm_mat = np.full((NX, NY, nz), 100.0)
    return model


def fold_of(closure=CLOSURE):
    return st.closure(**{**closure, "center": MIDDLE})


def inside(salt, x, y, z):
    return bool(salt.inside(np.array([x]), np.array([y]), np.array([z]))[0])


# ---- the flank: where the contact lies ----

def test_the_contact_passes_wall_at_of_the_closures_half_width_beyond_the_crest_and_cuts_it_off():
    """A disc of 800 m radius about (2000, 1500); salt to +x with the contact at 0.4 x 800 = 320 m from the crest on
    the far side, so the crest lies 320 m inside the salt: the contact crosses y = 1500 at x = 2000 - 320 = 1680, and a
    stock of 1,000 m radius reaches 1680 + 2000 = 3680 on the other side. The reach is measured in steps of half a
    cell (25 m), so the contact may be 0.4 x 25 = 10 m off."""
    built = flank()
    salt, z = built["kwargs"]["salt"], built["meta"]["z_ref"]
    for x, there in ((1680.0 - 30.0, False), (1680.0 + 30.0, True), (3680.0 - 30.0, True), (3680.0 + 30.0, False)):
        assert inside(salt, x, MIDDLE[1], z) == there
    assert inside(salt, MIDDLE[0], MIDDLE[1], z)                          # the crest itself is in the salt


def test_the_normal_turns_the_wall_around_the_crest():
    """Normal 90 degrees puts the salt toward +y: the contact crosses x = 2000 at y = 1500 - 320 = 1180."""
    salt, z = (lambda b: (b["kwargs"]["salt"], b["meta"]["z_ref"]))(flank(normal=90.0))
    assert not inside(salt, MIDDLE[0], 1180.0 - 30.0, z) and inside(salt, MIDDLE[0], 1180.0 + 30.0, z)
    assert not inside(salt, 1000.0, MIDDLE[1], z)                         # nothing toward -x


def test_the_half_width_is_measured_away_from_the_salt():
    """A closure whose forelimb (the +y side, azimuth 0) is steeper than its backlimb, limb ratio 2, is half as wide on the
    fore side (widths 2 / (1 + 2) and 4 / (1 + 2) of the same unit): salt toward +y puts the contact on the wide back side, and
    salt toward -y on the narrow fore side, so the contact lies about twice as far from the crest in the first case."""
    asymmetric = dict(CLOSURE, limb_ratio=2.0)
    toward_plus = flank(closure=asymmetric, normal=90.0)["meta"]
    toward_minus = flank(closure=asymmetric, normal=270.0)["meta"]
    far = abs(toward_plus["contact"][1] - MIDDLE[1])
    near = abs(toward_minus["contact"][1] - MIDDLE[1])
    assert toward_plus["contact"][1] < MIDDLE[1] < toward_minus["contact"][1]            # each on the side away from its salt
    assert far / near == pytest.approx(2.0, rel=0.15)


def test_a_wall_lies_across_the_normal_with_its_short_axis_toward_the_closure():
    """axes (4000, 600), no azimuth given: the long axis runs across the normal (along y for normal 0), so the contact
    is a straight-ish wall 600 m from its middle line: the middle is at 1680 + 600, and at y = 1500 +- 1500 it still
    stands at x = 1680 within a superellipse's curl (ellipse: x = 2280 - 600 sqrt(1 - (dy/4000)^2))."""
    built = flank(body=dict(axes=(4000.0, 600.0)))
    salt, z = built["kwargs"]["salt"], built["meta"]["z_ref"]
    edge_at = lambda dy: 2280.0 - 600.0 * math.sqrt(1.0 - (dy / 4000.0) ** 2)
    for dy in (0.0, 1000.0, -1200.0):
        x = edge_at(dy)
        assert not inside(salt, x - 25.0, MIDDLE[1] + dy, z) and inside(salt, x + 25.0, MIDDLE[1] + dy, z)


def test_an_elliptical_stock_is_centred_so_its_ellipse_passes_through_the_contact_point():
    """Axes (1500, 700) at azimuth 30 degrees (long axis along (cos 30, -sin 30)): its polar radius toward the crest from
    the middle, hand-computed, puts the contact point q = (1680, 1500) on the outline."""
    built = flank(body=dict(axes=(1500.0, 700.0), azimuth=30.0))
    salt, z = built["kwargs"]["salt"], built["meta"]["z_ref"]
    t = np.array([math.cos(math.radians(30.0)), -math.sin(math.radians(30.0))])
    n = np.array([math.sin(math.radians(30.0)), math.cos(math.radians(30.0))])
    r = 1.0 / math.sqrt((t[0] / 1500.0) ** 2 + (n[0] / 700.0) ** 2)       # the polar radius along +x
    assert not inside(salt, 1680.0 - 30.0, MIDDLE[1], z) and inside(salt, 1680.0 + 30.0, MIDDLE[1], z)
    assert salt.center == pytest.approx((1680.0 + r, MIDDLE[1]), abs=15.0)


# ---- the overhang and the neck ----

def test_an_overhang_widens_the_salt_by_its_lateral_extent_over_its_height_above_the_neck():
    """overhang (400, 300) and neck 0.5, a plain upturn of 300 m, 40 degrees (exponent 2, since 300 tan 40 / 350 = 0.72
    is under 2): relief 300 tan(40) / 2 = 125.9 m. The beds' top at the contact is the fold at q less that, and the neck lies
    0.5 x 300 m below it. At the neck the contact is at q; at 300 m above it, 400 m beyond (toward -x)."""
    built = flank(body=dict(axes=(1000.0, 1000.0), overhang=(400.0, 300.0)), neck=0.5)
    fold = fold_of()
    q = (MIDDLE[0] - 0.4 * RADIUS, MIDDLE[1])
    relief = 300.0 * math.tan(math.radians(40.0)) / 2.0
    expected = TOP + float(fold(np.array([q[0]]), np.array([q[1]]))[0]) - relief + 0.5 * 300.0
    z, salt = built["meta"]["z_ref"], built["kwargs"]["salt"]
    assert z == pytest.approx(expected, abs=3.0) and built["meta"]["relief_m"] == pytest.approx(relief, abs=1e-6)
    x0 = q[0]
    assert not inside(salt, x0 - 30.0, MIDDLE[1], z) and inside(salt, x0 + 30.0, MIDDLE[1], z)
    assert inside(salt, x0 - 400.0 + 30.0, MIDDLE[1], z - 300.0) and not inside(salt, x0 - 400.0 - 30.0, MIDDLE[1], z - 300.0)
    assert built["meta"]["overhang"] == (400.0, 300.0)


def test_without_an_overhang_the_contact_is_a_waist_below_the_middle_of_the_reservoir():
    """No overhang: the reference depth is the fold's depth at the contact point plus half the reservoir plus the
    waist."""
    waist = 80.0
    built = flank(waist=waist)
    q = (MIDDLE[0] - 0.4 * RADIUS, MIDDLE[1])
    expected = TOP + float(fold_of()(np.array([q[0]]), np.array([q[1]]))[0]) + 0.5 * THICK + waist
    assert built["meta"]["z_ref"] == pytest.approx(expected, abs=3.0) and built["meta"]["overhang"] is None


def test_the_outline_keywords_reach_the_body():
    built = flank(body=dict(axes=(1000.0, 1000.0), lobes=0.18, rough=0.04, hurst=1.0, seed=9))
    salt = built["kwargs"]["salt"]
    assert salt.lobes == 0.18 and 0.0 < salt.rough_scale <= 1.0


# ---- the upturn: plain, and the sequences ----

def test_a_plain_upturn_keeps_the_exponent_that_holds_its_relief_under_the_cap():
    """width 600 m, dip 60: 600 tan(60) = 1039 m; over the cap of 350 m the exponent is 1039 / 350 = 2.97, and the relief
    1039 / 2.97 = 350 m. With the cap at 1,000 m the exponent stays at 2 and the relief is 519.6 m."""
    capped = flank(width=600.0, dip=60.0)["meta"]
    assert capped["power"] == pytest.approx(600.0 * math.tan(math.radians(60.0)) / 350.0)
    assert capped["relief_m"] == pytest.approx(350.0)
    free = flank(width=600.0, dip=60.0, relief_cap=1000.0)["meta"]
    assert free["power"] == 2.0 and free["relief_m"] == pytest.approx(600.0 * math.tan(math.radians(60.0)) / 2.0)


def test_a_plain_upturn_thins_the_layers_and_a_sequence_takes_its_thinning_from_the_unconformity():
    plain = flank(thinning=0.4, n_layers=3)["kwargs"]
    assert len(plain["isochore"]) == 3 and "erode_above" not in plain
    seq = flank(width=150.0, taper=62.0, angle=None, loss=0.3, dip=None)["kwargs"]
    assert "isochore" not in seq and "erode_above" in seq


def test_a_plain_upturn_thins_to_one_minus_the_share_inside_the_contact_and_not_at_all_beyond_the_zone():
    """The isochore factor is 1 - a (1 - d / W)^p: 1 - a = 0.6 in the salt (d <= 0), 1 beyond W = 300 m."""
    built = flank(thinning=0.4, n_layers=2)
    factor = built["kwargs"]["isochore"][0]
    salt = built["kwargs"]["salt"]
    assert float(factor(np.array([2680.0]), np.array([MIDDLE[1]]))[0]) == pytest.approx(0.6)
    assert float(factor(np.array([300.0]), np.array([MIDDLE[1]]))[0]) == pytest.approx(1.0)
    d = 150.0                                                             # half the zone: 1 - 0.4 (1 - 0.5)^2 = 0.9
    x = built["meta"]["contact"][0] - d
    assert float(factor(np.array([x]), np.array([MIDDLE[1]]))[0]) == pytest.approx(0.9, abs=0.01)
    assert salt.distance(np.array([x]), np.array([MIDDLE[1]]), built["meta"]["z_ref"])[0] == pytest.approx(d, abs=12.0)


def test_a_hook_that_keeps_the_interval_loses_its_share_of_the_thickness_at_the_wall():
    """taper 62 degrees: the beds meet the contact at atan(2 tan 62) = 75.1 degrees, so the relief of a 150 m zone is
    150 tan(75.1) / 2 = 282.1 m; losing 30 % of the 60 m reservoir at the wall takes a cut of 0.3 x 60 / 282.1 = 0.0638."""
    meta = flank(width=150.0, taper=62.0, loss=0.3, dip=None)["meta"]
    dip = math.degrees(math.atan(2.0 * math.tan(math.radians(62.0))))
    relief = 150.0 * math.tan(math.radians(dip)) / 2.0
    assert meta["dip"] == pytest.approx(dip) and meta["relief_m"] == pytest.approx(relief)
    assert relief == pytest.approx(282.1, abs=0.1)
    assert meta["cut"] == pytest.approx(0.3 * THICK / relief) and meta["cut"] == pytest.approx(0.0638, abs=1e-3)


def test_the_unconformity_of_a_pinching_hook_meets_the_beds_at_the_angle_asked_for():
    """taper 60 degrees: dip atan(2 tan 60) = 73.9; an angle of 72 needs the cut 1 - tan(73.9 - 72) / tan(73.9) = 0.9904,
    and the angle between beds and unconformity at the contact is dip - atan((1 - cut) tan dip) back again."""
    meta = flank(width=120.0, taper=60.0, angle=72.0, dip=None)["meta"]
    dip = math.degrees(math.atan(2.0 * math.tan(math.radians(60.0))))
    cut = 1.0 - math.tan(math.radians(dip - 72.0)) / math.tan(math.radians(dip))
    assert cut == pytest.approx(0.9904, abs=1e-4) and meta["cut"] == pytest.approx(cut)
    assert dip - math.degrees(math.atan((1.0 - meta["cut"]) * math.tan(math.radians(dip)))) == pytest.approx(72.0)
    assert meta["truncation_angle_deg"] == pytest.approx(72.0)
    assert meta["cut"] == pytest.approx(truncation_cut(60.0, 72.0))


def test_a_taper_that_leaves_a_flatter_dip_than_the_angle_is_refused():
    """A hook cut at over 70 degrees needs a taper above 54 degrees (dip atan(2 tan 54) = 70.0 degrees)."""
    assert sequence_dip(54.0) == pytest.approx(math.degrees(math.atan(2.0 * math.tan(math.radians(54.0)))))
    with pytest.raises(ValueError, match="cannot be cut at"):
        flank(width=120.0, taper=45.0, angle=72.0, dip=None)


@pytest.mark.parametrize("override,message", [
    (dict(dip=None), "dip"),                                              # plain without a dip
    (dict(taper=60.0, loss=0.2, angle=70.0, dip=None), "exactly one"),    # a sequence with both truncations
    (dict(taper=60.0, dip=None), "exactly one"),                          # a sequence with neither
    (dict(taper=60.0, loss=0.2), "dip"),                                  # a sequence takes no dip of its own
    (dict(loss=0.2), "taper"),                                            # a plain upturn takes no truncation
    (dict(wall_at=0.0), "wall_at"),
    (dict(wall_at=1.0), "wall_at"),                                       # the contact cannot pass the closure's far edge
    (dict(wall_at=1.5), "wall_at"),
    (dict(body=dict(axes=(1000.0,))), "axes"),
])
def test_arguments_that_do_not_say_one_thing_are_refused(override, message):
    with pytest.raises(ValueError, match=message):
        flank(**override)


def test_a_hook_that_pinches_the_reservoir_out_leaves_no_active_cell_within_the_cut_distance_of_the_salt():
    """Folding zone 200 m, taper 60, cut 0.9904: the lift at distance d from the contact is 346.4 (1 - d / 200)^2 m (relief
    200 tan(73.9) / 2), the unconformity removes 0.9904 of it, and the 60 m reservoir is gone where 0.9904 x lift >= 60, i.e.
    d <= 200 (1 - sqrt(60 / (0.9904 x 346.4))) = 116.4 m. A column is alive when its outer corner (half a column further out)
    holds rock: dead up to 91 m, alive from 116.4 + 25."""
    built = flank(width=200.0, taper=60.0, angle=72.0, dip=None, body=dict(axes=(1000.0, 1000.0)))
    relief = 200.0 * math.tan(math.radians(sequence_dip(60.0))) / 2.0
    cut = truncation_cut(60.0, 72.0)
    d_p = 200.0 * (1.0 - math.sqrt(THICK / (cut * relief)))
    assert relief == pytest.approx(346.4, abs=0.1) and d_p == pytest.approx(116.4, abs=0.1)
    model = layer()
    _, _, _, act = _build_geometry([model], **built["kwargs"])
    live = act.any(axis=2)[:, NY // 2]                                    # along x through the middle row
    salt = built["kwargs"]["salt"]
    xs = (np.arange(NX) + 0.5) * DX
    distance = salt.distance(xs, np.full(NX, MIDDLE[1] + 0.5 * DX), built["meta"]["z_ref"])
    west = xs < MIDDLE[0]
    dead_zone, live_zone = west & (distance > 0.0) & (distance < d_p - 0.5 * DX), west & (distance > d_p + 0.5 * DX)
    assert dead_zone.sum() >= 1 and live_zone.sum() >= 3                  # the test sees columns on both sides
    assert not live[dead_zone].any() and live[live_zone].all()


def test_the_cells_are_checked_against_the_upturn_zone_by_the_exporter(tmp_path):
    """A zone narrower than two cells is the exporter's to refuse, with the salt's own message."""
    built = flank(width=60.0)
    with pytest.raises(ValueError, match="wider than half"):
        to_grdecl(layer(), tmp_path / "m.grdecl", **built["kwargs"])


# ---- the faults ----

def test_radial_faults_follow_the_rate_per_km_of_contact():
    """The same wall with 0.5 and 2.5 radial faults per km of contact: the count grows with the rate (to within the
    kinds the density budget adds, the radial set alone is what moves)."""
    count = {rate: sum(f.kind == "radial" for f in flank(faults=dict(density=1.0, radial_rate=rate, seed=5))["kwargs"]["faults"])
             for rate in (0.5, 2.5)}
    assert count[2.5] > count[0.5] >= 0


def test_the_faults_are_the_salt_flank_style_on_the_fold_and_none_without_a_block():
    drawn = flank(faults=dict(density=1.0, radial_rate=1.5, seed=5))
    sets = drawn["meta"]["fault_sets"]
    assert drawn["meta"]["n_faults"] == len(drawn["kwargs"]["faults"]) > 0 and "radial" in sets
    assert flank()["kwargs"]["faults"] == [] and flank()["meta"]["n_faults"] == 0
    # drawn on the fold alone (no roughness) with the salt and the upturn the grid adds: their tip lines centre on the
    # reservoir as it lies lifted beside the salt
    salt = drawn["kwargs"]["salt"]
    expected = fold_faults("salt_flank", fold_of(), X_LEN, Y_LEN, DX, 1.0, TOP, THICK, seed=5, salt=salt,
                           upturn=salt_upturn(salt, 40.0, 300.0), radial_rate=1.5)
    assert [(f.center, f.z_center, f.kind) for f in drawn["kwargs"]["faults"]] == [(f.center, f.z_center, f.kind) for f in expected]


def test_the_builder_is_a_function_of_its_arguments():
    a = flank(faults=dict(density=1.0, seed=5), roughness=dict(sd=8.0, range_m=2000.0, seed=2))
    b = flank(faults=dict(density=1.0, seed=5), roughness=dict(sd=8.0, range_m=2000.0, seed=2))
    c = flank(faults=dict(density=1.0, seed=6), roughness=dict(sd=8.0, range_m=2000.0, seed=2))
    centres = lambda built: [f.center for f in built["kwargs"]["faults"]]
    assert a["meta"] == b["meta"] and centres(a) == centres(b) and centres(a) != centres(c)
    x, y = np.meshgrid(np.linspace(0.0, X_LEN, 9), np.linspace(0.0, Y_LEN, 7), indexing="ij")
    assert np.array_equal(a["kwargs"]["structure"](x, y), b["kwargs"]["structure"](x, y))


# ---- the model as the exporter sees it ----

def test_a_flank_has_dead_columns_on_the_salt_side_and_a_trap_against_the_wall(tmp_path):
    """Salt toward +x, a stock of 1,000 m radius about x = 2680: its middle column is dead and none of the first twenty (x <
    1000) is; the trap the salt leaves is a three-way closure against the wall and its crest lies on the far side of the
    contact."""
    built = flank(faults=dict(density=0.5, seed=5))
    model = layer()
    _, _, _, act = _build_geometry([model], **built["kwargs"])
    dead = ~act.any(axis=2)
    assert dead[int(2680.0 // DX), NY // 2] and not dead[:20].any()      # the stock's middle is dead, the far side alive
    report = {}
    to_grdecl(model, tmp_path / "m.grdecl", report=report, **built["kwargs"])
    blocks = blocks_at(report["block_inputs"], DX, DX, Capillary(delta_rho=300.0))
    main = max(blocks, key=lambda b: b["area"] * b["height"])
    assert main["height"] > 10.0 and main["area"] > 1e5
    x_crest = (main["crest"][0] + 0.5) * DX
    assert x_crest < MIDDLE[0] - 0.4 * RADIUS + 60.0                        # beside the wall, not inside the salt
    assert built["meta"]["crest"][0] == pytest.approx(x_crest, abs=300.0)  # the label points at the same corner
    assert not inside(built["kwargs"]["salt"], *built["meta"]["crest"], built["meta"]["z_ref"])   # not under the salt


def test_the_outline_can_be_cut_from_the_structure_with_the_upturn_and_is_otherwise_the_folds(tmp_path):
    """ResSimMill's review B5: the cut is read on the fold and its roughness, which the upturn's ridge along the wall is not
    part of, so a wide folding zone leaves half of the main trap of the built model outside it (width 500 m, wall at 0.55:
    53 % of its 1,308 columns). With ``upturn_in_outline`` the trap is read on the structure written, from the crest of the
    trap the salt leaves, and the cut holds it; without it (the default) the outline is what it was."""
    args = dict(rim=500.0, width=500.0, wall_at=0.55, normal=0.0, dip=40.0)
    default, upturned = flank(**args), flank(upturn_in_outline=True, **args)
    assert default["kwargs"]["outline"].sum() == flank(upturn_in_outline=False, **args)["kwargs"]["outline"].sum()
    assert upturned["meta"]["outline_columns"] == int(upturned["kwargs"]["outline"].sum()) > int(
        default["kwargs"]["outline"].sum())
    shares = {}
    for name, built in (("default", default), ("upturned", upturned)):
        report = {}
        to_grdecl(layer(), tmp_path / f"{name}.grdecl", report=report, **built["kwargs"])
        main = max(blocks_at(report["block_inputs"], DX, DX, Capillary(delta_rho=300.0)),
                   key=lambda b: b["area"] * b["height"])
        shares[name] = float((main["mask"] & ~built["kwargs"]["outline"]).sum() / main["mask"].sum())
    assert shares["default"] > 0.4 and shares["upturned"] < 0.01


def test_an_overhang_on_a_sequence_hangs_from_the_beds_as_the_unconformity_leaves_them():
    """The overhang's reference depth is ``neck`` of its height H below the lifted beds' top at the contact, and on a
    sequence the lift is what the unconformity leaves: ``relief (1 - cut)``. A hook cut by 0.5 (loss 0.2 of 60 m) over
    a 346 m lift hangs its overhang 173 m lower than one that ignored the cut would."""
    built = flank(width=200.0, taper=60.0, angle=None, loss=0.9, dip=None,
                  body=dict(axes=(1000.0, 1000.0), overhang=(300.0, 200.0)), neck=0.6)
    meta = built["meta"]
    assert 0.05 < meta["cut"] < 1.0
    fold = fold_of()
    z_top = TOP + float(fold(np.array([meta["contact"][0]]), np.array([meta["contact"][1]]))[0])
    lifted = z_top - meta["relief_m"] * (1.0 - meta["cut"])
    assert meta["z_ref"] == pytest.approx(lifted + 0.6 * 200.0, abs=0.01)
    assert abs(meta["z_ref"] - (z_top - meta["relief_m"] + 0.6 * 200.0)) > 0.05 * meta["relief_m"]


def test_the_labels_say_what_was_built():
    meta = flank(faults=dict(density=1.0, seed=5))["meta"]
    assert meta["closure_area_km2"] == pytest.approx(math.pi * RADIUS ** 2 / 1e6, rel=0.08)
    assert meta["closure_height_m"] == pytest.approx(80.0, abs=2.0)
    assert (meta["normal"], meta["wall_at"], meta["width"], meta["upturn"]) == (0.0, 0.4, 300.0, "plain")
    assert meta["power"] == 2.0 and meta["dip"] == 40.0 and meta["thinning"] == 0.0
    assert all(isinstance(v, (int, float, str, bool, type(None), list, tuple)) for v in meta.values())


# ---- the outline cut ----

def test_the_rim_cuts_the_outline_the_way_fold_trap_cuts_it():
    """The closure and a rim of 600 m: the same columns fold_trap keeps for the same closure and roughness (the salt
    covers some of them: the exporter writes those inactive anyway)."""
    low = dict(CLOSURE, height=10.0)                                      # a low relief that the roughness moves about
    rough = dict(sd=6.0, range_m=1500.0, seed=2)
    ours = flank(closure=low, roughness=rough, rim=600.0)
    theirs = fold_trap("four_way", X_LEN, Y_LEN, DX, TOP, THICK, low, rough, None, rim=600.0)
    smooth = fold_trap("four_way", X_LEN, Y_LEN, DX, TOP, THICK, low, None, None, rim=600.0)
    assert np.array_equal(ours["kwargs"]["outline"], theirs["kwargs"]["outline"])
    assert not np.array_equal(ours["kwargs"]["outline"], smooth["kwargs"]["outline"])     # the roughness is read
    assert ours["meta"]["rim"] == 600.0 and ours["meta"]["outline_columns"] == int(ours["kwargs"]["outline"].sum())
    assert "outline" not in flank()["kwargs"]


# ---- subsalt ----

def trap_of(closure=CLOSURE, rough=None):
    """The trap at the fold's crest on fold plus roughness, read here with closure_stats."""
    fold = fold_of(closure)
    x, y = np.meshgrid((np.arange(NX) + 0.5) * DX, (np.arange(NY) + 0.5) * DX, indexing="ij")
    depth = TOP + fold(x, y) + (0.0 if rough is None else rough(x, y))
    crest = (int((MIDDLE[0] + fold.crest_offset[0]) // DX), int((MIDDLE[1] + fold.crest_offset[1]) // DX))
    return st.closure_stats(depth, DX, DX, crest=crest), depth


def test_a_truncated_base_cuts_the_trap_by_its_share_of_the_relief_at_the_crest():
    """The base of the salt lies at crest depth + cut x relief at the crest column (rugosity zero there by construction)."""
    trap, depth = trap_of()
    built = subsalt(cut=0.5)
    base = built["kwargs"]["erode_above"]
    crest = trap["crest"]
    at = built["meta"]["crest"]                  # the four columns about the crest tie: the composer's pick is its own
    expected = depth[crest] + 0.5 * trap["height"]
    assert float(base(np.array([at[0]]), np.array([at[1]]))[0]) == pytest.approx(expected, abs=0.01)
    assert built["meta"]["depth_base_at_crest"] == pytest.approx(expected, abs=0.01)
    assert built["meta"]["closure_height_m"] == pytest.approx(trap["height"]) and built["meta"]["crest_depth_m"] == pytest.approx(depth[crest])


def test_the_crest_depth_is_that_of_the_crest_column_not_the_maps_shallowest():
    """On a low relief with rough horizon the map's shallowest column (a micro-high far from the trap) is not the crest
    column of the trap: the label's ``crest_depth_m`` is the depth of the column ``crest`` names, and the base is placed
    on that. The first seed of 1-40 whose map minimum lies over a metre above it is used."""
    low = dict(CLOSURE, height=20.0)
    x, y = np.meshgrid((np.arange(NX) + 0.5) * DX, (np.arange(NY) + 0.5) * DX, indexing="ij")
    for seed in range(1, 41):
        rough = dict(sd=6.0, range_m=800.0, seed=seed)
        depth = TOP + fold_of(low)(x, y) + st.roughness(x_len=X_LEN, y_len=Y_LEN, **rough)(x, y)
        built = subsalt(closure=low, roughness=rough, cut=0.5)
        at = built["meta"]["crest"]
        cell = (int(at[0] // DX), int(at[1] // DX))
        if depth.min() < depth[cell] - 1.0:
            assert built["meta"]["crest_depth_m"] == pytest.approx(float(depth[cell]), abs=1e-6)
            assert built["meta"]["crest_depth_m"] > depth.min() + 1.0
            return
    pytest.fail("no rough low relief of seeds 1-40 has a shallower column than its trap's crest")


def test_the_trap_is_read_on_the_rough_surface_at_the_shallowest_column_near_the_crest():
    """On a low relief with 6 m of roughness the shallowest column within two of the fold's crest column is not that column:
    the base is placed on the crest of the trap there, read off the same depth map with closure_stats."""
    low = dict(CLOSURE, height=20.0)
    rough = dict(sd=6.0, range_m=1500.0, seed=2)
    field = st.roughness(x_len=X_LEN, y_len=Y_LEN, **rough)
    fold = fold_of(low)
    x, y = np.meshgrid((np.arange(NX) + 0.5) * DX, (np.arange(NY) + 0.5) * DX, indexing="ij")
    depth = TOP + fold(x, y) + field(x, y)
    ci, cj = int(MIDDLE[0] // DX), int(MIDDLE[1] // DX)
    window = depth[ci - 2:ci + 3, cj - 2:cj + 3]
    i, j = np.unravel_index(int(np.argmin(window)), window.shape)
    cell = (ci - 2 + int(i), cj - 2 + int(j))
    assert cell != (ci, cj)                                               # the window search matters on this surface
    trap = st.closure_stats(depth, DX, DX, crest=cell)
    built = subsalt(closure=low, roughness=rough, cut=0.5)
    assert built["meta"]["crest_depth_m"] == pytest.approx(depth[cell]) and built["meta"]["closure_height_m"] == pytest.approx(trap["height"])
    assert built["meta"]["depth_base_at_crest"] == pytest.approx(depth[cell] + 0.5 * trap["height"])
    assert built["meta"]["crest"] == pytest.approx(((cell[0] + 0.5) * DX, (cell[1] + 0.5) * DX))


def test_a_cover_lies_a_margin_above_the_crest_and_cuts_nothing_there(tmp_path):
    trap, depth = trap_of()
    built = subsalt(kind="cover", margin=200.0, cut=None)
    base = built["kwargs"]["erode_above"]
    crest = trap["crest"]
    at = built["meta"]["crest"]
    assert float(base(np.array([at[0]]), np.array([at[1]]))[0]) == pytest.approx(depth[crest] - 200.0, abs=0.01)
    model = layer()
    without = {k: v for k, v in built["kwargs"].items() if k != "erode_above"}
    assert _build_geometry([model], **built["kwargs"])[3].sum() == _build_geometry([model], **without)[3].sum()
    truncated = subsalt(kind="truncated", cut=0.5)["kwargs"]
    assert _build_geometry([model], **truncated)[3].sum() < _build_geometry([model], **{k: v for k, v in truncated.items() if k != "erode_above"})[3].sum()


def test_the_base_dips_toward_its_azimuth_by_the_tangent_of_the_dip():
    """Dip 10 degrees toward azimuth 90 (ramp's convention: the base deepens along (sin az, cos az) = +x): 500 m along it
    the base is 500 tan(10) = 88.2 m deeper (a plane when the rugosity is zero), and level across it."""
    built = subsalt(base_dip=10.0, base_azimuth=90.0)
    base = built["kwargs"]["erode_above"]
    x0, y0 = MIDDLE
    near = float(base(np.array([x0]), np.array([y0]))[0])
    far = float(base(np.array([x0 + 500.0]), np.array([y0]))[0])
    assert far - near == pytest.approx(500.0 * math.tan(math.radians(10.0)), abs=1e-6)
    side = float(base(np.array([x0]), np.array([y0 + 500.0]))[0])
    assert side == pytest.approx(near)                                    # level across the dip direction


def test_the_sheet_is_a_label_and_the_rugosity_is_zero_at_the_crest():
    rough = subsalt(rugosity_sd=20.0, sheet=1234.0)
    base = rough["kwargs"]["erode_above"]
    assert base.thickness == 1234.0 and rough["meta"]["sheet_m"] == 1234.0
    trap, depth = trap_of()
    crest, at = trap["crest"], rough["meta"]["crest"]
    assert float(base(np.array([at[0]]), np.array([at[1]]))[0]) == pytest.approx(depth[crest] + 0.5 * trap["height"], abs=0.01)
    xs = np.linspace(0.0, X_LEN, 40)
    assert np.std(base(xs, np.full_like(xs, MIDDLE[1]))) > 5.0            # and rugose elsewhere (sd 20 m)


def test_subsalt_faults_are_the_faulted_anticlines_and_the_rim_cuts_the_outline():
    built = subsalt(faults=dict(density=1.0, seed=7), roughness=dict(sd=8.0, range_m=2000.0, seed=2), rim=500.0)
    expected = fold_faults("faulted_anticline", fold_of(), X_LEN, Y_LEN, DX, 1.0, TOP, THICK, seed=7)
    assert [f.center for f in built["kwargs"]["faults"]] == [f.center for f in expected]
    assert built["kwargs"]["outline"].shape == (NX, NY) and built["meta"]["rim"] == 500.0
    assert set(built["kwargs"]) == {"structure", "faults", "erode_above", "outline"}


def test_subsalt_refuses_an_unknown_kind_and_a_missing_share():
    with pytest.raises(ValueError, match="kind"):
        subsalt(kind="sheet")
    with pytest.raises(ValueError, match="cut"):
        subsalt(kind="truncated", cut=None)
    with pytest.raises(ValueError, match="margin"):
        subsalt(kind="cover", cut=None, margin=None)
