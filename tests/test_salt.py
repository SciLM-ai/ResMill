"""Salt bodies: their outline at depth (lean, flare, lobes), the signed distance to the contact, and the upturn and
thinning terms beside them. Every number is computed by hand from the geometry or taken from the structure research
(step5_salt.md: N9, N11-N15, N30)."""
import math

import numpy as np
import pytest
from scipy import ndimage

from resmill import salt as sl
from resmill import structure as st

R = 300.0
CX, CY = 1000.0, 1000.0


def grid(step=5.0, half=900.0):
    """Cell centres of a square map around the body, ``step`` m apart."""
    xs = np.arange(CX - half, CX + half + 1e-9, step) + 0.5 * step
    return np.meshgrid(xs, xs + (CY - CX), indexing="ij")


def stock(**kw):
    return sl.salt_body((CX, CY), (R, R), **kw)


def test_a_circle_covers_pi_r_squared():
    X, Y = grid()
    area = stock().inside(X, Y, 0.0).sum() * 5.0 ** 2
    assert area == pytest.approx(math.pi * R ** 2, rel=0.01)


def test_an_ellipse_covers_pi_a_b_and_azimuth_turns_the_long_axis():
    a, b = 600.0, 100.0
    X, Y = grid(step=4.0, half=900.0)
    area = sl.salt_body((CX, CY), (a, b)).inside(X, Y, 0.0).sum() * 4.0 ** 2
    assert area == pytest.approx(math.pi * a * b, rel=0.01)
    wall = sl.salt_body((CX, CY), (a, b), azimuth=90.0)             # the package convention: 90 runs along -y
    assert wall.inside(CX, CY - 0.9 * a, 0.0) and wall.inside(CX, CY + 0.9 * a, 0.0)
    assert not wall.inside(CX + 0.9 * a, CY, 0.0)
    skew = sl.salt_body((CX, CY), (a, b), azimuth=45.0)             # along (cos 45, -sin 45)
    s = 0.9 * a / math.sqrt(2.0)
    assert skew.inside(CX + s, CY - s, 0.0) and not skew.inside(CX + s, CY + s, 0.0)


def test_the_superellipse_exponent_squares_the_outline():
    # |x/R|^8 + |y/R|^8 < 1 covers 4 Gamma(1 + 1/8)^2 / Gamma(1 + 2/8) = 3.914 R^2 of the 4 R^2 square around it
    square = sl.salt_body((CX, CY), (R, R), shape=8.0)
    X, Y = grid()
    assert square.inside(X, Y, 0.0).sum() * 25.0 == pytest.approx(3.914 * R ** 2, rel=0.01)


def test_lean_moves_the_outline_by_cot_dip_per_metre_of_depth():
    cot = 1.0 / math.tan(math.radians(60.0))                         # a wall dipping 60 degrees toward +x
    body = stock(z_ref=2000.0, lean=(cot, 0.0))
    shift = cot * 100.0                                              # 57.735 m 100 m deeper
    assert body.inside(CX + R + shift - 1.0, CY, 2100.0) and not body.inside(CX + R + shift + 1.0, CY, 2100.0)
    assert body.inside(CX - R - shift + 1.0, CY, 1900.0) and not body.inside(CX - R - shift - 1.0, CY, 1900.0)
    assert body.inside(CX + R - 1.0, CY, 2000.0) and not body.inside(CX + R + 1.0, CY, 2000.0)


def test_a_negative_flare_is_an_overhang_a_positive_one_a_pedestal():
    z = np.array([1800.0, 1900.0, 2000.0, 2100.0, 2200.0])
    over, pedestal = stock(z_ref=2000.0, flare=-0.2), stock(z_ref=2000.0, flare=0.2)
    x = CX + R + 10.0                                                # 10 m outside the contact at the reference depth
    assert list(over.inside(x, CY, z)) == [True, True, False, False, False]       # R + 0.2 (2000 - z) > R + 10 above 1950
    assert list(pedestal.inside(x, CY, z)) == [False, False, False, True, True]   # R + 0.2 (z - 2000) > R + 10 below 2050


def test_an_overhang_widens_the_body_by_its_lateral_extent_over_its_height_above_the_neck():
    """``overhang=(L, H)``: the outline is the plain one at the neck (z_ref) and below it, L wider at H above it and higher, and
    linear between (the underside of an overhang L wide, H high, which dips atan(H / L) from horizontal). Hand-computed for
    R = 300 m, L = 100, H = 200, neck at 2000 m: radius 300 at 2000 and below, 325 at 1950, 350 at 1900, 400 at 1800 and above."""
    body = stock(z_ref=2000.0, overhang=(100.0, 200.0))
    for z, expected in ((2200.0, 300.0), (2000.0, 300.0), (1950.0, 325.0), (1900.0, 350.0), (1800.0, 400.0), (1500.0, 400.0)):
        out = body.outline(z)
        assert np.hypot(out[:, 0] - CX, out[:, 1] - CY) == pytest.approx(expected, abs=0.05)
    assert body.inside(CX + 349.0, CY, 1900.0) and not body.inside(CX + 351.0, CY, 1900.0)
    plain = stock(z_ref=2000.0)
    assert (plain.outline(1800.0) == stock(z_ref=2000.0, overhang=None).outline(1800.0)).all()
    both = stock(z_ref=2000.0, flare=-0.1, overhang=(100.0, 200.0))                  # the two add: 300 + 10 + 50 at 1900
    assert np.hypot(*(both.outline(1900.0) - [CX, CY]).T) == pytest.approx(360.0, abs=0.05)


@pytest.mark.parametrize("bad", [(0.0, 100.0), (100.0, 0.0), (-5.0, 100.0), (100.0,)])
def test_an_overhang_that_cannot_exist_is_refused(bad):
    with pytest.raises(ValueError):
        stock(overhang=bad)


def test_distance_is_signed_exact_and_unit_slope_for_a_circle():
    body = stock()
    rho = np.array([0.0, 100.0, 299.0, 301.0, 450.0, 800.0])
    got = body.distance(CX + rho, np.full(rho.size, CY))
    assert got == pytest.approx(rho - R, abs=0.3)                    # negative inside, positive outside, zero on the outline
    for angle in (0.3, 1.9, 4.0):
        p = np.array([CX + 420.0 * math.cos(angle), CY + 420.0 * math.sin(angle)])
        h = 2.0
        gx = (body.distance(p[0] + h, p[1]) - body.distance(p[0] - h, p[1])) / (2.0 * h)
        gy = (body.distance(p[0], p[1] + h) - body.distance(p[0], p[1] - h)) / (2.0 * h)
        assert math.hypot(gx, gy) == pytest.approx(1.0, abs=0.02)


def test_distance_to_a_straight_wall_is_the_perpendicular_distance_and_follows_the_depth():
    cot = 1.0 / math.tan(math.radians(60.0))
    wall = sl.salt_body((CX, CY), (4000.0, 500.0), z_ref=2000.0, lean=(0.0, cot))      # long axis along x, leaning to +y
    assert wall.distance(CX, CY + 540.0) == pytest.approx(40.0, abs=0.1)            # 40 m outside the flank at y = CY + 500
    flank = 500.0 * math.sqrt(1.0 - (300.0 / 4000.0) ** 2)                          # 300 m along, the ellipse has curved away
    assert wall.distance(CX + 300.0, CY + 540.0) == pytest.approx(540.0 - flank, abs=0.1)
    assert wall.distance(CX, CY + 540.0, 2100.0) == pytest.approx(40.0 - cot * 100.0, abs=0.3)   # moved cot * 100 m outward


def test_lobes_zero_is_the_plain_shape_and_a_seed_makes_a_body_reproducible():
    X, Y = grid()
    plain = stock().inside(X, Y, 0.0)
    assert (sl.salt_body((CX, CY), (R, R), lobes=0.0).inside(X, Y, 0.0) == plain).all()
    a = sl.salt_body((CX, CY), (R, R), lobes=0.15, seed=4).inside(X, Y, 0.0)
    assert (a == sl.salt_body((CX, CY), (R, R), lobes=0.15, seed=4).inside(X, Y, 0.0)).all()
    assert (a != sl.salt_body((CX, CY), (R, R), lobes=0.15, seed=5).inside(X, Y, 0.0)).any()
    assert (a != plain).any()
    with pytest.raises(ValueError, match="seed"):
        sl.salt_body((CX, CY), (R, R), lobes=0.1)


def test_lobes_make_the_outline_irregular_by_that_fraction_of_the_radius():
    rms = []
    for seed in range(24):
        body = sl.salt_body((CX, CY), (R, R), lobes=0.1, seed=seed)
        out = body.outline()
        rms.append(np.std(np.hypot(out[:, 0] - CX, out[:, 1] - CY)))
    assert np.mean(rms) == pytest.approx(0.1 * R, rel=0.35)         # about lobes x radius, the same in every direction


RAD = 3000.0                                                        # the radius of the bodies of the roughness tests


def residual(body, z=None):
    """The outline's radial residual (m) from a circle of radius RAD about the body's centre, and its polar angle."""
    out = body.outline(z)
    return np.hypot(out[:, 0] - CX, out[:, 1] - CY) - RAD, np.arctan2(out[:, 1] - CY, out[:, 0] - CX)


def octaves(radius, hurst, rough):
    """Hand-computed octaves of the outline's roughness: ranges radius, radius / 2, ... down to MIN_RANGE, and the sd of
    each, rough x radius x (a / radius)^hurst."""
    ranges = [radius / 2 ** i for i in range(10) if radius / 2 ** i >= sl.MIN_RANGE]
    return ranges, [rough * radius * (a / radius) ** hurst for a in ranges]


@pytest.mark.parametrize("hurst", [1.0, 1.5])
def test_rough_makes_the_outline_irregular_by_the_drawn_sd_octave_by_octave(hurst):
    """``rough`` is the sd of the outline's displacement (fraction of the radius) at a range of one radius, halving in range
    and falling as range^hurst down to MIN_RANGE (150 m): the Santos stock (H2, Coleman et al. 2018) and the Sigsbee feeders
    (Duffy et al. 2019) show about 4-6 % of the radius at wavelengths of 1-2 radii and 2-3 times less per octave. Hand-computed:
    a 3 km radius, rough 0.04: octaves of range 3000, 1500, 750, 375, 187.5 m with sd 120 m x (range / 3000)^hurst."""
    ranges, sds = octaves(RAD, hurst, 0.04)
    assert ranges == [3000.0, 1500.0, 750.0, 375.0, 187.5]
    assert sds[0] == pytest.approx(120.0) and sds[1] == pytest.approx(120.0 * 0.5 ** hurst)
    expected = math.sqrt(sum(x * x for x in sds))
    rms = []
    for seed in range(24):
        rho, _ = residual(sl.salt_body((CX, CY), (RAD, RAD), rough=0.04, hurst=hurst, seed=seed))
        rms.append(math.sqrt(np.mean(rho ** 2)))
    assert np.mean(rms) == pytest.approx(expected, rel=0.2)          # one curve through the field: few independent samples
    assert 0.4 * expected < min(rms) and max(rms) < 1.8 * expected


def test_a_long_body_is_irregular_at_the_scale_of_its_length_not_only_of_its_width():
    """The octaves start at the larger of the radius and half the long axis, so that a wall meanders over its length as the
    Precaspian walls and the Sigsbee feeders do (sd rises with range as range^hurst): a 6 km x 1 km wall with rough 0.03 has
    octaves of range 3000, 1500, 750, 375 and 187.5 m with sd 90, 45, 22.5, 11 and 5.6 m (hurst 1): 102 m in all, read as the
    outline's distance from the plain ellipse."""
    plain = sl.salt_body((CX, CY), (6000.0, 1000.0))
    expected = math.sqrt(sum(x * x for x in (90.0, 45.0, 22.5, 11.25, 5.625)))
    rms = []
    for seed in range(16):
        out = sl.salt_body((CX, CY), (6000.0, 1000.0), rough=0.03, hurst=1.0, seed=seed).outline()
        rms.append(math.sqrt(np.mean(plain.distance(out[:, 0], out[:, 1]) ** 2)))
    assert np.mean(rms) == pytest.approx(expected, rel=0.25)


def test_rough_adds_short_wavelengths_in_the_proportion_the_hurst_exponent_sets():
    """With hurst 0 every octave has the same sd, so waves shorter than 1.2 km hold a larger share of the outline's variance
    than with hurst 1.5, where the coarse octaves dominate (hand-computed from the octave sds: the 375 and 187.5 m octaves
    carry 2 of 5 equal sds at hurst 0 and 0.002 of the variance at 1.5)."""
    short = {}
    for hurst in (0.0, 1.5):
        share = []
        for seed in range(12):
            rho, theta = residual(sl.salt_body((CX, CY), (RAD, RAD), rough=0.03, hurst=hurst, seed=seed))
            order = np.argsort(theta)
            t = np.linspace(-math.pi, math.pi, 4096, endpoint=False)
            rho_t = np.interp(t, theta[order], rho[order], period=2.0 * math.pi)
            F = np.fft.rfft(rho_t - rho_t.mean())
            wavelength = 2.0 * math.pi * RAD / np.maximum(np.arange(len(F)), 1)             # m along the circle's circumference
            power = np.abs(F) ** 2
            share.append(power[wavelength < 1200.0].sum() / power[1:].sum())
        short[hurst] = np.mean(share)
    assert short[0.0] > 2.0 * short[1.5]


def test_rough_zero_is_the_plain_shape_and_the_roughness_is_reproducible_and_validated():
    X, Y = grid()
    plain = stock().inside(X, Y, 0.0)
    assert (sl.salt_body((CX, CY), (R, R), rough=0.0).inside(X, Y, 0.0) == plain).all()
    a = sl.salt_body((CX, CY), (R, R), rough=0.03, seed=4).inside(X, Y, 0.0)
    assert (a == sl.salt_body((CX, CY), (R, R), rough=0.03, seed=4).inside(X, Y, 0.0)).all()
    assert (a != sl.salt_body((CX, CY), (R, R), rough=0.03, seed=5).inside(X, Y, 0.0)).any() and (a != plain).any()
    for bad in (dict(rough=0.03), dict(rough=-0.01, seed=1), dict(rough=0.2, seed=1), dict(rough=0.03, hurst=-1.0, seed=1)):
        with pytest.raises(ValueError):
            sl.salt_body((CX, CY), (R, R), **bad)


@pytest.mark.parametrize("kw", [dict(rough=0.05, hurst=1.2), dict(rough=0.05, hurst=1.0, lobes=0.15), dict(rough=0.04, hurst=1.0, shape=3.0)])
def test_a_rough_outline_stays_one_curve_without_islands_or_holes(kw):
    """The warp that roughens the outline does not fold: the salt cells' area on a raster matches the polygon the outline
    traces (shoelace), and the cells form one connected body without holes."""
    from scipy import ndimage
    for seed in range(8):
        body = sl.salt_body((CX, CY), (2.0 * R, 1.2 * R), azimuth=30.0, seed=seed, **kw)
        step = 15.0
        xs = np.arange(CX - 3.0 * R, CX + 3.0 * R, step) + 0.5 * step
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        inside = body.inside(X, Y, 0.0)
        _, n = ndimage.label(inside)
        _, holes = ndimage.label(~inside)
        out = body.outline()
        x, y = out[:, 0], out[:, 1]
        poly_area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        assert n == 1 and holes == 1
        assert inside.sum() * step ** 2 == pytest.approx(poly_area, rel=0.02)


def test_a_capped_distance_is_the_distance_within_the_cap_and_the_cap_beyond_it():
    body = stock()
    p = np.array([CX + 100.0, CX + 350.0, CX + 700.0, CX + 1100.0])           # 200 m inside, 50, 400 and 800 m outside
    exact, capped = body.distance(p, CY), body.distance(p, CY, cap=300.0)
    assert capped == pytest.approx([exact[0], exact[1], 300.0, 300.0], abs=0.01)
    assert exact == pytest.approx([-200.0, 50.0, 400.0, 800.0], abs=0.3)
    assert body.distance(CX, CY, cap=100.0) == -100.0                         # deep inside: minus the cap


def test_the_outline_is_a_closed_curve_on_which_the_distance_vanishes():
    body = sl.salt_body((CX, CY), (900.0, 300.0), azimuth=30.0, lobes=0.12, seed=2, shape=2.6, lean=(0.3, -0.1),
                        flare=-0.1, z_ref=2000.0)
    out = body.outline(2150.0)
    assert np.abs(body.distance(out[::97, 0], out[::97, 1], 2150.0)).max() < 0.5
    assert np.hypot(*np.diff(np.vstack([out, out[:1]]), axis=0).T).max() < 40.0         # no gap in the curve
    assert body.inside(*out.mean(axis=0), 2150.0)


def exact_distance(body, step, cap):
    """The cells (x, y) within ``cap`` of the body's contact on a raster of ``step`` m, and the signed distance of each to it
    by an exact Euclidean distance transform of the mask ``inside`` gives (negative inside; half a pixel from the pixel
    centres to the edge between them, so good to half a pixel)."""
    half = 1.3 * max(body.axes) + 1500.0
    xs = np.arange(-half, half, step) + 0.5 * step
    X, Y = np.meshgrid(xs + body.center[0], xs + body.center[1], indexing="ij")
    inside = body.inside(X, Y)
    d = np.where(inside, -ndimage.distance_transform_edt(inside, sampling=step) + 0.5 * step,
                 ndimage.distance_transform_edt(~inside, sampling=step) - 0.5 * step)
    near = np.abs(d) < cap
    return X[near], Y[near], np.clip(d[near], -cap, cap)


LOBATE = {"wall": dict(axes=(6000.0, 1200.0), azimuth=25.0, lobes=0.18, rough=0.04, seeds=(1, 2)),
          "stock": dict(axes=(1710.0, 1315.0), azimuth=40.0, lobes=0.3, rough=0.04, seeds=(2, 5))}      # R 1500 m, axial ratio 1.3


@pytest.mark.parametrize("kind,seed", [(k, s) for k, v in LOBATE.items() for s in v["seeds"]])
def test_the_distance_of_a_lobate_wall_or_stock_is_the_exact_one_near_its_contact(kind, seed):
    """A wall with bays of 1-2 km (lobes 0.18 of its half-width, roughness 0.04) and a stock with lobes 0.3 are not star-shaped
    about their centres, so rays from the centre skip the bays and the polyline chords across them (up to 1.4 km): every cell
    within the 300 m folding zone must still be within a pixel of the distance an exact transform of the mask gives (the old
    ray-cast polyline missed it by 229 m on the wall of seed 1, 296 m on the stock of seed 5)."""
    kw = dict(LOBATE[kind])
    kw.pop("seeds")
    body = sl.salt_body((CX, CY), hurst=1.0, seed=seed, **kw)
    step = 10.0
    X, Y, exact = exact_distance(body, step, 300.0)
    assert exact.size > 50000                                                             # a band of 300 m on both sides
    assert np.abs(body.distance(X, Y, cap=300.0) - exact).max() < step
    out = body.outline()
    assert np.hypot(*np.diff(np.vstack([out, out[:1]]), axis=0).T).max() < 1.5 * max(sl._MIN_STEP, min(kw["axes"]) / sl._GRID)
    assert np.abs(body.distance(out[:, 0], out[:, 1])).max() < 0.1 * step                    # every vertex lies on the contact


@pytest.mark.parametrize("kind", ["wall", "stock"])
def test_the_upturn_meets_a_lobate_contact_at_the_drawn_dip_all_along_it(kind):
    """Along the whole contact, whatever its bays, the strata rise toward the salt at the drawn dip: the slope of the upturn along
    the contact's own normal, a few metres outside it, is tan(dip) (1 - d / W) at p = 2."""
    kw = dict(LOBATE[kind])
    body = sl.salt_body((CX, CY), hurst=1.0, seed=kw.pop("seeds")[0], **kw)
    dip, width = 35.0, 400.0
    term = sl.salt_upturn(body, dip, width)
    out = body.outline()[::max(len(body.outline()) // 60, 1)]
    n = body.normal(out[:, 0], out[:, 1])
    d1, d2 = 1.0, 5.0
    slope = (term(*(out + d2 * n).T) - term(*(out + d1 * n).T)) / (d2 - d1)               # dz / dd, positive up
    assert np.degrees(np.arctan(slope)) == pytest.approx(dip, abs=1.5)


def test_the_normal_is_the_gradient_of_the_contact_whatever_the_spacing_of_its_vertices():
    """For an ellipse of semi-axes 900 and 300 m the outward normal at (900 cos t, 300 sin t) is the unit vector of
    (cos t / 900, sin t / 300); rotated by the azimuth, in the package's convention (the long axis along (cos az, -sin az))."""
    az = math.radians(30.0)
    body = sl.salt_body((CX, CY), (900.0, 300.0), azimuth=30.0)
    t = np.array([0.0, 0.7, 2.0, 4.1])
    u, v = 900.0 * np.cos(t), 300.0 * np.sin(t)                                              # along and across the long axis
    x, y = CX + u * math.cos(az) + v * math.sin(az), CY - u * math.sin(az) + v * math.cos(az)
    nu, nv = np.cos(t) / 900.0, np.sin(t) / 300.0
    nu, nv = nu / np.hypot(nu, nv), nv / np.hypot(nu, nv)
    expected = np.column_stack([nu * math.cos(az) + nv * math.sin(az), -nu * math.sin(az) + nv * math.cos(az)])
    assert body.normal(x, y) == pytest.approx(expected, abs=1e-4)
    assert body.normal(x[1], y[1]).shape == (2,)


@pytest.mark.parametrize("seed", range(12))
def test_the_contact_traces_every_curve_of_the_body_and_the_salt_lies_on_its_left(seed):
    """The signed area of all the loops of the contact (counterclockwise round salt, clockwise round a hole) is the area of the
    mask on a raster: nothing of a folded body (an island, a hole) is missed or doubled, and the box that is contoured holds
    the whole body, for walls and stocks with any lobes (to 0.3), roughness (to 0.05) and Hurst exponent."""
    rng = np.random.default_rng(seed)
    b = float(rng.uniform(500.0, 1500.0))
    axes = (b * float(rng.uniform(1.0, 5.0)), b)
    body = sl.salt_body((CX, CY), axes, azimuth=float(rng.uniform(0.0, 180.0)), lobes=float(rng.uniform(0.05, 0.3)),
                        rough=float(rng.uniform(0.02, 0.05)), hurst=float(rng.uniform(0.0, 1.5)), seed=seed,
                        shape=float(rng.uniform(2.0, 3.0)))
    loops = body._contact(body.z_ref)[0]
    area = sum(0.5 * np.sum(l[:, 0] * np.roll(l[:, 1], -1) - np.roll(l[:, 0], -1) * l[:, 1]) for l in loops)
    step = 20.0
    half = 1.3 * max(axes) + 1500.0
    xs = np.arange(-half, half, step) + 0.5 * step
    X, Y = np.meshgrid(xs + CX, xs + CY, indexing="ij")
    assert area == pytest.approx(body.inside(X, Y).sum() * step ** 2, rel=0.02)
    assert body.inside(X[0], Y[0]).sum() == 0 and body.inside(X[-1], Y[-1]).sum() == 0     # the raster's own edge is outside


def test_the_roughness_is_reduced_where_its_slope_would_fold_the_outline_and_the_reduction_is_reported(monkeypatch):
    """hurst 0 gives the finest octave (range 150 m) the sd of the coarsest, rough x radius = 0.05 x 1.5 km = 75 m: the slope of
    a Gaussian-covariance field has an sd of sqrt(6) sd / range = 1.2 per component, and over thousands of independent cells
    the steepest slope reaches 4-5, which would fold the outline (3-8 islands or holes on bodies of 6 km). One factor scales both
    displacements down to a steepest slope of 0.9: 0.9 / 5 = 0.18 of the nominal roughness here, and the body stays whole."""
    def pieces(body):
        step = 25.0
        xs = np.arange(-6500.0, 6500.0, step) + 0.5 * step
        inside = body.inside(*np.meshgrid(xs + CX, xs + CY, indexing="ij"))
        return ndimage.label(inside)[1] - 1 + ndimage.label(~inside)[1] - 1

    make = lambda seed: sl.salt_body((CX, CY), (3000.0, 1800.0), azimuth=30.0, rough=0.05, hurst=0.0, seed=seed)
    bodies = [make(seed) for seed in range(4)]
    assert all(0.1 <= b.rough_scale <= 0.3 for b in bodies) and sl.salt_body((CX, CY), (R, R)).rough_scale == 1.0
    assert [pieces(b) for b in bodies] == [0, 0, 0, 0]
    monkeypatch.setattr(sl, "_MAX_SLOPE", 9.0)                                                # the guard off
    unguarded = [make(seed) for seed in range(4)]
    assert all(b.rough_scale == 1.0 for b in unguarded) and sum(pieces(b) for b in unguarded) >= 4


def test_a_body_that_has_pinched_out_at_depth_is_absent_there():
    cone = stock(z_ref=2000.0, flare=0.5)                           # R + 0.5 (z - 2000): gone 600 m above (R = 300)
    assert not cone.inside(CX, CY, 1300.0)
    with pytest.raises(ValueError, match="no outline"):
        cone.outline(1300.0)


@pytest.mark.parametrize("kw", [dict(axes=(0.0, 100.0)), dict(axes=(100.0, -5.0)), dict(shape=1.5), dict(lobes=-0.1),
                                dict(lobes=0.4, seed=1)])
def test_a_body_that_cannot_exist_is_refused(kw):
    base = dict(center=(CX, CY), axes=(R, R))
    with pytest.raises(ValueError):
        sl.salt_body(**{**base, **kw})


# --- the upturn and the thinning ---------------------------------------------------------------------------------

def radial_shift(term, d, side=1.0):
    """A term along the +x radius of the circular stock at signed distance d from its contact."""
    return term(CX + side * (R + d), CY)


@pytest.mark.parametrize("dip", [20.0, 45.0, 70.0, 85.0])
def test_the_upturn_meets_the_contact_at_the_drawn_dip(dip):
    term = sl.salt_upturn(stock(), dip=dip, width=400.0)
    h = 0.5
    slope = (radial_shift(term, 2 * h) - radial_shift(term, h)) / h            # dz/dd just outside the contact, m per m
    assert math.degrees(math.atan(slope)) == pytest.approx(dip, abs=1.0)


def test_the_upturn_peaks_at_w_tan_dip_over_p_and_is_zero_beyond_w():
    w, p, dip = 400.0, 2.0, 40.0
    term = sl.salt_upturn(stock(), dip=dip, width=w, power=p)
    peak = w * math.tan(math.radians(dip)) / p
    assert radial_shift(term, 0.0) == pytest.approx(-peak, rel=1e-3)           # negative: lifted toward the surface
    assert radial_shift(term, -80.0) == pytest.approx(-peak, rel=1e-3)         # inside the salt it stays at the peak
    assert radial_shift(term, w) == pytest.approx(0.0, abs=1e-6)
    assert radial_shift(term, w + 500.0) == 0.0
    assert radial_shift(term, 100.0) == pytest.approx(-peak * (1 - 100.0 / w) ** p, rel=1e-3)
    d = np.linspace(0.0, w, 41)
    assert np.all(np.diff([radial_shift(term, x) for x in d]) >= -1e-9)        # it dies away monotonically


def test_the_dip_is_capped_at_85_degrees():
    capped = sl.salt_upturn(stock(), dip=89.0, width=400.0)
    limit = sl.salt_upturn(stock(), dip=85.0, width=400.0)
    assert radial_shift(capped, 0.0) == pytest.approx(radial_shift(limit, 0.0))
    assert radial_shift(limit, 0.0) == pytest.approx(-400.0 * math.tan(math.radians(85.0)) / 2.0, rel=1e-3)
    assert sl.MAX_DIP == 85.0


def test_the_upturn_follows_the_contact_at_the_reference_depth():
    cot = 1.0 / math.tan(math.radians(60.0))
    body = stock(z_ref=2000.0, lean=(cot, 0.0))                                # the contact is 57.7 m further out at 2100
    here, deeper = sl.salt_upturn(body, 30.0, 300.0), sl.salt_upturn(body, 30.0, 300.0, z_ref=2100.0)
    peak = 300.0 * math.tan(math.radians(30.0)) / 2.0                          # 86.6 m
    assert here(CX + R + 1.0, CY) == pytest.approx(-peak * (1.0 - 1.0 / 300.0) ** 2, rel=1e-3)
    assert deeper(CX + R + 1.0, CY) == pytest.approx(-peak, rel=1e-3)          # inside the contact at 2100 m: at the peak
    assert deeper(CX + R + 2.3 + 57.735, CY) == pytest.approx(-peak * (1.0 - 2.3 / 300.0) ** 2, rel=2e-3)


def test_the_upturn_is_a_structure_that_composes_and_can_be_checked_by_the_exporter():
    term = sl.salt_upturn(stock(), 30.0, 300.0)
    assert isinstance(term, st.Structure)
    both = st.dome(10.0, 800.0, center=(CX, CY)) + term
    assert both(CX + R + 50.0, CY) == pytest.approx(st.dome(10.0, 800.0, center=(CX, CY))(CX + R + 50.0, CY)
                                                    + term(CX + R + 50.0, CY))


def test_the_zones_a_body_carries_set_the_widest_cell_it_can_be_gridded_with():
    body = stock()
    assert body.max_cell == math.inf
    sl.salt_upturn(body, 30.0, 300.0)
    sl.salt_thinning(body, 0.4, 150.0)
    assert body.max_cell == 75.0                                               # half the narrowest folding zone


@pytest.mark.parametrize("kw", [dict(dip=-1.0, width=100.0), dict(dip=30.0, width=0.0), dict(dip=30.0, width=100.0, power=0.5)])
def test_an_upturn_that_cannot_exist_is_refused(kw):
    with pytest.raises(ValueError):
        sl.salt_upturn(stock(), **kw)


def test_thinning_is_a_factor_from_1_minus_a_at_the_contact_to_1_at_w():
    a, w, p = 0.4, 400.0, 2.0
    term = sl.salt_thinning(stock(), a, w, power=p)
    assert radial_shift(term, 0.0) == pytest.approx(1.0 - a)
    assert radial_shift(term, -50.0) == pytest.approx(1.0 - a)
    assert radial_shift(term, 100.0) == pytest.approx(1.0 - a * 0.75 ** 2)
    assert radial_shift(term, w) == pytest.approx(1.0) and radial_shift(term, 2 * w) == 1.0
    assert sl.salt_thinning(stock(), 0.0, w)(CX + R + 10.0, CY) == 1.0
    for bad in (-0.1, 1.2):
        with pytest.raises(ValueError):
            sl.salt_thinning(stock(), bad, w)
