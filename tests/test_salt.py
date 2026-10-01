"""Salt bodies: their outline at depth (lean, flare, lobes), the signed distance to the contact, and the upturn and
thinning terms beside them. Every number is computed by hand from the geometry or taken from the structure research
(step5_salt.md: N9, N11-N15, N30)."""
import math

import numpy as np
import pytest

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
