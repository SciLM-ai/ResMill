"""The salt-flank fault style: radial faults round a salt contact (the Santos stock, N19-N23), ring faults over a
gently dipping flank (N21), and the style's place in fold_faults."""
import math

import numpy as np
import pytest

from resmill import salt as sl
from resmill import structure as st
from resmill.export import to_grdecl
from resmill.fault_patterns import MIN_THROW, STYLES, _frame, _in_reservoir, fold_faults
from resmill.layers.base import Layer

X_LEN, Y_LEN, DX = 16000.0, 12000.0, 100.0
CENTER = (8000.0, 6000.0)
TOP, THICK = 2000.0, 60.0
Z = TOP + 0.5 * THICK


def fold(**kw):
    return st.closure(area=20e6, height=150.0, aspect=1.4, azimuth=20.0, center=CENTER, **{"warp": 0.1, "seed": 2, **kw})


def crest_of(f):
    return CENTER[0] + f.crest_offset[0], CENTER[1] + f.crest_offset[1]


def stock(f, radius=1200.0, offset=(0.0, 0.0), axes=None, **kw):
    """A salt body with its centre ``offset`` m from the fold's crest (0: the crest is inside the salt)."""
    cx, cy = crest_of(f)
    return sl.salt_body((cx + offset[0], cy + offset[1]), axes or (radius, radius), z_ref=Z, **kw)


def flank(f, distance=1000.0, radius=3000.0, **kw):
    """A flank whose contact passes ``distance`` m south of the crest, so the crest is cut off into the salt and the trap
    lies on the far (south) side."""
    cx, cy = crest_of(f)
    return sl.salt_body((cx, cy + radius - distance), (radius, radius), z_ref=Z, **kw)


def faults_of(f, body, seed, density=2.0, **kw):
    return fold_faults("salt_flank", f, X_LEN, Y_LEN, DX, density, TOP, THICK, seed=seed, salt=body, **kw)


def pooled(f, body, kinds=("radial",), seeds=range(40), density=2.0, **kw):
    return [g for s in seeds for g in faults_of(f, body, s, density, **kw) if g.kind in kinds]


def unit(strike):
    s = math.radians(strike)
    return np.array([math.cos(s), -math.sin(s)]), np.array([math.sin(s), math.cos(s)])     # trace, left normal


def outward(body, x, y, h=2.0):
    """The unit normal of the contact at (x, y), pointing out of the salt (from the distance's gradient)."""
    g = np.array([body.distance(x + h, y) - body.distance(x - h, y), body.distance(x, y + h) - body.distance(x, y - h)])
    return g / np.hypot(*g)


def off_normal(body, g):
    """Angle (degrees, 0-90) between a fault's trace and the contact's normal at its centre."""
    t, _ = unit(g.strike)
    n = outward(body, *g.center)
    return math.degrees(math.acos(min(abs(float(t @ n)), 1.0)))


def test_the_style_needs_a_salt_body_and_only_it_takes_one():
    f = fold()
    assert "salt_flank" in STYLES
    with pytest.raises(ValueError, match="salt"):
        fold_faults("salt_flank", f, X_LEN, Y_LEN, DX, 1.0, TOP, THICK, seed=1)
    with pytest.raises(ValueError, match="salt"):
        fold_faults("four_way", f, X_LEN, Y_LEN, DX, 1.0, TOP, THICK, seed=1, salt=flank(f))


def test_radial_faults_have_the_santos_lengths_dips_aspect_and_throws():
    """N19 (Coleman et al. 2018, 42 faults round one stock): length P10/P50/P90 0.6 / 1.1 / 2.0 km (0.4-3.7 km), planar,
    dips 50-60 degrees, throws under 80 m, tip-line aspect (length over height) 1.9."""
    f = fold()
    radial = pooled(f, stock(f), density=4.0)
    assert len(radial) > 150
    length = np.array([g.length for g in radial])
    assert length.min() >= 400.0 and length.max() <= 3700.0
    p10, p50, p90 = np.percentile(length, [10, 50, 90])
    assert 500.0 <= p10 <= 800.0 and 950.0 <= p50 <= 1400.0 and 1700.0 <= p90 <= 2700.0      # drawn 0.6 / 1.1 / 2.0, a little
    assert all(50.0 <= g.dip <= 60.0 for g in radial)                                          # longer: 5 m is the least counted
    assert all(g.throw <= 80.0 + 1e-9 for g in radial) and all(g.aspect == 1.9 for g in radial)
    top = lambda g: TOP + float(f(np.array([g.center[0]]), np.array([g.center[1]]))[0])
    assert all(g.throw * _in_reservoir(g.z_center, top(g), THICK, 0.5 * g.length / 1.9 * math.sin(math.radians(g.dip)))
               >= MIN_THROW - 1e-9 for g in radial)


def test_radial_faults_are_centred_on_the_contact_half_in_the_salt_and_strike_along_its_normal():
    f = fold()
    body = stock(f, radius=1500.0)
    radial = pooled(f, body, density=3.0)
    assert len(radial) > 100
    assert max(abs(float(body.distance(*g.center))) for g in radial) < 2.0                    # the centre is on the contact
    assert np.mean([off_normal(body, g) <= 30.0 for g in radial]) >= 0.9                      # 10 degrees scatter: 3 sigma
    half = [g for g in radial if g.length < 2000.0]
    ends = [(np.array(g.center) + s * 0.5 * g.length * unit(g.strike)[0]) for g in half for s in (-1.0, 1.0)]
    in_salt = np.array([body.inside(*e) for e in ends]).reshape(-1, 2)
    assert np.mean(in_salt.sum(axis=1) == 1) >= 0.85                                          # one end in the salt, one out


def clusters_of(body, radial, gap=30.0):
    """The groups of radial faults round a body, split where the angle between neighbours exceeds ``gap`` degrees."""
    cx, cy = body.center
    ang = np.sort(np.degrees(np.arctan2([g.center[1] - cy for g in radial], [g.center[0] - cx for g in radial])))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 360.0]]))
    return int((gaps > gap).sum()) or 1


def test_radial_faults_come_in_three_to_six_groups_round_a_stock():
    """N20 (Davison et al. 2000): in map view they form 3-6 main groups."""
    f = st.closure(area=40e6, height=150.0, aspect=1.2, azimuth=20.0, center=CENTER, seed=2)
    body = stock(f, radius=1800.0)
    counts = []
    for s in range(30):
        radial = [g for g in faults_of(f, body, s, density=1.5) if g.kind == "radial"]
        if len(radial) >= 6:
            counts.append(clusters_of(body, radial))
    assert len(counts) >= 20
    assert min(counts) >= 2 and max(counts) <= 6
    assert 3.0 <= np.mean(counts) <= 5.5


def test_radial_faults_gather_at_the_ends_of_an_elongate_body():
    """N20: they concentrate at the elongate ends of elliptical diapirs."""
    f = st.closure(area=40e6, height=150.0, aspect=1.2, azimuth=20.0, center=CENTER, seed=2)
    a, b = 2800.0, 900.0
    body = stock(f, axes=(a, b), azimuth=0.0, shape=2.0)
    radial = pooled(f, body, seeds=range(30), density=1.5)
    cx, cy = body.center
    end = np.abs([g.center[0] - cx for g in radial]) / a > 0.6
    poly = body.outline()
    seg = np.hypot(*(np.roll(poly, -1, axis=0) - poly).T)
    uniform = float(seg[np.abs(poly[:, 0] - cx) / a > 0.6].sum() / seg.sum())                  # the contact's own share there
    assert len(radial) > 100 and end.mean() > 1.25 * uniform


def test_ring_faults_only_over_a_flank_dipping_60_degrees_or_less():
    """N21 (Rowan et al. 1999): major faults are tangential where the flank dips up to about 60 degrees; none off
    vertical plugs. The flank leans outward by cot(dip) m per metre of depth."""
    f = fold()
    assert not pooled(f, stock(f, radius=1500.0), kinds=("ring",), seeds=range(40))             # a vertical plug
    assert not pooled(f, stock(f, radius=1500.0, lean=(0.0, 0.0), flare=-0.2), kinds=("ring",))  # an overhang
    for dip, expected in ((45.0, True), (75.0, False)):
        body = stock(f, radius=1500.0, flare=1.0 / math.tan(math.radians(dip)))                 # wider with depth: dips outward
        ring = pooled(f, body, kinds=("ring",), seeds=range(40))
        assert bool(ring) == expected
    cx, cy = body.center
    body = stock(f, radius=1500.0, flare=1.0)
    ring = pooled(f, body, kinds=("ring",), seeds=range(40))
    assert ring and len(ring) <= 2 * 40
    for g in ring:
        trace, normal = unit(g.strike)
        curv = np.array(g.center) + g.radius * normal                                            # the trace's centre of curvature
        assert np.hypot(curv[0] - cx, curv[1] - cy) < 1.0                                        # concentric: concave toward the salt
        assert abs(float(trace @ outward(body, *g.center))) < 0.2                                # tangential to the contact
        assert g.length <= math.pi * abs(g.radius) + 1e-6
    assert {g.hanging_wall * np.sign(g.radius) for g in ring} == {1.0, -1.0}                      # toward the salt and away


def test_the_density_counts_the_visible_trap_and_the_salt_faults_are_part_of_it():
    """S6: faults with 5 m in the reservoir per km2 of the trap, which the salt has cut. The regional faults take their
    share R / (1 + R) of that count over the whole model (R = 1: half); the rest is the trap's own, and the radial and
    ring faults are part of it, as the fold's own sets are (the radial a 0.4-0.8 share)."""
    f = fold()
    body = flank(f, distance=900.0)
    fr = _frame(f, X_LEN, Y_LEN, DX)
    visible = fr["mask"] & ~body.inside(fr["X"], fr["Y"], TOP + fr["depth"] + 0.5 * THICK)
    whole, area = (m.sum() * DX * DX / 1e6 for m in (fr["mask"], visible))
    assert 0.3 * whole < area < 0.9 * whole                                 # the salt took a good part of the closure
    for density in (0.5, 2.0, 4.0):
        own, radial = [], []
        for s in range(12):
            fs = faults_of(f, body, s, density, regional=(1.0, 40.0))
            own.append(sum(g.kind in ("radial", "ring", "longitudinal", "oblique", "transverse", "inherited") for g in fs))
            radial.append(sum(g.kind == "radial" for g in fs))
        budget = round(density * area) - round(0.5 * density * area)       # the trap's count less the regional share
        assert abs(np.mean(own) - budget) <= 1.0 + 0.05 * budget
        assert 0.4 * budget - 1.0 <= np.mean(radial) <= 0.8 * budget + 1.0   # more than the ring faults and the rest add


def test_no_fault_is_drawn_for_salt_far_from_the_trap_and_the_style_still_draws_the_rest():
    f = fold()
    cx, cy = crest_of(f)
    far = sl.salt_body((cx, cy + 9000.0), (500.0, 500.0), z_ref=Z)
    fs = faults_of(f, far, 3)
    assert fs and not [g for g in fs if g.kind in ("radial", "ring")]


def test_the_style_is_reproducible_and_its_faults_are_valid_for_the_export(tmp_path):
    f = fold()
    body = flank(f, lobes=0.1, seed=4, flare=-0.1)
    sl.salt_upturn(body, 40.0, 400.0)
    a, b = faults_of(f, body, 5), faults_of(f, body, 5)
    assert a and a == b
    assert len({g.name for g in a}) == len(a) and all(len(g.name) <= 8 for g in a)
    nx, ny, nz = 160, 120, 4
    L = Layer(nx, ny, nz, X_LEN, Y_LEN, THICK, top_depth=TOP, kzkx=0.1)
    L.poro_mat, L.perm_mat = np.full((nx, ny, nz), 0.2), np.full((nx, ny, nz), 100.0)
    text = to_grdecl(L, tmp_path / "m.grdecl", structure=f + sl.salt_upturn(body, 40.0, 400.0), faults=a, salt=body).read_text()
    assert "\nFAULTS\n" in text and "\nACTNUM\n" in text
