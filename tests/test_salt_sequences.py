"""Halokinetic sequences beside a salt contact, as the grid sees them: hooks and wedges (Giles & Rowan 2012; Pichel & Jackson,
EarthArXiv 657, Tables 1-3). A sequence is an upturn (the drape fold: width of the folding zone W, taper angle t, so a relief
A = W tan t) and the unconformity that truncates the beds it lifts, which pinches the reservoir out within the zone. Every
number is computed by hand from the formulas of the module, then read back from the grid's ZCORN."""
import math

import numpy as np
import pytest

from resmill import salt as sl
from resmill.export import _build_geometry, to_grdecl
from resmill.faults import Fault
from resmill.layers.base import Layer
from tests.test_export import read_grdecl, zcorn_cube

TOP, P = 2000.0, 2.0
CONTACT = 100.0                                                  # m: the contact of every wall below, salt to its south


def build(width, taper, cut, nz, dz, dx, **kw):
    """A reservoir of nz layers of dz m (flat, from TOP) beside a straight wall whose contact is y = CONTACT: pillars every dx
    m, the folding zone and 300 m beyond it to the north. Returns the geometry's depths, activity, the distance of each pillar
    row from the contact (negative in the salt) and the sequence."""
    nx, ny = 8, int(round((CONTACT + width + 300.0) / dx))
    L = Layer(nx, ny, nz, nx * dx, ny * dx, nz * dz, top_depth=TOP, kzkx=0.1)
    L.poro_mat, L.perm_mat = np.full((nx, ny, nz), 0.2), np.full((nx, ny, nz), 100.0)
    body = sl.salt_body((0.5 * nx * dx, CONTACT - 5000.0), (20000.0, 5000.0))
    seq = sl.salt_sequence(body, width, taper, cut, datum=TOP, **kw)
    zc, act = _build_geometry([L], structure=seq.upturn, erode_above=seq.truncation, salt=body)[2:]
    rows = np.arange(ny + 1) * dx - CONTACT                      # pillar rows 0..ny
    return dict(zc=zc, act=act, d=rows, seq=seq, L=L, body=body, nx=nx, ny=ny, nz=nz, dx=dx)


def node(m, j, k):
    """Depth (m) of interface k at pillar row j (column 0): the doubled corner arrays hold a pillar's value at 2j-1 and 2j."""
    return m["zc"][0, min(2 * j, 2 * m["ny"] - 1), k]


@pytest.mark.parametrize("width,taper,dx", [(100.0, 62.0, 5.0), (690.0, 29.0, 10.0)])
def test_the_upturn_has_the_drawn_width_and_relief_on_the_grid(width, taper, dx):
    """The taper angle is that of the line from the fold's inflection point to its tip (Pichel & Jackson), so the relief is
    W tan(taper); with no truncation the grid's base lifts as A (1 - d / W)^2 at every pillar and not at all beyond W."""
    m = build(width, taper, 0.0, 6, 5.0, dx)
    relief = width * math.tan(math.radians(taper))
    assert m["seq"].relief == pytest.approx(relief)
    far = TOP + 6 * 5.0
    rows = [j for j in range(1, m["ny"]) if m["d"][j] > 0.0]
    lift = np.array([far - node(m, j, 6) for j in rows])
    d = m["d"][rows]
    assert np.abs(lift - relief * np.clip(1.0 - d / width, 0.0, 1.0) ** 2).max() < 0.5      # the polyline's own error
    assert lift[d > width + 1e-9].max() < 1e-6 and lift[d < 0.9 * width].min() > 0.0         # exactly W wide


@pytest.mark.parametrize("width,taper,cut,dx", [(100.0, 62.0, 1.0, 5.0), (690.0, 29.0, 0.5, 10.0)])
def test_the_reservoir_pinches_out_where_the_unconformity_cuts_the_lift_down_to_it(width, taper, cut, dx):
    """The unconformity cuts ``cut`` of the lift A (1 - d / W)^2 off the beds, so a reservoir h0 thick is gone where
    cut A (1 - d / W)^2 >= h0, at d_p = W (1 - sqrt(h0 / (cut A))): hand-computed 43.5 m for the hook (Pichel & Jackson: up to
    200 m) and 303.5 m for the wedge (300-1000 m). The first active column's outer pillar lies in (d_p, d_p + dx]."""
    nz, dz = 12, 5.0
    h0, relief = nz * dz, width * math.tan(math.radians(taper))
    d_p = width * (1.0 - math.sqrt(h0 / (cut * relief)))
    assert d_p == pytest.approx(43.5 if width == 100.0 else 303.5, abs=0.1)
    m = build(width, taper, cut, nz, dz, dx)
    live = [j for j in range(m["ny"]) if m["act"][0, j].any() and 0.5 * (m["d"][j] + m["d"][j + 1]) > 0.0]
    first = live[0]
    assert d_p < m["d"][first + 1] <= d_p + dx + 1e-9
    assert not m["act"][0, :first][m["d"][:first] >= 0.0].any()                  # nothing between the salt and the pinch-out
    assert m["act"][0, first:, :].any(axis=1).all()                              # and the reservoir is there beyond it


@pytest.mark.parametrize("width,taper,cut,low,high", [(100.0, 62.0, 1.0, 70.0, 90.0), (690.0, 29.0, 0.5, 5.0, 30.0)])
def test_the_truncation_angle_is_high_for_a_hook_and_low_for_a_wedge_on_the_grid(width, taper, cut, low, high):
    """Giles & Rowan: beds are truncated beneath the bounding unconformity at over 70 degrees in a hook and under 30 in a
    wedge. On a reservoir thick enough to survive to the contact, the grid's base interface and its cut top, read at the first
    pillars, differ in dip by atan(p tan t) - atan((1 - cut) p tan t): 75.1 degrees (hook), 19.0 (wedge)."""
    dx = 2.5 if width == 100.0 else 5.0
    m = build(width, taper, cut, 60, 10.0, dx)
    j = int(round(CONTACT / dx))                                                # the contact's pillar row
    slope = lambda k: (node(m, j + 1, k) - node(m, j, k)) / dx                  # dip away from the contact (> 0)
    bed, top = math.degrees(math.atan(slope(60))), math.degrees(math.atan(slope(0)))
    expected = math.degrees(math.atan(P * math.tan(math.radians(taper)))) - math.degrees(
        math.atan((1.0 - cut) * P * math.tan(math.radians(taper))))
    assert (bed - top) == pytest.approx(expected, abs=0.5)
    assert low < bed - top < high


def test_the_truncation_cuts_nothing_beyond_the_folding_zone_where_faults_throw_the_beds_up():
    """Faults far from the salt throw a footwall above the datum; the unconformity must not erode it (it is far above the
    model there), so the same model with and without the truncation is identical beyond the zone."""
    width, dx = 100.0, 10.0
    m = build(width, 62.0, 1.0, 6, 5.0, dx)
    fault = Fault(center=(40.0, CONTACT + width + 150.0), strike=0.0, length=2000.0, throw=15.0, dip=60.0, name="F1")
    L, body = m["L"], m["body"]
    up = m["seq"].upturn
    with_cut = _build_geometry([L], structure=up, erode_above=m["seq"].truncation, faults=[fault], salt=body)
    without = _build_geometry([L], structure=up, faults=[fault], salt=body)
    far = slice(2 * int((CONTACT + width + 20.0) / dx), None)
    assert np.array_equal(with_cut[2][:, far], without[2][:, far]) and np.array_equal(with_cut[3][:, int((CONTACT + width + 20.0) / dx):],
                                                                                         without[3][:, int((CONTACT + width + 20.0) / dx):])
    assert (with_cut[3] != without[3]).any() or (with_cut[2] != without[2]).any()      # while inside the zone it does cut


def test_a_grid_that_cannot_resolve_a_hook_is_refused():
    """A hook is 50-200 m wide, so cells may be at most half of it (an upturn of 100 m needs cells of 50 m or less)."""
    nx, ny, nz = 8, 20, 4
    body = sl.salt_body((0.5 * nx * 100.0, CONTACT - 5000.0), (20000.0, 5000.0))
    seq = sl.salt_sequence(body, 100.0, 62.0, 1.0, datum=TOP)
    assert body.max_cell == 50.0
    for dx, ok in ((100.0, False), (50.0, True)):
        L = Layer(nx, ny, nz, nx * dx, ny * dx, 20.0, top_depth=TOP, kzkx=0.1)
        L.poro_mat, L.perm_mat = np.full((nx, ny, nz), 0.2), np.full((nx, ny, nz), 100.0)
        if ok:
            _build_geometry([L], structure=seq.upturn, erode_above=seq.truncation, salt=body)
        else:
            with pytest.raises(ValueError, match="wider than half"):
                _build_geometry([L], structure=seq.upturn, erode_above=seq.truncation, salt=body)


@pytest.mark.parametrize("kw", [dict(width=100.0, taper=-1.0, cut=0.5), dict(width=0.0, taper=30.0, cut=0.5),
                                dict(width=100.0, taper=30.0, cut=1.5), dict(width=100.0, taper=30.0, cut=-0.1)])
def test_a_sequence_that_cannot_exist_is_refused(kw):
    body = sl.salt_body((0.0, 0.0), (500.0, 500.0))
    with pytest.raises(ValueError):
        sl.salt_sequence(body, datum=TOP, **kw)


def test_the_cells_a_truncation_removes_are_inactive_but_not_salt_in_the_labels():
    """salt_labels marks salt only: the cells the unconformity cuts away (a hook pinches the reservoir out 43.5 m from the contact)
    are written inactive too, and are not part of the salt mask, its volume fraction or its thickness."""
    m = build(100.0, 62.0, 1.0, 12, 5.0, 5.0)
    body, seq, L = m["body"], m["seq"], m["L"]
    labels = sl.salt_labels(L, salt=body, structure=seq.upturn, erode_above=seq.truncation)
    xc, yc, zc, act = _build_geometry([L], structure=seq.upturn, erode_above=seq.truncation, salt=body)
    salt = sl.salt_cells(body, xc, yc, zc)
    inactive = act == 0
    assert (labels["mask"] == salt).all() and labels["mask"].sum() == salt.sum()
    assert inactive.sum() > salt.sum()                                    # the truncated columns are dead as well
    assert labels["volume_fraction"] == pytest.approx(salt.mean())
    cut = inactive & ~salt
    assert cut.any() and not (labels["mask"][cut]).any()


# --- the two forms of a hook (the owner's ruling of 2026-10-02) -----------------------------------------------------------

def written(tmp_path, width, taper, cut, nz, dz, dx):
    """The model of :func:`build` written by ``to_grdecl`` and read back: its ZCORN cube (2nx, 2ny, 2nz), ACTNUM and sequence."""
    m = build(width, taper, cut, nz, dz, dx)
    path = to_grdecl(m["L"], tmp_path / "hook.grdecl", structure=m["seq"].upturn, erode_above=m["seq"].truncation, salt=m["body"])
    blocks = read_grdecl(path)
    return {**m, "z": zcorn_cube(blocks, m["nx"], m["ny"], nz),
            "act": np.asarray(blocks["ACTNUM"]).reshape((m["nx"], m["ny"], nz), order="F")}


def thickness(w):
    """The reservoir's thickness (m) at each pillar row 0..ny-1 of the written grid: base of the last layer less top of the first."""
    return w["z"][0, 0::2, -1][:w["ny"]] - w["z"][0, 0::2, 0][:w["ny"]]


def contact_angle(w):
    """The angle (degrees) between the beds and the unconformity at the contact, from the written ZCORN of a reservoir thick enough
    to survive there: the dips of its base (the beds) and of its top (the surface that cuts them), each extrapolated to the contact
    from the slopes over the first two pillar intervals (d = dx / 2 and 3 dx / 2) as 1.5 s1 - 0.5 s2."""
    j, dx = int(round(CONTACT / w["dx"])), w["dx"]
    dips = []
    for k in (-1, 0):
        d = w["z"][0, 0::2, k][:w["ny"]]
        s1, s2 = (d[j + 1] - d[j]) / dx, (d[j + 2] - d[j + 1]) / dx
        dips.append(math.degrees(math.atan(1.5 * s1 - 0.5 * s2)))
    return dips[0] - dips[1]


def lift_at(d, width, taper):
    """The upturn's lift (m) at distance d from the contact: A (1 - d / W)^2, A = W tan(taper), as drawn."""
    return width * math.tan(math.radians(taper)) * np.clip(1.0 - d / width, 0.0, 1.0) ** 2


def test_a_hook_that_pinches_the_reservoir_out_is_cut_at_over_70_degrees_on_the_written_grid(tmp_path):
    """The pinch-out form meets Giles & Rowan's definition (a truncation angle over 70 degrees) by construction. A hook of
    W = 100 m and taper 62 degrees has beds that meet the contact at atan(2 tan 62) = 75.11 degrees; cut at 72 degrees takes
    cut = 1 - tan(3.11) / tan(75.11) = 0.9856 of the 188.1 m of lift, so a reservoir of 60 m is gone where
    0.9856 x 188.1 (1 - d / 100)^2 >= 60, from d_p = 100 (1 - sqrt(60 / 185.4)) = 43.1 m to the contact."""
    width, taper, angle, h0, dx = 100.0, 62.0, 72.0, 60.0, 2.5
    cut = sl.truncation_cut(taper, angle)
    assert cut == pytest.approx(1.0 - math.tan(math.radians(75.1124 - angle)) / math.tan(math.radians(75.1124)), abs=1e-4)
    assert cut == pytest.approx(0.9856, abs=1e-4)
    thin = written(tmp_path, width, taper, cut, 12, 5.0, dx)
    assert thin["seq"].angle == pytest.approx(angle) and thin["seq"].dip == pytest.approx(75.1124, abs=1e-3)
    d = np.arange(thin["ny"]) * dx - CONTACT                                      # distance of each pillar row from the contact
    outside = d > 0.0
    assert thickness(thin)[outside] == pytest.approx(np.clip(h0 - cut * lift_at(d[outside], width, taper), 0.0, None), abs=0.015)
    d_p = width * (1.0 - math.sqrt(h0 / (cut * width * math.tan(math.radians(taper)))))
    assert d_p == pytest.approx(43.1, abs=0.05)
    live = [j for j in range(thin["ny"] - 1) if thin["act"][0, j].any() and d[j] >= 0.0]
    assert d_p < d[live[0] + 1] <= d_p + dx + 1e-9                                 # the first live column starts where it is thick enough
    assert not thin["act"][0, :live[0]][d[:live[0]] >= 0.0].any()                  # and nothing lives between it and the salt
    thick = written(tmp_path, width, taper, cut, 60, 10.0, dx)
    assert contact_angle(thick) == pytest.approx(angle, abs=0.5) and contact_angle(thick) > 70.0


@pytest.mark.parametrize("taper,angle", [(60.0, 71.0), (70.0, 78.0), (80.0, 84.0)])
def test_the_pinch_out_form_is_over_70_degrees_for_any_taper_that_can_reach_it(tmp_path, taper, angle):
    """Another hook, W = 150 m (lift 260, 412 and 850 m) on 1,500 m of reservoir that survives to the contact: the measured
    angle is the one asked for, whatever the taper."""
    width, dx = 150.0, 2.5
    thick = written(tmp_path, width, taper, sl.truncation_cut(taper, angle), 100, 15.0, dx)
    assert contact_angle(thick) == pytest.approx(angle, abs=0.6) and contact_angle(thick) > 70.0


def test_a_hook_that_keeps_the_interval_runs_it_up_to_the_wall_at_a_low_angle_on_the_written_grid(tmp_path):
    """The keep form: a cut of 0.2 of the 188.1 m lift removes 37.6 m at the contact and 0.2 x 188.1 (1 - d / 100)^2 beyond it, so
    a reservoir of 60 m keeps 22.4 m at the wall and every column is live up to it; the beds are cut at
    75.11 - atan(0.8 tan 75.11) = 3.49 degrees, far under 70."""
    width, taper, cut, h0, dx = 100.0, 62.0, 0.2, 60.0, 2.5
    thin = written(tmp_path, width, taper, cut, 12, 5.0, dx)
    assert thin["seq"].angle == pytest.approx(3.49, abs=0.01)
    d = np.arange(thin["ny"]) * dx - CONTACT
    outside = d >= 0.0
    t = thickness(thin)
    assert t[outside] == pytest.approx(h0 - cut * lift_at(d[outside], width, taper), abs=0.015)
    assert t[outside].min() == pytest.approx(h0 - cut * width * math.tan(math.radians(taper)), abs=0.02)       # 22.4 m at the wall
    assert t[outside].min() > 22.0
    first_cell = int(round(CONTACT / dx))                                          # the cell whose inner face is the contact
    assert thin["act"][0, first_cell:].any(axis=1).all() and not thin["act"][0, first_cell - 1].any()
    thick = written(tmp_path, width, taper, cut, 60, 10.0, dx)
    assert contact_angle(thick) == pytest.approx(3.49, abs=0.5) and contact_angle(thick) < 70.0


def test_the_cut_of_an_angle_is_the_inverse_of_the_angle_of_a_cut():
    """cut = 1 - tan(dip - angle) / tan(dip): at taper 62 (dip 75.11) an angle of 70 degrees takes a cut of 0.9762, the dip itself
    (a flat unconformity) a cut of 1 and 0 degrees a cut of 0; at taper 86 the dip is capped at 85 and 80 degrees take
    1 - tan(5) / tan(85) = 0.9923. The least taper for 70 degrees is atan(tan(70) / 2) = 53.95; below it the angle is refused."""
    assert sl.truncation_cut(62.0, 70.0) == pytest.approx(0.9762, abs=1e-4)
    assert sl.truncation_cut(62.0, math.degrees(math.atan(2.0 * math.tan(math.radians(62.0))))) == pytest.approx(1.0, abs=1e-9)
    assert sl.truncation_cut(62.0, 0.0) == 0.0
    assert sl.truncation_cut(86.0, 80.0) == pytest.approx(0.9923, abs=1e-4)
    assert sl.truncation_cut(54.0, 70.0) == pytest.approx(0.99978, abs=1e-5)
    body = sl.salt_body((0.0, 0.0), (500.0, 500.0))
    for taper, angle in ((62.0, 72.0), (40.0, 30.0), (86.0, 80.0), (30.0, 5.0)):
        seq = sl.salt_sequence(body, 100.0, taper, sl.truncation_cut(taper, angle), datum=TOP)
        assert seq.angle == pytest.approx(angle, abs=1e-9)
    for taper, angle in ((53.9, 70.0), (50.0, 70.0), (62.0, 76.0), (62.0, -1.0)):
        with pytest.raises(ValueError, match="cannot be cut"):
            sl.truncation_cut(taper, angle)


def test_the_sequence_reports_the_dip_and_relief_the_grid_has_when_the_dip_is_capped(tmp_path):
    """A taper of 86 degrees would give a dip of 88 and a relief of W tan(88) / 2, but the grid caps the dip at 85: the
    relief is 100 tan(85) / 2 = 571.5 m (not 1,430 m), the dip 85, and the written base is lifted by that much at the contact."""
    w = written(tmp_path, 100.0, 86.0, 0.0, 12, 5.0, 5.0)
    assert w["seq"].dip == 85.0 and w["seq"].relief == pytest.approx(571.5, abs=0.1)
    j = int(round(CONTACT / w["dx"]))
    base = w["z"][0, 0::2, -1]
    assert base[w["ny"] - 1] - base[j] == pytest.approx(571.5, abs=0.5)       # the contact is placed to a few cm on a grid of 62 m
    plain = sl.salt_sequence(w["body"], 100.0, 40.0, 0.5, datum=TOP)
    assert plain.dip == pytest.approx(math.degrees(math.atan(2.0 * math.tan(math.radians(40.0)))))
    assert plain.relief == pytest.approx(100.0 * math.tan(math.radians(40.0)), rel=1e-9)
