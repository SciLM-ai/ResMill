"""Halokinetic sequences beside a salt contact, as the grid sees them: hooks and wedges (Giles & Rowan 2012; Pichel & Jackson,
EarthArXiv 657, Tables 1-3). A sequence is an upturn (the drape fold: width of the folding zone W, taper angle t, so a relief
A = W tan t) and the unconformity that truncates the beds it lifts, which pinches the reservoir out within the zone. Every
number is computed by hand from the formulas of the module, then read back from the grid's ZCORN."""
import math

import numpy as np
import pytest

from resmill import salt as sl
from resmill import structure as st
from resmill.export import _build_geometry
from resmill.faults import Fault
from resmill.layers.base import Layer

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
