"""Finite faults on the corner-point grid.

A :class:`Fault` is a normal (or reverse) fault of finite extent. Its displacement is largest at its
centre and dies out to zero at an elliptical tip line ``length`` long and ``length / aspect`` high,
following Walsh & Watterson (1987), D = Dmax (1 - r)^1.5 (1 + 3 r)^0.5, with a tip-line aspect of 2.15
(Nicol et al. 1996). The displacement is split between the hanging wall (``hw_share``) and the footwall,
and bends the layers next to the fault (reverse drag) with a taper (1 - d / reach)^2, d measured from the
fault plane at each horizon's own depth, over ``drag`` times the length on each side (Georgsen et al.
2012; Wu et al. 2020). The fault dips at ``dip`` toward its
hanging wall, so its trace moves with depth (a reverse fault's does not: a column, ordered in k, cannot
repeat a section, so each column goes whole to one side), and its trace may curve (``radius``) and wander about its
chord or arc (``bends``: a self-affine profile, Hurst exponent 0.8 as fault surfaces across their slip
(Candela et al. 2012), for the bends left where segments linked; Walsh et al. 2003).

With ``detach`` the fault is listric: its plane is a circular arc, ``dip`` at the tip ellipse's centre and horizontal at the
detachment depth, and the footwall is rigid while the hanging wall moves by vertical shear with constant heave (Gibbs 1983;
White et al. 1986): the horizon of depth z with throw d at the fault has the heave H = trace(z + d) - trace(z) and drops by
plane(h) - plane(h - H) under the column h from the trace. It is d at the fault and zero where the plane is flat, so the
hanging wall rolls over toward the fault, and a planar plane gives the constant throw of a rigid hanging wall. ``hw_share``
and ``drag`` play no part. (The plane is vertical above the arc's vertical point, a depth at which this kinematics carries
no throw: keep it above the model, a dip at ``z_center`` of 70 degrees or less.)

:func:`apply_fault` displaces the interface stack in 3-D and reports which side of the fault each cell
ended on; :func:`face_records` turns that into the stair-stepped cell faces the GRDECL export writes as
``FAULTS``, with the fault's ``mult`` as ``MULTFLT`` (a face multiplier acts on every connection through
the face, so a same-side connection sharing a listed face is sealed too: a little more seal, never a
leak). ``azimuth``-style angles follow the package
convention (degrees clockwise from +x; the trace runs along (cos strike, -sin strike)).
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Fault:
    """One finite fault (lengths in m, angles in degrees; see the module docstring)."""

    center: tuple
    strike: float
    length: float
    throw: float
    dip: float = 60.0
    hanging_wall: int = 1          # +1: the hanging wall lies on the side the strike's normal points to
    hw_share: float = 0.6          # share of the displacement taken by the hanging wall
    drag: tuple = (0.4, 0.2)       # reverse-drag reach on the hanging wall and footwall, x length (0: none)
    radius: float = math.inf       # signed trace curvature radius; inf: a straight trace
    z_center: float | None = None  # depth of the tip ellipse's centre; None: the stack's middle there
    aspect: float = 2.15           # tip-line length / height
    reverse: bool = False          # a reverse fault: the hanging wall moves up
    mult: float = 1.0              # transmissibility multiplier across the fault (MULTFLT)
    name: str = ""
    bends: float = 0.0             # rms wander of the trace about its chord or arc, x length (0: none)
    seed: int | None = None        # draws the bends (required with them)
    kind: str = ""                 # the set a fault pattern drew it from (a label only)
    detach: float | None = None    # depth (m) where the plane turns horizontal (listric fault); None: a planar fault

    def __post_init__(self):
        problems = [msg for bad, msg in (
            (not self.length > 0.0, "length must be positive"),
            (not self.throw > 0.0, "throw must be positive"),
            (not 0.0 < self.dip <= 90.0, "dip must lie in (0, 90]"),
            (self.hanging_wall not in (1, -1), "hanging_wall must be +1 or -1"),
            (not 0.0 <= self.hw_share <= 1.0, "hw_share must lie in [0, 1]"),
            (len(self.drag) != 2 or not min(self.drag) >= 0.0, "drag must be two reaches >= 0"),
            (not abs(self.radius) >= self.length / math.pi, "abs(radius) must be at least length / pi (half a circle)"),
            (not self.aspect > 0.0, "aspect must be positive"),
            (not self.mult >= 0.0, "mult must be >= 0"),
            (not self.bends >= 0.0, "bends must be >= 0"),
            (self.bends > 0.0 and self.seed is None, "bends needs a seed"),
            (self.detach is not None and not 0.0 < self.detach < math.inf, "detach must be a finite depth > 0"),
            (self.detach is not None and self.reverse, "a listric fault (detach) is a normal fault: reverse must be False"),
            (self.detach is not None and self.z_center is not None and not self.detach > self.z_center,
             "detach must lie below z_center"),
        ) if bad]
        if problems:
            raise ValueError(f"Fault {self.name!r}: " + "; ".join(problems))


def ww_profile(r):
    """Walsh & Watterson (1987) displacement over Dmax at the normalized tip-line radius ``r``."""
    r = np.clip(r, 0.0, 1.0)
    return (1.0 - r) ** 1.5 * np.sqrt(1.0 + 3.0 * r)


def _frame(fault, x, y):
    """Distance along the trace ``s`` and signed distance toward the hanging wall ``h`` (m)."""
    az = math.radians(fault.strike)
    t = np.array([math.cos(az), -math.sin(az)])
    n = np.array([math.sin(az), math.cos(az)])
    cx, cy = fault.center
    if math.isinf(fault.radius):
        s = (x - cx) * t[0] + (y - cy) * t[1]
        h = (x - cx) * n[0] + (y - cy) * n[1]
    else:
        r = fault.radius
        ox, oy = cx + r * n[0], cy + r * n[1]                 # the trace's centre of curvature
        wx, wy = x - ox, y - oy
        h = math.copysign(1.0, r) * (abs(r) - np.hypot(wx, wy))
        ux, uy = (cx - ox) / abs(r), (cy - oy) / abs(r)
        s = r * np.arctan2(ux * wy - uy * wx, ux * wx + uy * wy)
    if fault.bends > 0.0:
        h = h - _bend(fault, s)
    return s, fault.hanging_wall * h


def _bend(fault, s):
    """Wander of the trace (m) toward +h at ``s``: 32 wavelengths from twice the length down to a sixteenth
    of it, amplitudes falling as wavenumber^-1.3 (Hurst 0.8), with its best-fit line over the fault removed
    (so ``center`` and ``strike`` keep their meaning) and ``bends`` x length rms there."""
    a = np.random.default_rng(fault.seed).standard_normal((2, 32)) * np.arange(1, 33) ** -1.3

    def profile(t):
        w = np.zeros_like(t)
        for n in range(1, 33):
            w += a[0, n - 1] * np.cos(np.pi * n * t) + a[1, n - 1] * np.sin(np.pi * n * t)
        return w

    u = np.linspace(-0.5, 0.5, 257)
    slope, mean = np.polyfit(u, profile(u), 1)
    scale = fault.bends * fault.length / np.sqrt(np.mean((profile(u) - slope * u - mean) ** 2))
    t = s / fault.length
    return scale * (profile(t) - slope * t - mean)


def _cells(a, nx, ny):
    """Mean over each cell's four corners of a doubled-corner array."""
    return a.reshape(nx, 2, ny, 2, *a.shape[2:]).mean(axis=(1, 3))


def _corners(a):
    """Repeat a per-cell array onto its four corners."""
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


def _listric(dip, zc, detach):
    """The plane of a listric fault: a circular arc of radius R = (detach - zc) / (1 - cos dip), dipping ``dip`` degrees
    where its trace is at depth ``zc`` and horizontal at ``detach``. Returns ``plane(h)``, its depth under a column ``h`` m
    from that trace toward the hanging wall (``detach`` beyond the flat's start, R sin dip from the trace; a plane vertical
    above the arc's vertical point rather than turning back), and ``trace(z)``, the inverse. Both are differences of
    squares, so they stay accurate when the detachment is far enough to make the plane planar."""
    a = math.radians(dip)
    sin_d, cos_d = math.sin(a), math.cos(a)
    radius = (detach - zc) / (2.0 * math.sin(0.5 * a) ** 2)

    def plane(h):
        u = np.clip(np.asarray(h, dtype=float) / radius, sin_d - 1.0, sin_d)          # sin dip - sin(dip there)
        return zc + radius * u * (2.0 * sin_d - u) / (np.sqrt(np.maximum(cos_d ** 2 + u * (2.0 * sin_d - u), 0.0)) + cos_d)

    def trace(z):
        e = np.clip((np.asarray(z, dtype=float) - zc) / radius, -cos_d, 1.0 - cos_d)  # cos(dip there) - cos dip
        return radius * e * (2.0 * cos_d + e) / (np.sqrt(np.maximum(sin_d ** 2 - e * (2.0 * cos_d + e), 0.0)) + sin_d)

    return plane, trace


def _plane(fault, zc):
    """``plane(h)`` and ``trace(z)`` of ``fault``'s plane with its tip ellipse centred at depth ``zc``: the plane's depth
    under a column ``h`` m from the trace there (toward the hanging wall), and the distance of the plane at depth ``z``."""
    if fault.detach is None:
        tan_d = math.tan(math.radians(fault.dip))
        return (lambda h: zc + h * tan_d), (lambda z: (z - zc) / tan_d)
    if not fault.detach > zc:
        raise ValueError(f"Fault {fault.name!r}: detach ({fault.detach:g} m) must lie below the tip ellipse's centre ({zc:g} m)")
    return _listric(fault.dip, zc, fault.detach)


def apply_fault(fault, Xc, Yc, Zc):
    """Displace the interface stack ``Zc`` (2nx, 2ny, nk) by ``fault``.

    Returns the new stack (depths still increasing downward) and ``side`` (nx, ny, nk - 1): +1 for
    cells that ended in the hanging wall, -1 in the footwall, 0 where the fault did not reach.
    """
    nx, ny = Xc.shape[0] // 2, Xc.shape[1] // 2
    Xm, Ym = _cells(Xc, nx, ny), _cells(Yc, nx, ny)
    Zcell = _cells(Zc, nx, ny)
    sin_d = math.sin(math.radians(fault.dip))
    if fault.z_center is None:
        i0 = int(np.clip(np.argmin(np.abs(Xm[:, 0] - fault.center[0])), 0, nx - 1))
        j0 = int(np.clip(np.argmin(np.abs(Ym[0, :] - fault.center[1])), 0, ny - 1))
        zc = float(0.5 * (Zcell[i0, j0, 0] + Zcell[i0, j0, -1]))
    else:
        zc = float(fault.z_center)
    plane, trace = _plane(fault, zc)
    lx = 0.5 * fault.length
    ly = lx / fault.aspect

    def displacement(s, h, z):
        r = np.sqrt((s[..., None] / lx) ** 2 + ((z - zc) / sin_d / ly) ** 2)
        d = fault.throw * ww_profile(r)
        if fault.detach is not None:                    # vertical shear: the horizon's heave from its throw, a rigid footwall
            return plane(h[..., None]) - plane(h[..., None] - (trace(z + d) - trace(z))), 0.0
        hp = np.abs(h[..., None] if fault.reverse else h[..., None] - trace(z))           # from the plane at depth z
        taper = [np.clip(1.0 - hp / (reach * fault.length), 0.0, None) ** 2 if reach > 0.0 else 1.0
                 for reach in fault.drag]
        return fault.hw_share * d * taper[0], (1.0 - fault.hw_share) * d * taper[1]

    sign = -1.0 if fault.reverse else 1.0
    s_k, h_k = _frame(fault, Xc, Yc)
    s_c, h_c = _frame(fault, Xm, Ym)
    dhw_k, dfw_k = displacement(s_k, h_k, Zc)
    dhw_c, dfw_c = displacement(s_c, h_c, Zcell)
    zp = plane(h_c)                                               # the fault plane's depth under each column
    zpk = _corners(zp)[..., None]
    if fault.reverse:                     # a k-ordered column cannot repeat a section: whole columns to one side
        hw = np.broadcast_to((h_c > 0.0)[..., None], Zcell.shape)
        fw = ~hw
        hwk, fwk = _corners(hw), _corners(fw)
    else:                                 # each corner against the column's flat tread, with its own displacement
        hw = Zcell + dhw_c <= zp[..., None]
        fw = Zcell - dfw_c >= zp[..., None]
        hwk, fwk = Zc + dhw_k <= zpk, Zc - dfw_k >= zpk
    Znew = np.where(hwk, Zc + sign * dhw_k, np.where(fwk, Zc - sign * dfw_k, zpk))
    reach = (dhw_c + dfw_c) > 1e-9
    mid = 0.5 * (hw[..., 1:].astype(int) + hw[..., :-1] - fw[..., 1:] - fw[..., :-1].astype(int))
    above = 0.5 * (Zcell[..., 1:] + Zcell[..., :-1]) <= zp[..., None]      # a cell the plane cuts: its middle's side
    mid = np.where(mid == 0, np.where(above, 1.0, -1.0), mid)
    side = np.where(reach[..., 1:] | reach[..., :-1], np.sign(mid), 0).astype(np.int8)
    return np.maximum.accumulate(Znew, axis=2), side


def _face_overlap(a, b):
    """Largest overlap (m) of every pair of cells, k on one side and k' on the other, along a shared face whose pillar
    depths are ``a`` and ``b`` (2, nk). Between the pillars the overlap is concave and piecewise linear, so its maximum
    lies at a pillar or where the two tops or the two bases cross."""
    at, ab, bt, bb = a[:, :-1, None], a[:, 1:, None], b[:, None, :-1], b[:, None, 1:]

    def overlap(t):
        lerp = lambda p: p[0] + t * (p[1] - p[0])
        return np.minimum(lerp(ab), lerp(bb)) - np.maximum(lerp(at), lerp(bt))

    best = np.maximum(overlap(0.0), overlap(1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((at, bt), (ab, bb)):
            cross = np.nan_to_num((q[0] - p[0]) / ((p[1] - p[0]) - (q[1] - q[0])))
            best = np.maximum(best, overlap(np.clip(cross, 0.0, 1.0)))
    return best


def face_records(name, side, zc, act):
    """``FAULTS`` records (1-based I1 I2 J1 J2 K1 K2 and face) of every cell face on which a cell meets a cell of the
    fault's other side, on the final interface stack ``zc`` (2nx, 2ny, nk) with active cells ``act`` (nx, ny, nz).

    Flow multiplies each connection by the multiplier of its lower-index cell's face, so a face is listed wherever
    its cell meets the other side: sideways in any layer (layer k against layer k' across the throw, 'X' and 'Y'),
    and downward where a cell rests on one of the other side, directly or across cut-out cells PINCH bridges ('Z').
    """
    records = []
    for axis, face in ((0, "X"), (1, "Y")):
        lo = np.take(side, np.arange(side.shape[axis] - 1), axis=axis)
        hi = np.take(side, np.arange(1, side.shape[axis]), axis=axis)
        meet = ((lo > 0).any(axis=2) & (hi < 0).any(axis=2)) | ((lo < 0).any(axis=2) & (hi > 0).any(axis=2))
        for i, j in zip(*np.nonzero(meet)):
            if axis == 0:
                a, b = zc[2 * i + 1, 2 * j:2 * j + 2], zc[2 * i + 2, 2 * j:2 * j + 2]
            else:
                a, b = zc[2 * i:2 * i + 2, 2 * j + 1], zc[2 * i:2 * i + 2, 2 * j + 2]
            other = lo[i, j][:, None] * hi[i, j][None, :] == -1
            ks = np.flatnonzero(((_face_overlap(a, b) > 1e-6) & other).any(axis=1))
            for run in np.split(ks, np.flatnonzero(np.diff(ks) > 1) + 1) if ks.size else ():
                records.append((name, i + 1, i + 1, j + 1, j + 1, int(run[0]) + 1, int(run[-1]) + 1, face))
    for i, j in zip(*np.nonzero((side > 0).any(axis=2) & (side < 0).any(axis=2))):
        live = np.flatnonzero(act[i, j])
        s = side[i, j, live]
        for k in live[:-1][s[:-1] * s[1:] == -1]:
            records.append((name, i + 1, i + 1, j + 1, j + 1, int(k) + 1, int(k) + 1, "Z"))
    return records
