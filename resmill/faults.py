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

With ``flatten`` the fault is listric: its plane is straight at ``dip`` down to the base of its ramp (``ramp_base``, default the
tip ellipse's centre), as the faults of the Gulf's seismic sections are down to their bends, and below it tan(dip) falls by 1/e
every ``flatten`` m of depth, so the plane flattens into a long gentle tail. ``flatten`` is 2.4 km at the median of six published
faults (Xiao & Suppe 1992; Ewing et al. 1986, whose bends lie 1.0-2.7 km down) and 2.2 km if Bruce's (1973) fall from 60 to 15
degrees takes about 4 km. The footwall is rigid while the hanging wall moves by vertical shear with constant heave (Gibbs 1983;
White et al. 1986): one heave H for the whole block, so that a column of it slides down the plane as a unit and no zone changes
thickness. ``throw`` is the throw, at the fault's centre line, of the horizon at the tip ellipse's centre (``z_center``), dying out
along the strike like the tip line's profile, and gives the heave H = trace(z_center + throw) - trace(z_center); the column h from
the trace then drops by plane(h) - plane(h - H), whatever its depth. The throw of a horizon is H tan(dip) where it cuts the plane:
``throw`` at ``z_center``, more above it where the plane is steeper, less as the plane flattens, so that the hanging wall rolls
over toward the fault above the bend, from the cutoff itself when the cutoff lies below it. The tip ellipse's depth plays no part
(the block moves at every depth), and a plane that is planar as ``flatten`` grows gives the constant throw of a rigid hanging
wall, that of a planar fault with a tall tip ellipse. ``hw_share`` and ``drag`` play no part.

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

TIP_ASPECT = 2.15                   # tip-line length / height of a fault (Nicol et al. 1996)


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
    aspect: float = TIP_ASPECT     # tip-line length / height
    reverse: bool = False          # a reverse fault: the hanging wall moves up
    mult: float = 1.0              # transmissibility multiplier across the fault (MULTFLT)
    name: str = ""
    bends: float = 0.0             # rms wander of the trace about its chord or arc, x length (0: none)
    seed: int | None = None        # draws the bends (required with them)
    kind: str = ""                 # the set a fault pattern drew it from (a label only)
    flatten: float | None = None   # listric: tan(dip) falls by 1/e per this much depth (m) below the ramp; None: a planar fault
    ramp_base: float | None = None  # listric: depth (m) where the straight ramp ends and the flattening starts; None: z_center

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
            (self.flatten is not None and not 0.0 < self.flatten < math.inf, "flatten must be a finite length > 0"),
            (self.flatten is not None and self.reverse, "a listric fault (flatten) is a normal fault: reverse must be False"),
            (self.ramp_base is not None and (self.flatten is None or not math.isfinite(self.ramp_base)),
             "ramp_base is a finite depth and needs flatten (a listric fault)"),
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


def _listric(dip, zc, flatten, zb=None):
    """The plane of a listric fault: straight at ``dip`` degrees down to the bend at depth ``zb`` (default ``zc``), and below it
    tan(dip) falls as exp(-(z - zb) / flatten): z = zb + flatten ln(1 + h tan(dip) / flatten) under a column ``h`` m from the
    trace at ``zb``, a curve that leaves the ramp without a kink and is the ramp itself as ``flatten`` grows. Returns
    ``plane(h)``, its depth under a column ``h`` m from the trace at depth ``zc`` toward the hanging wall, and ``trace(z)``,
    the inverse (the exponent is capped at 700, where it would overflow: a depth no model reaches)."""
    tan_d = math.tan(math.radians(dip))
    zb = zc if zb is None else zb

    def plane0(h):                                                       # from the bend's trace
        h = np.asarray(h, dtype=float)
        return zb + np.minimum(h, 0.0) * tan_d + flatten * np.log1p(np.maximum(h, 0.0) * tan_d / flatten)

    def trace0(z):
        z = np.asarray(z, dtype=float) - zb
        return np.minimum(z, 0.0) / tan_d + flatten / tan_d * np.expm1(np.minimum(np.maximum(z, 0.0) / flatten, 700.0))

    shift = float(trace0(zc))                                            # the trace at zc, from the bend's: 0 when the bend is at zc
    return (lambda h: plane0(np.asarray(h, dtype=float) + shift)), (lambda z: trace0(z) - shift)


def _plane(fault, zc):
    """``plane(h)`` and ``trace(z)`` of ``fault``'s plane with its tip ellipse centred at depth ``zc``: the plane's depth
    under a column ``h`` m from the trace there (toward the hanging wall), and the distance of the plane at depth ``z``."""
    if fault.flatten is None:
        tan_d = math.tan(math.radians(fault.dip))
        return (lambda h: zc + h * tan_d), (lambda z: (z - zc) / tan_d)
    return _listric(fault.dip, zc, fault.flatten, fault.ramp_base)


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
        if fault.flatten is not None:                   # vertical shear, a rigid footwall: one heave per column, from the throw at z_center
            heave = trace(zc + fault.throw * ww_profile(np.abs(s) / lx)) - trace(zc)
            return np.broadcast_to((plane(h) - plane(h - heave))[..., None], z.shape), 0.0
        r = np.sqrt((s[..., None] / lx) ** 2 + ((z - zc) / sin_d / ly) ** 2)
        d = fault.throw * ww_profile(r)
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
