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

:func:`apply_fault` displaces the interface stack in 3-D and reports which side of the fault each cell
ended on; :func:`face_records` turns that into the stair-stepped cell faces the GRDECL export writes as
``FAULTS``, with the fault's ``mult`` as ``MULTFLT``. ``azimuth``-style angles follow the package
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
    seed: int = 0                  # draws the bends


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


def apply_fault(fault, Xc, Yc, Zc):
    """Displace the interface stack ``Zc`` (2nx, 2ny, nk) by ``fault``.

    Returns the new stack (depths still increasing downward) and ``side`` (nx, ny, nk - 1): +1 for
    cells that ended in the hanging wall, -1 in the footwall, 0 where the fault did not reach.
    """
    nx, ny = Xc.shape[0] // 2, Xc.shape[1] // 2
    Xm, Ym = _cells(Xc, nx, ny), _cells(Yc, nx, ny)
    Zcell = _cells(Zc, nx, ny)
    sin_d, tan_d = math.sin(math.radians(fault.dip)), math.tan(math.radians(fault.dip))
    if fault.z_center is None:
        i0 = int(np.clip(np.argmin(np.abs(Xm[:, 0] - fault.center[0])), 0, nx - 1))
        j0 = int(np.clip(np.argmin(np.abs(Ym[0, :] - fault.center[1])), 0, ny - 1))
        zc = float(0.5 * (Zcell[i0, j0, 0] + Zcell[i0, j0, -1]))
    else:
        zc = float(fault.z_center)
    lx = 0.5 * fault.length
    ly = lx / fault.aspect

    def displacement(s, h, z):
        r = np.sqrt((s[..., None] / lx) ** 2 + ((z - zc) / sin_d / ly) ** 2)
        d = fault.throw * ww_profile(r)
        hp = np.abs(h[..., None] if fault.reverse else h[..., None] - (z - zc) / tan_d)   # from the plane at depth z
        taper = [np.clip(1.0 - hp / (reach * fault.length), 0.0, None) ** 2 if reach > 0.0 else 1.0
                 for reach in fault.drag]
        return fault.hw_share * d * taper[0], (1.0 - fault.hw_share) * d * taper[1]

    sign = -1.0 if fault.reverse else 1.0
    s_k, h_k = _frame(fault, Xc, Yc)
    s_c, h_c = _frame(fault, Xm, Ym)
    dhw_k, dfw_k = displacement(s_k, h_k, Zc)
    dhw_c, dfw_c = displacement(s_c, h_c, Zcell)
    zp = zc + h_c * tan_d                                         # the fault plane's depth under each column
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
            a, b = np.vstack([a, a.mean(axis=0)]), np.vstack([b, b.mean(axis=0)])   # both pillars and the middle
            overlap = np.minimum(a[:, 1:, None], b[:, None, 1:]) - np.maximum(a[:, :-1, None], b[:, None, :-1])
            other = lo[i, j][:, None] * hi[i, j][None, :] == -1
            ks = np.flatnonzero(((overlap > 1e-6).any(axis=0) & other).any(axis=1))
            for run in np.split(ks, np.flatnonzero(np.diff(ks) > 1) + 1) if ks.size else ():
                records.append((name, i + 1, i + 1, j + 1, j + 1, int(run[0]) + 1, int(run[-1]) + 1, face))
    for i, j in zip(*np.nonzero((side > 0).any(axis=2) & (side < 0).any(axis=2))):
        live = np.flatnonzero(act[i, j])
        s = side[i, j, live]
        for k in live[:-1][s[:-1] * s[1:] == -1]:
            records.append((name, i + 1, i + 1, j + 1, j + 1, int(k) + 1, int(k) + 1, "Z"))
    return records
