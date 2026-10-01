"""Finite faults on the corner-point grid.

A :class:`Fault` is a normal (or reverse) fault of finite extent. Its displacement is largest at its
centre and dies out to zero at an elliptical tip line ``length`` long and ``length / aspect`` high,
following Walsh & Watterson (1987), D = Dmax (1 - r)^1.5 (1 + 3 r)^0.5, with a tip-line aspect of 2.15
(Nicol et al. 1996). The displacement is split between the hanging wall (``hw_share``) and the footwall,
and bends the layers next to the fault (reverse drag) with a taper (1 - d / reach)^2 over ``drag`` times
the length on each side (Georgsen et al. 2012; Wu et al. 2020). The fault dips at ``dip`` toward its
hanging wall, so its trace moves with depth, and its trace may curve (``radius``).

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
    return s, fault.hanging_wall * h


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
        zc = float(Zcell[i0, j0, Zc.shape[2] // 2])
    else:
        zc = float(fault.z_center)
    lx = 0.5 * fault.length
    ly = lx / fault.aspect

    def displacement(s, h, z):
        r = np.sqrt((s[..., None] / lx) ** 2 + ((z - zc) / sin_d / ly) ** 2)
        d = fault.throw * ww_profile(r)
        taper = []
        for reach in fault.drag:
            if reach > 0.0:
                taper.append(np.clip(1.0 - np.abs(h) / (reach * fault.length), 0.0, None) ** 2)
            else:
                taper.append(np.ones_like(h))
        return fault.hw_share * d * taper[0][..., None], (1.0 - fault.hw_share) * d * taper[1][..., None]

    sign = -1.0 if fault.reverse else 1.0
    s_k, h_k = _frame(fault, Xc, Yc)
    s_c, h_c = _frame(fault, Xm, Ym)
    dhw_k, dfw_k = displacement(s_k, h_k, Zc)
    dhw_c, dfw_c = displacement(s_c, h_c, Zcell)
    zp = zc + h_c * tan_d                                         # the fault plane's depth under each column
    hw = Zcell + sign * dhw_c <= zp[..., None]
    fw = Zcell - sign * dfw_c >= zp[..., None]
    Znew = np.where(_corners(hw), Zc + sign * dhw_k,
                    np.where(_corners(fw), Zc - sign * dfw_k, _corners(zp)[..., None]))
    reach = (dhw_c + dfw_c) > 1e-9
    mid = 0.5 * (hw[..., 1:].astype(int) + hw[..., :-1] - fw[..., 1:] - fw[..., :-1].astype(int))
    above = 0.5 * (Zcell[..., 1:] + Zcell[..., :-1]) <= zp[..., None]      # a cell the plane cuts: its middle's side
    mid = np.where(mid == 0, np.where(above, 1.0, -1.0), mid)
    side = np.where(reach[..., 1:] | reach[..., :-1], np.sign(mid), 0).astype(np.int8)
    return np.maximum.accumulate(Znew, axis=2), side


def face_records(name, side):
    """``FAULTS`` records (1-based I1 I2 J1 J2 K1 K2 and face) of the cell faces between the two sides."""
    records = []
    for axis, face in ((0, "X"), (1, "Y")):
        lo = np.take(side, np.arange(side.shape[axis] - 1), axis=axis)
        hi = np.take(side, np.arange(1, side.shape[axis]), axis=axis)
        cut = (lo * hi) == -1
        for i, j in zip(*np.nonzero(cut.any(axis=2))):
            ks = np.flatnonzero(cut[i, j])
            breaks = np.flatnonzero(np.diff(ks) > 1)
            for run in np.split(ks, breaks + 1):
                records.append((name, i + 1, i + 1, j + 1, j + 1, int(run[0]) + 1, int(run[-1]) + 1, face))
    return records
