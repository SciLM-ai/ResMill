"""Explicit constructions the fault tests hold the code to, written without the code's own forms (np.log and bisection for the listric plane,
a brute-force sampling of every shared face for the FAULTS records)."""
import math

import numpy as np


def lateral_contacts(side, zc):
    """Brute force: (face, i, j, k) of every cell meeting a cell of the other side across its + face, each shared face
    sampled at 201 points between its pillars."""
    t = np.linspace(0.0, 1.0, 201)[:, None]
    out = set()
    nx, ny, _ = side.shape
    for axis, face in ((0, "X"), (1, "Y")):
        for i in range(nx - (axis == 0)):
            for j in range(ny - (axis == 1)):
                lo, hi = side[i, j], side[i + 1, j] if axis == 0 else side[i, j + 1]
                other = lo[:, None] * hi[None, :] == -1
                if not other.any():
                    continue
                if axis == 0:
                    a, b = zc[2 * i + 1, 2 * j:2 * j + 2], zc[2 * i + 2, 2 * j:2 * j + 2]
                else:
                    a, b = zc[2 * i:2 * i + 2, 2 * j + 1], zc[2 * i:2 * i + 2, 2 * j + 2]
                A, B = a[0] + t * (a[1] - a[0]), b[0] + t * (b[1] - b[0])     # (201, nk)
                overlap = (np.minimum(A[:, 1:, None], B[:, None, 1:]) - np.maximum(A[:, :-1, None], B[:, None, :-1]))
                for k in np.flatnonzero(((overlap > 1e-6).any(axis=0) & other).any(axis=1)):
                    out.add((face, i, j, int(k)))
    return out


def log_plane(dip, z_c, flatten):
    """The listric plane written out again with np.log and bisection (nothing of the code's log1p and expm1 forms): straight
    at ``dip`` down to the bend (x < 0, x from the trace at z_c), then z_c + L ln(1 + x tan(dip) / L), L = ``flatten``, the
    plane whose tan(dip) falls as exp(-(z - z_c) / L). Returns the plane's depth at x and, by bisection, x at a depth."""
    t = math.tan(math.radians(dip))

    def plane(x):
        x = np.asarray(x, float)
        return np.where(x < 0.0, z_c + x * t, z_c + flatten * np.log(1.0 + np.maximum(x, 0.0) * t / flatten))

    def inverse(z):
        z = np.asarray(z, float)
        lo, hi = np.full(z.shape, -1.0e6), np.full(z.shape, 1.0e6)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            below = plane(mid) < z
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    return plane, inverse


def research_rollover(dip, z_b, flatten, z_r, throw, z_anchor=None, z_ref=None):
    """The rollover of a flat horizon at z_r over such a plane (ramp to z_b, the bend), by vertical shear with constant heave
    (Gibbs 1983; White et al. 1986): the footwall cutoff is at x_c on the plane, the hanging-wall cutoff where the plane is
    z_r + throw, the heave H between them, and the horizon's depth z_r + F(x_c + p) - F(x_c + p - H) at the distance p east of
    the footwall cutoff. One heave moves the whole hanging wall: ``throw`` is that of the horizon at ``z_ref`` (default z_r), whose
    cutoffs give it. Returns H, x_c (from the bend's trace, or from the trace at ``z_anchor`` when given) and that depth."""
    plane, inverse = log_plane(dip, z_b, flatten)
    z_ref = z_r if z_ref is None else z_ref
    x_c = float(inverse(z_r))
    heave = float(inverse(z_ref + throw)) - float(inverse(z_ref))
    x_a = 0.0 if z_anchor is None else float(inverse(z_anchor))
    return heave, x_c - x_a, (lambda p: z_r + plane(x_c + p) - plane(x_c + p - heave))
