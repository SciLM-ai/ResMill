"""Salt in a corner-point model: bodies, the strata upturned beside them, and the base of a salt canopy.

Salt is a mask and two terms of the structure, nothing more (rows N1-N39 of ``step5_salt.md`` of the structure research):

* :class:`SaltBody` is an implicit shape. Cells whose centre lies inside it are written ``ACTNUM = 0`` by
  ``to_grdecl(salt=...)``, so the contact is a no-flow boundary (no pore volume, no connection), and the stratigraphy
  runs on through the salt unseen. The outline is a superellipse of semi-axes ``axes`` at the depth ``z_ref``; it leans
  by ``lean`` m per metre of depth (a wall dipping at ``phi`` leans ``cot(phi)``), grows by ``flare`` m per metre of
  depth (negative: salt wider above, an overhang; positive: a pedestal) and is made irregular by ``lobes`` (smooth random
  waves displacing its coordinates, as :func:`resmill.structure.closure`'s ``warp``) and, at the smaller scales, by
  ``rough`` (octaves of :func:`resmill.structure.roughness` of falling sd, ranges one radius down to 150 m). The outline is
  tested at each cell's own depth, so a cell beneath an overhang stays active: that is the trap beneath the overhang that 5 of
  the 9 producing East Texas stocks have (N27). ``overhang=(L, H)`` gives it an underside L wide and H high (the grid
  holds the reservoir interval only, so the part of a trap beneath an overhang is as wide as the underside reaches across the
  interval, ``h0 / (s + H / L)`` for beds of slope ``s``: a steep underside leaves one cell, a gentle one many). Pillars are
  vertical, so the wall is a staircase on cell faces; the error of its position is under half a cell, and the upturn beside it must
  be at least two cells wide (:attr:`SaltBody.max_cell`). The contact the upturn follows is the contour of that implicit shape
  (:meth:`SaltBody.distance`), so the bays of a lobate wall count and nothing assumes the body is star-shaped.
* :func:`salt_upturn` lifts the strata toward the wall by ``A (1 - d/W)^p`` over a folding zone of width ``W`` (d the
  horizontal distance from the contact), with ``A = W tan(dip) / p`` so that the strata meet the contact at ``dip``
  (the largest upturn dip, 5-50 degrees in East Texas, N9; to 85 degrees and overturned in deep water, N15, N16). The
  zone is a hook (50-200 m), a wedge (0.3-1 km) or a megaflap (3-4.6 km) (N12-N15). :func:`salt_thinning` is the
  isochore factor ``1 - a (1 - d/W)^p`` that thins the strata toward the salt (halokinetic wedges, megaflaps 37-93 %).
  The upturn shifts the strata vertically and so keeps their vertical thickness: the thickness measured along the
  bed's normal is already ``cos(dip)`` of it, and ``a`` thins it further. :func:`salt_sequence` is the hook or wedge
  halokinetic sequence (Giles & Rowan 2012): the upturn of a narrow (50-200 m) or broad (300-1,000 m) folding zone and
  :func:`salt_truncation`, the unconformity that cuts the lifted beds off at the angle its ``cut`` gives (``Sequence.angle``,
  and :func:`truncation_cut` the cut for an angle): a hook is cut at over 70 degrees by a nearly flat unconformity that pinches
  the reservoir out within the zone, or by a small cut that keeps the interval to the wall (the beds of a corner-point column
  cannot overturn, so a hook is as steep as :data:`MAX_DIP`).
* :func:`base_of_salt` is the surface a salt sheet rests on, for ``to_grdecl(erode_above=...)``: cells above it are
  salt (inactive), the cells it cuts are truncated against it, and a reservoir below it is a subsalt trap (N30-N35).

Limits, stated: a column of a corner-point grid cannot repeat a section, so there are no overturned flaps and the dip
is capped at :data:`MAX_DIP` (the owner's choice of 2026-10-01). What a steep flank costs, measured in OPM Flow
(``geology_demo/structure/step5/bench``, 58,000 and 230,000 cells, one thread): no connections, since columns that meet at a
pillar share their corners, so every cell keeps its k neighbour and Flow makes no NNC (0 at 30 to 85 degrees; only faults make
them); the same Newton iterations (48 at 85 degrees against 47 at 30 with 50 m cells, 53 against 53 with 25 m), 1.2 to 1.9 times the
linear iterations, and a run time within about 20 % of the gentle flank's. The cells shear instead, and the along-bed
transmissibility of those beside the contact falls: its 10th percentile is 0.41 of the unsheared value at 85 degrees (0.83 at
30) with 50 m cells. There is no structure inside the salt, and the contact is always sealed (no sheath, no weld leak).
"""
import math
from collections import namedtuple

import contourpy
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import ndtr, ndtri

from .structure import Structure, _as_field, _axes, _mid, _wave_sum, _waves, roughness, surface

# Salt thickness over discovered subsalt reservoirs (m): SMI 200, GB 171, WC 505, Mica, GB 165, Hickory, Tahiti (N30,
# Moore & Brooks 2009 and the MMS pages): median 1.0 km, the canopy "more than 15,000 ft (4,572 m) thick in some places".
_THICKNESS_M = (302.0, 338.0, 515.0, 1006.0, 2118.0, 2438.0, 3353.0)
MAX_THICKNESS = 4600.0    # m: the canopy's thickest, just above the 4,572 m it is "more than" in places [J]
MIN_THICKNESS = 100.0     # m: below the thinnest sample (302 m) a log-normal tail has a weld, not a sheet [J]
MAX_DIP = 85.0            # degrees: the largest upturn dip (the owner's choice over 75, 2026-10-01)
MAX_LOBES = 0.3           # the largest outline irregularity, as a fraction of the radius (closure's warp reaches 0.35); at 0.3
                          # the warp can fold a body (3 of 96 stocks had an island and 3 a hole): the contact follows them
MAX_ROUGH = 0.05          # the largest outline roughness, as a fraction of the radius: the slope guard (_MAX_SLOPE) scales the
                          # octaves down together where they would steepen past it, to 0.6-0.9 of rough at hurst 1 (rough_scale)
MIN_RANGE = 150.0         # m: the finest octave of the outline's roughness, three cells of 50 m [J]
_MAX_SLOPE = 0.9          # the steepest slope the roughness' displacement may have: a guard against the warp folding, not a proof
_ABOVE = -1.0e5           # m: a depth above every cell, where the truncating surface cuts nothing
_GRID = 80                # cells across the narrowest semi-axis of the grid the contact is contoured on
_MIN_STEP = 6.0           # m: the finest such grid


def _to_segment(p, a, b):
    """Distance from points ``p`` (M, 2) to the segments ``a``-``b`` (M, 2)."""
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-30), 0.0, 1.0)
    return np.hypot(*(p - a - t[:, None] * ab).T)


class SaltBody:
    """A salt body: an outline that depends on depth (see the module docstring and :func:`salt_body`)."""

    def __init__(self, center, axes, azimuth=0.0, z_ref=0.0, lean=(0.0, 0.0), flare=0.0, lobes=0.0, shape=2.0,
                 seed=None, rough=0.0, hurst=1.0, overhang=None):
        problems = [msg for bad, msg in (
            (not min(axes) > 0.0, "axes must be positive"),
            (not shape >= 2.0, "shape (the superellipse exponent) must be at least 2"),
            (not 0.0 <= lobes <= MAX_LOBES, f"lobes must lie in [0, {MAX_LOBES}]"),
            (lobes > 0.0 and seed is None, "lobes needs a seed (a shared default made every body alike)"),
            (not 0.0 <= rough <= MAX_ROUGH, f"rough must lie in [0, {MAX_ROUGH}]"),
            (rough > 0.0 and seed is None, "rough needs a seed (a shared default made every body alike)"),
            (not hurst >= 0.0, "hurst must be >= 0"),
            (overhang is not None and not (len(overhang) == 2 and min(overhang) > 0.0),
             "overhang must be (lateral extent, height), both positive"),
        ) if bad]
        if problems:
            raise ValueError("SaltBody: " + "; ".join(problems))
        self.center, self.axes = (float(center[0]), float(center[1])), (float(axes[0]), float(axes[1]))
        self.azimuth, self.z_ref = float(azimuth), float(z_ref)
        self.lean, self.flare, self.lobes, self.shape = (float(lean[0]), float(lean[1])), float(flare), float(lobes), float(shape)
        self.overhang = None if overhang is None else (float(overhang[0]), float(overhang[1]))
        self.radius = min(self.axes)                                   # lobes are a fraction of it
        rng = np.random.default_rng(seed)
        self._waves = _waves(rng) if lobes else None
        self._warp, self.rough_scale, shift = self._roughen(rough, float(hurst), rng) if rough else (None, 1.0, 0.0)
        self._pad = shift + 6.0 * math.sqrt(2.0 / 6.0) * lobes * self.radius   # the most the warps move the outline (m)
        self._widths = []                                              # the folding zones of the terms built on this body
        self._contacts = {}

    def _roughen(self, rough, hurst, rng):
        """The displacement fields (m, along and across the body's axes) that roughen its outline at several scales: a sum
        of octaves of :func:`resmill.structure.roughness`, ranges from the larger of the radius and half the long axis, halving
        down to :data:`MIN_RANGE`, each of sd ``rough x radius x (range / radius)^hurst``, drawn over a box round the body and
        resampled once on a grid of 30 m. Both are reduced by one factor ``scale`` (<= 1) where the steepest slope of either would
        exceed :data:`_MAX_SLOPE`, so the realised ``rough`` is ``scale`` times the nominal one (``rough_scale``: 1 where no
        slope is that steep, 0.6-0.9 at hurst 1 and rough 0.05 on bodies of 0.8-3 km radius, 0.1-0.4 at hurst 0.5 or less on the
        larger ones). That keeps the warp from folding to a first approximation only: the lobes' warp can fold too, and a folded
        body has islands or holes, which the contact and the mask both follow. Returns ``(((fu, fv), half), scale, bound)``: two
        Structures over ``[0, 2 half]^2``, the factor and the largest displacement (m)."""
        a0 = self.radius
        top = max(a0, 0.5 * max(self.axes))                           # a wall meanders over its length
        ranges = [a for a in (top / 2 ** i for i in range(24)) if a >= MIN_RANGE] or [a0]
        half = 1.5 * max(self.axes) + 6.0 * rough * a0 * max(1.0, top / a0) ** max(hurst, 1.0)
        size, step = 2.0 * half, MIN_RANGE / 5.0
        grid = np.linspace(0.0, size, int(math.ceil(size / step)) + 1)
        G = np.meshgrid(grid, grid, indexing="ij")
        fields = [sum(roughness(rough * a0 * (a / a0) ** hurst, a, size, size, seed=int(rng.integers(2 ** 31)))(*G)
                      for a in ranges) for _ in range(2)]
        scale = min(1.0, _MAX_SLOPE / max(np.hypot(*np.gradient(f, grid[1])).max() for f in fields))
        return (tuple(surface(scale * f, size, size) for f in fields), half), scale, scale * max(np.abs(f).max() for f in fields)

    @property
    def max_cell(self):
        """The widest cell (m) the body can be gridded with: half its narrowest folding zone (inf: none registered)."""
        return 0.5 * min(self._widths) if self._widths else math.inf

    def _widen(self, dz):
        """How much the semi-axes have grown (m) at ``dz`` m below the reference depth: ``flare`` per metre, plus the overhang's
        lateral extent over its height above the neck (nothing below it, all of it above the shoulder)."""
        wide = self.flare * dz
        if self.overhang is not None:
            wide = wide + self.overhang[0] * np.clip(-dz / self.overhang[1], 0.0, 1.0)
        return wide

    def _gauge(self, x, y, z):
        """Superellipse radius at (x, y, z): below 1 inside the body, 1 on its contact (inf where it has pinched out)."""
        dz = np.asarray(z, dtype=float) - self.z_ref
        dx = np.asarray(x, dtype=float) - (self.center[0] + self.lean[0] * dz)
        dy = np.asarray(y, dtype=float) - (self.center[1] + self.lean[1] * dz)
        az = math.radians(self.azimuth)
        u, v = dx * math.cos(az) - dy * math.sin(az), dx * math.sin(az) + dy * math.cos(az)
        if self._warp:                                                 # the same irregularity at every depth
            (fu, fv), half = self._warp
            q = np.clip(u + half, 0.0, 2.0 * half), np.clip(v + half, 0.0, 2.0 * half)
            u, v = u + fu.fn(*q), v + fv.fn(*q)
        if self._waves:
            du, dv = _wave_sum(self._waves, u / self.radius, v / self.radius)
            u, v = u + self.lobes * self.radius * du, v + self.lobes * self.radius * dv
        a, b = self.axes[0] + self._widen(dz), self.axes[1] + self._widen(dz)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (np.abs(u / a) ** self.shape + np.abs(v / b) ** self.shape) ** (1.0 / self.shape)
        return np.where((a > 0.0) & (b > 0.0), g, np.inf)

    def inside(self, x, y, z=None):
        """Whether points (x, y, z) (m, depth positive down; broadcast together) lie in the salt. ``z`` defaults to
        ``z_ref``."""
        return self._gauge(x, y, self.z_ref if z is None else z) < 1.0

    def _contact(self, z):
        """The contact at depth ``z``: ``(loops, tree, vertices, next, previous, step)``, its closed polylines (counterclockwise
        round the salt, so that the salt lies on the left of each), a KD-tree of all their vertices with the index of the
        two neighbours of each, and the longest segment. It is the contour of the gauge at 1 (marching squares) on a grid
        of ``min(a, b) / 80`` (at least 6 m) over the box that no displacement of the outline can leave, in the body's own axes:
        nothing is assumed about its shape, so a wall, a bay, an island or a hole is in it as the mask has it."""
        if z not in self._contacts:
            dz = z - self.z_ref
            a, b = self.axes[0] + self._widen(dz), self.axes[1] + self._widen(dz)
            if not (a > 0.0 and b > 0.0):
                raise ValueError(f"the body has no outline at {z:g} m (it has pinched out)")
            h = max(_MIN_STEP, min(a, b) / _GRID)
            us, vs = (h * np.arange(-n, n + 1) for n in (int(math.ceil((r + self._pad) / h)) + 2 for r in (a, b)))
            c, s = math.cos(math.radians(self.azimuth)), math.sin(math.radians(self.azimuth))
            cx, cy = self.center[0] + self.lean[0] * dz, self.center[1] + self.lean[1] * dz
            U, V = np.meshgrid(us, vs, indexing="ij")
            g = self._gauge(cx + c * U + s * V, cy - s * U + c * V, z)
            lines = contourpy.contour_generator(us, vs, g.T, line_type=contourpy.LineType.Separate).lines(1.0)
            loops = [np.column_stack([cx + c * l[:, 0] + s * l[:, 1], cy - s * l[:, 0] + c * l[:, 1]])[-2::-1]
                     for l in lines if len(l) > 3]                           # reversed (salt on the left), closing vertex dropped
            loops = [l[(l != np.roll(l, -1, axis=0)).any(axis=1)] for l in loops]   # a grid point on the contact comes three times
            sizes = np.array([len(l) for l in loops])
            first = np.repeat(np.cumsum(sizes) - sizes, sizes)
            at = np.arange(sizes.sum()) - first
            n = np.repeat(sizes, sizes)
            V = np.vstack(loops)
            nxt, prv = first + (at + 1) % n, first + (at - 1) % n
            step = float(np.hypot(*(V[nxt] - V).T).max())
            self._contacts = {**dict(list(self._contacts.items())[-3:]), z: (loops, cKDTree(V), V, nxt, prv, step)}
        return self._contacts[z]

    def outline(self, z=None):
        """The contact at depth ``z`` (default ``z_ref``) as a closed polyline (N, 2), counterclockwise round the salt: the
        longest closed curve of the body's boundary there (a lobate body can have islands and holes as well, which
        :meth:`distance` honours and this does not return)."""
        return max(self._contact(self.z_ref if z is None else float(z))[0], key=len)

    def normal(self, x, y, z=None):
        """Unit normals (..., 2) pointing out of the salt at points (x, y) on its contact at depth ``z`` (default ``z_ref``): the
        gradient of the gauge, which is exact where the polyline's tangents wobble with the spacing of its vertices."""
        z = self.z_ref if z is None else float(z)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        g = np.stack([self._gauge(x + 0.5, y, z) - self._gauge(x - 0.5, y, z),
                      self._gauge(x, y + 0.5, z) - self._gauge(x, y - 0.5, z)], axis=-1)
        return g / np.hypot(g[..., 0], g[..., 1])[..., None]

    def distance(self, x, y, z=None, cap=None):
        """Signed horizontal distance (m) from points (x, y) to the contact at depth ``z`` (default ``z_ref``, a single
        depth): negative inside the salt, positive outside, zero on the contact. It is the distance to the contact's polylines
        (all of them: nearest vertex, then the two segments there), so its gradient has length 1. With ``cap`` a distance
        beyond it is returned as ``+-cap``, which is much faster (a point far from the contact has many equally near
        vertices to rule out): the upturn only needs the distance within its folding zone."""
        z = self.z_ref if z is None else float(z)
        _, tree, V, nxt, prv, step = self._contact(z)
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        p = np.column_stack([x.ravel(), y.ravel()])
        d, i = tree.query(p, distance_upper_bound=math.inf if cap is None else cap + step)
        far = ~np.isfinite(d)
        i = np.where(far, 0, i)
        d = np.minimum(_to_segment(p, V[i], V[nxt[i]]), _to_segment(p, V[i], V[prv[i]]))
        if cap is not None:
            d = np.where(far, cap, np.minimum(d, cap))
        return np.where(self._gauge(x, y, z).ravel() < 1.0, -d, d).reshape(x.shape)


def salt_body(center, axes, azimuth=0.0, z_ref=0.0, lean=(0.0, 0.0), flare=0.0, lobes=0.0, shape=2.0, seed=None,
              rough=0.0, hurst=1.0, overhang=None):
    """A salt body: stock (equal ``axes``) or wall (axes ratio 3-10; N5).

    ``center`` (x, y) and ``axes`` (the semi-axes along and across ``azimuth``, m) give the outline at depth ``z_ref``
    (m, positive down); ``azimuth`` is the package convention (degrees clockwise from +x, the long axis runs along
    ``(cos az, -sin az)``). ``lean`` = (dx/dz, dy/dz) shifts the outline per metre of depth, ``cot(dip)`` for a wall
    dipping at ``dip`` degrees (0: vertical); ``flare`` = d(radius)/dz changes both semi-axes per metre of depth (< 0
    an overhang, > 0 a pedestal; N3, N8); ``lobes`` (0 to 0.3) is the outline's rms irregularity as a fraction of the
    smaller semi-axis, drawn from ``seed`` (required with lobes); ``shape`` is the superellipse exponent (2 an ellipse,
    larger flatter-sided: walls). ``rough`` (0 to :data:`MAX_ROUGH`, a fraction of the smaller semi-axis) adds the irregularity
    of real outlines at the smaller scales: octaves of :func:`resmill.structure.roughness` from a range of one radius (half the
    long axis of a wall, if larger) halving down to :data:`MIN_RANGE` (150 m), each of sd ``rough x radius x (range /
    radius)^hurst``, displacing the outline along its normal and moving with its lean (the Santos stock and the Sigsbee
    feeders show 4-6 % of the radius at wavelengths of 1-2 radii and 2-3 times less per octave: ``hurst`` about 1); ``seed``
    is required with it. A slope limit scales all the octaves down together where one would fold the outline: the roughness
    realised is ``rough_scale`` (an attribute, 1 where nothing is that steep) times ``rough``. ``overhang`` = (lateral
    extent L, height H) (m) makes the salt L wider than at ``z_ref`` (the neck) from H above it upward, linearly between: an
    underside dipping ``atan(H / L)`` from horizontal, which a reservoir lifted into it meets (East Texas stocks overhang by
    0.15-2.6 km, P50 0.37 km, over 0.5-2.4 km of height, an underside of 35-68 degrees; the shoulders of Precaspian walls
    are 0.3-1.5 km wide at 15-30 degrees). Returns a :class:`SaltBody`.
    """
    return SaltBody(center, axes, azimuth, z_ref, lean, flare, lobes, shape, seed, rough, hurst, overhang)


def salt_cells(salt, Xc, Yc, Zc):
    """The cells in ``salt`` as a boolean ``(nx, ny, nz)`` mask in the exporter's top-down k order: those whose centre
    lies inside the body at the cell's own mid-depth, on the doubled corner arrays of
    :func:`resmill.export._build_geometry`."""
    nx, ny = Xc.shape[0] // 2, Xc.shape[1] // 2
    xm, ym = (a.reshape(nx, 2, ny, 2).mean(axis=(1, 3)) for a in (Xc, Yc))
    zm = Zc.reshape(nx, 2, ny, 2, -1).mean(axis=(1, 3))
    return salt.inside(xm[..., None], ym[..., None], 0.5 * (zm[..., :-1] + zm[..., 1:]))


def _taper(d, width, power):
    """The profile ``(1 - d/W)^p``: 1 at and inside the contact, 0 from ``width`` on."""
    return np.clip(1.0 - np.asarray(d, dtype=float) / width, 0.0, 1.0) ** power


def _zone(salt, width, power):
    if not width > 0.0 or not power >= 1.0:
        raise ValueError(f"the folding zone needs width > 0 and power >= 1, got {width}, {power}")
    salt._widths.append(float(width))


def salt_upturn(salt, dip, width, power=2.0, z_ref=None):
    """Strata upturned against a salt contact: a structure term lifting them toward the salt.

    The shift is ``-A (1 - d/W)^p`` m (negative: up) at horizontal distance ``d`` from the contact at depth ``z_ref``
    (default the body's) for ``0 < d < W``, ``-A`` inside it and zero beyond ``W``, with ``A = W tan(dip) / p``: the
    strata meet the contact at ``dip`` degrees and flatten out smoothly. ``power`` 2 is a parabola; a larger one
    concentrates the upturn at the contact, and a relief ``A`` is reached with ``p = W tan(dip) / A`` (megaflaps
    measure 2.5-7.3 km of relief over folding widths of 3.1-4.6 km, N15: a steep one needs p well above 2).
    ``dip`` is capped at :data:`MAX_DIP`. ``width`` (the folding zone) also sets the widest cell the grid may have:
    ``to_grdecl`` refuses cells wider than ``salt.max_cell``, half of it.
    """
    if not dip >= 0.0:
        raise ValueError(f"dip must be >= 0, got {dip}")
    _zone(salt, width, power)
    peak = width * math.tan(math.radians(min(float(dip), MAX_DIP))) / power
    return Structure(lambda x, y: -peak * _taper(salt.distance(x, y, z_ref, width), width, power))


def salt_thinning(salt, a, width, power=2.0, z_ref=None):
    """Strata thinning toward a salt contact: an isochore factor ``1 - a (1 - d/W)^p`` for ``to_grdecl(isochore=[...])``
    (one entry per layer, the same term in each), ``a`` the fraction lost at the contact (halokinetic wedges and
    megaflaps thin by 37-93 %, N15; a = 0 leaves the thickness). It thins the vertical thickness, on top of the
    ``cos(dip)`` that an upturn's vertical shift already takes off the thickness along the bed's normal. Like
    :func:`salt_upturn` it registers ``width`` on the body."""
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"a must lie in [0, 1], got {a}")
    _zone(salt, width, power)
    return Structure(lambda x, y: 1.0 - a * _taper(salt.distance(x, y, z_ref, width), width, power))


def salt_truncation(upturn, datum, cut):
    """The unconformity that truncates the beds an upturn lifts, as an absolute-depth surface for
    ``to_grdecl(erode_above=...)``: cells above it are removed. Giles & Rowan (2012) find the beds beside a diapir truncated
    beneath a bounding unconformity, at over 70 degrees in a hook and under 30 in a wedge.

    Where ``upturn`` (:func:`salt_upturn`) lifts the beds the surface is ``datum`` (the reservoir's top before the upturn: the
    fold, its roughness and the top depth, any Structure-like) lifted by the fraction ``1 - cut`` of the upturn, so ``cut`` is
    the share of the lift it removes: 1 a flat surface (the beds are cut off at their full dip), 0 none. The angle between the
    beds and the surface at the contact is ``dip - atan((1 - cut) tan dip)``. A reservoir ``h`` thick is gone where ``cut`` x
    lift >= ``h``: it pinches out within the folding zone (Pichel & Jackson: up to 200 m from the salt for a hook, 300-1000 m
    for a wedge), and survives to the wall where ``cut`` x the peak lift stays under ``h``. Beyond the zone the surface lies far
    above the model and cuts nothing, so faults there keep their footwalls. It cuts the faulted stack, as the faults of a tier
    end at the unconformity above (Coleman et al. 2018)."""
    if not 0.0 <= cut <= 1.0:
        raise ValueError(f"cut must lie in [0, 1], got {cut}")
    datum, lift = _as_field(datum), _as_field(upturn)

    def fn(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        shift = lift(x, y)
        return np.where(shift < 0.0, datum(x, y) + (1.0 - cut) * shift, _ABOVE)

    return Structure(fn)


Sequence = namedtuple("Sequence", "upturn truncation relief dip angle")


def sequence_dip(taper, power=2.0):
    """The dip (degrees) at which the beds of a fold of this ``taper`` meet the contact: atan(p tan(taper)), capped at MAX_DIP;
    the relief of a folding zone ``W`` wide is then ``W tan(dip) / power`` (:attr:`Sequence.relief`)."""
    return min(math.degrees(math.atan(power * math.tan(math.radians(taper)))), MAX_DIP)


def salt_sequence(salt, width, taper, cut, datum, power=2.0, z_ref=None):
    """A halokinetic sequence beside a salt contact: its upturn and truncating unconformity (a :class:`Sequence`, for
    ``structure=`` and ``erode_above=``), from what Giles & Rowan (2012) and Pichel & Jackson (EarthArXiv 657) measure.

    ``width`` is the folding zone and ``taper`` the angle of the line from the fold's inflection point to its tip, so the
    relief is ``width x tan(taper)`` (at ``power`` 2, a parabola, the beds meet the contact at ``atan(2 tan taper)``, capped
    at :data:`MAX_DIP`; the relief is W tan(taper) up to a taper of about 80 degrees and W tan(85) / 2 above it). Hooks have
    20-200 m zones (P10-P90 38-181 m; 36 % of the 96 sequences of the Precaspian walls, 50 % on upright and 19 % on inclined
    walls) and tapers of 40-86 degrees; wedges 300-1,970 m (P10-P90 375-1,020 m) and 8-49 degrees. ``cut`` is the share of the
    lift the unconformity removes (see :func:`salt_truncation`) and the truncation angle at the contact, which Giles & Rowan
    put over 70 degrees for a hook and under 30 for a wedge, is ``Sequence.angle`` = dip - atan((1 - cut) tan dip): not a
    function of ``cut`` alone (a cut of 0.85-1 gives 46-75 degrees at a dip of 75), so :func:`truncation_cut` gives the cut for
    the angle wanted. A hook that pinches the reservoir out is cut at over 70 degrees (a cut of 0.97-1); one that keeps the
    interval to the wall has a ``cut`` x ``relief`` under the interval's thickness. ``Sequence.dip`` is the dip the grid has (capped).
    ``datum`` is the reservoir's top before the upturn (see :func:`salt_truncation`). Like :func:`salt_upturn` it registers
    ``width`` on the body: the grid must have cells of at most half of it, a hook's 50-200 m included."""
    if not 0.0 <= taper < 90.0:
        raise ValueError(f"taper must lie in [0, 90) degrees, got {taper}")
    dip = sequence_dip(taper, power)
    up = salt_upturn(salt, dip, width, power, z_ref)
    truncation = salt_truncation(up, datum, cut)
    angle = dip - math.degrees(math.atan((1.0 - cut) * math.tan(math.radians(dip))))
    return Sequence(up, truncation, width * math.tan(math.radians(dip)) / power, dip, angle)


def truncation_cut(taper, angle, power=2.0):
    """The ``cut`` for :func:`salt_sequence` that makes the unconformity meet the beds at ``angle`` degrees at the contact:
    ``1 - tan(dip - angle) / tan(dip)``, the dip being that of a fold of this ``taper`` (``atan(power tan taper)``, capped at
    :data:`MAX_DIP`). The angle cannot exceed the dip (a flat unconformity gives the dip itself), so a hook cut at over 70 degrees
    (Giles & Rowan 2012) needs a taper above ``atan(tan(70) / power)`` = 54 degrees at power 2, and then a cut of 0.969-1."""
    dip = sequence_dip(taper, power)
    if not 0.0 <= angle <= dip:
        raise ValueError(f"beds dipping {dip:.1f} degrees at the contact cannot be cut at {angle:g} degrees "
                         f"(the largest angle is the dip, for a flat unconformity)")
    tan = math.tan(math.radians(dip))
    return 1.0 - math.tan(math.radians(dip - angle)) / tan if tan > 0.0 else 0.0


class SaltBase(Structure):
    """The base of a salt sheet: a Structure of absolute depths for ``erode_above``, carrying the sheet's ``thickness``
    (m; a number, a Structure-like field or None: not drawn), which the grid does not hold."""

    def __init__(self, fn, thickness=None):
        super().__init__(fn)
        self.thickness = thickness


def base_of_salt(depth, dip=0.0, azimuth=0.0, rough=None, high=None, center=None, thickness=None):
    """The surface a salt sheet rests on, for ``to_grdecl(erode_above=...)``: cells above it are salt (they collapse and
    are written inactive), the cells it cuts are truncated against it, and a reservoir below is a subsalt trap.

    A plane at ``depth`` (m, positive down) through ``center`` (default the middle of the map), dipping ``dip`` degrees
    and deepening along the normal of ``azimuth`` (as :func:`resmill.structure.ramp`), plus ``rough`` (a rugose base:
    :func:`resmill.structure.roughness`) and ``high`` (a base-salt high or feeder: any Structure-like, negative
    lifts). Under it, three-way traps are truncated against the salt (Tahiti, Heidelberg: ``depth`` at the crest plus a
    fraction 0.2-0.8 [J] of the closure's relief, ``closure_stats(...)["spill_depth"] - (1 - f) * height``) and four-way
    folds lie beneath a cover (Mad Dog, Atlantis: a base above the crest, which cuts nothing and is only a label); the
    base of the Sigsbee canopy dips 0 to over 90 degrees, mostly under 35 (N31). ``thickness`` is the label (draw it
    with :func:`salt_thickness`); :func:`salt_labels` returns it with the base and the cut cells.
    """
    nx_, ny_ = _axes(azimuth)
    slope = math.tan(math.radians(dip))
    fields = [_as_field(f) for f in (rough, high) if f is not None]

    def fn(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        cx, cy = center if center is not None else (_mid(x), _mid(y))
        return depth + slope * ((x - cx) * nx_ + (y - cy) * ny_) + sum(f(x, y) for f in fields)

    return SaltBase(fn, thickness)


def salt_thickness(rng, size=None):
    """Thickness (m) of the salt over a subsalt trap: the log-normal fitted to the seven published values of N30 (median
    0.99 km, sample sd 0.99 in ln m) truncated, not clipped, to :data:`MIN_THICKNESS`-:data:`MAX_THICKNESS` by the inverse CDF, so
    that none of the draws sits at a bound: P10/P50/P90 283 / 929 / 2,719 m. ``rng`` a ``numpy.random.Generator``."""
    logs = np.log(_THICKNESS_M)
    mu, sd = logs.mean(), logs.std(ddof=1)
    lo, hi = ndtr((np.log([MIN_THICKNESS, MAX_THICKNESS]) - mu) / sd)
    return np.exp(mu + sd * ndtri(lo + rng.uniform(size=size) * (hi - lo)))


def salt_labels(model, salt=None, structure=None, top=None, base=None, erode_above=None, erode_below=None,
                isochore=None, onlap=False, faults=None):
    """What an episode carries about its salt, for the geometry ``to_grdecl`` writes (the same shaping arguments, with
    the :class:`SaltBase` of a subsalt model as ``erode_above``).

    Returns a dict: ``mask`` (uint8, ``(nx, ny, nz)``, the exporter's top-down k order: 1 where a cell is salt: in
    ``salt`` or collapsed by the base of salt), ``thickness`` and ``base`` (float32, ``(nx, ny)``, m) and
    ``volume_fraction`` (the salt's share of the cells). For a body they are what the grid holds: the vertical extent of
    its cells in the column (0 where none) and the depth of the bottom of the deepest (NaN where none; at the model's
    bottom where the salt runs on below it, at an overhang's underside above sediment). For a base of salt they are the
    sheet's own: its ``thickness`` where drawn (NaN where not) and its depth, in every column, cut cells or not.
    """
    from .export import _build_geometry, _field
    layers = list(getattr(model, "layers", [model]))
    shape = dict(structure=structure, top=top, base=base, erode_below=erode_below, isochore=isochore, onlap=onlap,
                 faults=faults)
    Xc, Yc, Zc, act = _build_geometry(layers, erode_above=erode_above, salt=salt, **shape)
    nx, ny, nz = act.shape
    mask = salt_cells(salt, Xc, Yc, Zc) if salt is not None else np.zeros(act.shape, dtype=bool)
    sheet = isinstance(erode_above, SaltBase)
    if sheet:
        mask |= (_build_geometry(layers, salt=salt, **shape)[3] > 0) & (act == 0)       # the cells the base collapsed
    z = Zc.reshape(nx, 2, ny, 2, -1).mean(axis=(1, 3))
    thickness = np.where(mask, z[..., 1:] - z[..., :-1], 0.0).sum(axis=2)
    bottom = np.where(mask.any(axis=2), np.where(mask, z[..., 1:], -np.inf).max(axis=2), np.nan)
    if sheet:
        xm, ym = (a.reshape(nx, 2, ny, 2).mean(axis=(1, 3)) for a in (Xc, Yc))
        bottom = erode_above(xm, ym)
        thickness = np.full((nx, ny), np.nan) if erode_above.thickness is None else \
            _field(erode_above.thickness, layers[0].x_len, layers[0].y_len)(xm, ym)
    return {"mask": mask.astype(np.uint8), "thickness": thickness.astype(np.float32),
            "base": bottom.astype(np.float32), "volume_fraction": float(mask.mean())}
