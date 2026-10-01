"""Salt in a corner-point model: bodies, the strata upturned beside them, and the base of a salt canopy.

Salt is a mask and two terms of the structure, nothing more (rows N1-N39 of ``step5_salt.md`` of the structure research):

* :class:`SaltBody` is an implicit shape. Cells whose centre lies inside it are written ``ACTNUM = 0`` by
  ``to_grdecl(salt=...)``, so the contact is a no-flow boundary (no pore volume, no connection), and the stratigraphy
  runs on through the salt unseen. The outline is a superellipse of semi-axes ``axes`` at the depth ``z_ref``; it leans
  by ``lean`` m per metre of depth (a wall dipping at ``phi`` leans ``cot(phi)``), grows by ``flare`` m per metre of
  depth (negative: salt wider above, an overhang; positive: a pedestal) and is made irregular by ``lobes`` (smooth random
  waves displacing its coordinates, as :func:`resmill.structure.closure`'s ``warp``). The outline is tested at each
  cell's own depth, so a cell beneath an overhang stays active: that is the trap beneath the overhang that 5 of the 9
  producing East Texas stocks have (N27). Pillars are vertical, so the wall is a staircase on cell faces; the error of
  its position is under half a cell, and the upturn beside it must be at least two cells wide (:attr:`SaltBody.max_cell`).
* :func:`salt_upturn` lifts the strata toward the wall by ``A (1 - d/W)^p`` over a folding zone of width ``W`` (d the
  horizontal distance from the contact), with ``A = W tan(dip) / p`` so that the strata meet the contact at ``dip``
  (the largest upturn dip, 5-50 degrees in East Texas, N9; to 85 degrees and overturned in deep water, N15, N16). The
  zone is a hook (50-200 m), a wedge (0.3-1 km) or a megaflap (3-4.6 km) (N12-N15). :func:`salt_thinning` is the
  isochore factor ``1 - a (1 - d/W)^p`` that thins the strata toward the salt (halokinetic wedges, megaflaps 37-93 %).
  The upturn shifts the strata vertically and so keeps their vertical thickness: the thickness measured along the
  bed's normal is already ``cos(dip)`` of it, and ``a`` thins it further.
* :func:`base_of_salt` is the surface a salt sheet rests on, for ``to_grdecl(erode_above=...)``: cells above it are
  salt (inactive), the cells it cuts are truncated against it, and a reservoir below it is a subsalt trap (N30-N35).

Limits, stated: a column of a corner-point grid cannot repeat a section, so there are no overturned flaps and the dip
is capped at :data:`MAX_DIP` (the owner's choice of 2026-10-01; the grid then holds a steeper flank as cells sheared by
tan(dip) cell widths, whose along-bed transmissibility falls by cos^2(dip)); there is no structure inside the salt; the
contact is always sealed (no sheath, no weld leak).
"""
import math

import numpy as np
from scipy.spatial import cKDTree

from .structure import Structure, _as_field, _axes, _mid

# Salt thickness over discovered subsalt reservoirs (m): SMI 200, GB 171, WC 505, Mica, GB 165, Hickory, Tahiti (N30,
# Moore & Brooks 2009 and the MMS pages): median 1.0 km, the canopy "more than 15,000 ft (4,572 m) thick in some places".
_THICKNESS_M = (302.0, 338.0, 515.0, 1006.0, 2118.0, 2438.0, 3353.0)
MAX_THICKNESS = 4600.0    # m: the canopy's thickest
MIN_THICKNESS = 100.0     # m: below the thinnest sample (302 m) a log-normal tail has a weld, not a sheet [J]
MAX_DIP = 85.0            # degrees: the largest upturn dip (the owner's choice over 75, 2026-10-01)
MAX_LOBES = 0.3           # the largest outline irregularity, as a fraction of the radius (closure's warp reaches 0.35)
_RAYS = 16000             # directions the outline is cast along


def _to_segment(p, a, b):
    """Distance from points ``p`` (M, 2) to the segments ``a``-``b`` (M, 2)."""
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-30), 0.0, 1.0)
    return np.hypot(*(p - a - t[:, None] * ab).T)


class SaltBody:
    """A salt body: an outline that depends on depth (see the module docstring and :func:`salt_body`)."""

    def __init__(self, center, axes, azimuth=0.0, z_ref=0.0, lean=(0.0, 0.0), flare=0.0, lobes=0.0, shape=2.0,
                 seed=None):
        problems = [msg for bad, msg in (
            (not min(axes) > 0.0, "axes must be positive"),
            (not shape >= 2.0, "shape (the superellipse exponent) must be at least 2"),
            (not 0.0 <= lobes <= MAX_LOBES, f"lobes must lie in [0, {MAX_LOBES}]"),
            (lobes > 0.0 and seed is None, "lobes needs a seed (a shared default made every body alike)"),
        ) if bad]
        if problems:
            raise ValueError("SaltBody: " + "; ".join(problems))
        self.center, self.axes = (float(center[0]), float(center[1])), (float(axes[0]), float(axes[1]))
        self.azimuth, self.z_ref = float(azimuth), float(z_ref)
        self.lean, self.flare, self.lobes, self.shape = (float(lean[0]), float(lean[1])), float(flare), float(lobes), float(shape)
        self.radius = min(self.axes)                                   # lobes are a fraction of it
        rng = np.random.default_rng(seed)
        self._waves = [(rng.uniform(1.5, 3.5, 6) / self.radius, rng.uniform(0.0, 2.0 * np.pi, 6),
                        rng.uniform(0.0, 2.0 * np.pi, 6)) for _ in range(2)] if lobes else None
        self._widths = []                                              # the folding zones of the terms built on this body
        self._trees = {}

    @property
    def max_cell(self):
        """The widest cell (m) the body can be gridded with: half its narrowest folding zone (inf: none registered)."""
        return 0.5 * min(self._widths) if self._widths else math.inf

    def _gauge(self, x, y, z):
        """Superellipse radius at (x, y, z): below 1 inside the body, 1 on its contact (inf where it has pinched out)."""
        dz = np.asarray(z, dtype=float) - self.z_ref
        dx = np.asarray(x, dtype=float) - (self.center[0] + self.lean[0] * dz)
        dy = np.asarray(y, dtype=float) - (self.center[1] + self.lean[1] * dz)
        az = math.radians(self.azimuth)
        u, v = dx * math.cos(az) - dy * math.sin(az), dx * math.sin(az) + dy * math.cos(az)
        if self._waves:
            amp = self.lobes * self.radius * math.sqrt(2.0 / 6.0)
            du, dv = (amp * sum(np.cos(k * (math.cos(a) * u + math.sin(a) * v) + ph) for k, a, ph in zip(*wave))
                      for wave in self._waves)
            u, v = u + du, v + dv
        a, b = self.axes[0] + self.flare * dz, self.axes[1] + self.flare * dz
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (np.abs(u / a) ** self.shape + np.abs(v / b) ** self.shape) ** (1.0 / self.shape)
        return np.where((a > 0.0) & (b > 0.0), g, np.inf)

    def inside(self, x, y, z=None):
        """Whether points (x, y, z) (m, depth positive down; broadcast together) lie in the salt. ``z`` defaults to
        ``z_ref``."""
        return self._gauge(x, y, self.z_ref if z is None else z) < 1.0

    def outline(self, z=None):
        """The contact at depth ``z`` (default ``z_ref``) as a closed polyline (N, 2): the body's boundary found along
        ``_RAYS`` directions from its centre by bisection."""
        z = self.z_ref if z is None else float(z)
        dz = z - self.z_ref
        a, b = self.axes[0] + self.flare * dz, self.axes[1] + self.flare * dz
        if not (a > 0.0 and b > 0.0):
            raise ValueError(f"the body has no outline at {z:g} m (it has pinched out)")
        cx, cy = self.center[0] + self.lean[0] * dz, self.center[1] + self.lean[1] * dz
        if not self._gauge(cx, cy, z) < 1.0:
            raise ValueError("the lobes carry the body's centre out of it; use a smaller lobes or another seed")
        t = np.linspace(0.0, 2.0 * np.pi, _RAYS, endpoint=False)
        ux, uy = np.cos(t), np.sin(t)
        lo, hi = np.zeros(_RAYS), np.full(_RAYS, 2.0 * max(a, b) + 4.0 * self.lobes * self.radius)
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            inside = self._gauge(cx + mid * ux, cy + mid * uy, z) < 1.0
            lo, hi = np.where(inside, mid, lo), np.where(inside, hi, mid)
        rho = 0.5 * (lo + hi)
        return np.column_stack([cx + rho * ux, cy + rho * uy])

    def distance(self, x, y, z=None, cap=None):
        """Signed horizontal distance (m) from points (x, y) to the contact at depth ``z`` (default ``z_ref``, a single
        depth): negative inside the salt, positive outside, zero on the contact. It is the distance to the outline's
        polyline (nearest vertex, then the two segments there), so its gradient has length 1. With ``cap`` a distance
        beyond it is returned as ``+-cap``, which is much faster (a point far from the contact has many equally near
        vertices to rule out): the upturn only needs the distance within its folding zone."""
        z = self.z_ref if z is None else float(z)
        if z not in self._trees:
            poly = self.outline(z)
            step = float(np.hypot(*np.diff(np.vstack([poly, poly[:1]]), axis=0).T).max())
            self._trees = {**dict(list(self._trees.items())[-3:]), z: (cKDTree(poly), poly, step)}
        tree, poly, step = self._trees[z]
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        p = np.column_stack([x.ravel(), y.ravel()])
        d, i = tree.query(p, distance_upper_bound=math.inf if cap is None else cap + step)
        far = ~np.isfinite(d)
        i = np.where(far, 0, i)
        d = np.minimum(_to_segment(p, poly[i], poly[(i + 1) % len(poly)]), _to_segment(p, poly[i], poly[i - 1]))
        if cap is not None:
            d = np.where(far, cap, np.minimum(d, cap))
        return np.where(self._gauge(x, y, z).ravel() < 1.0, -d, d).reshape(x.shape)


def salt_body(center, axes, azimuth=0.0, z_ref=0.0, lean=(0.0, 0.0), flare=0.0, lobes=0.0, shape=2.0, seed=None):
    """A salt body: stock (equal ``axes``) or wall (axes ratio 3-10; N5).

    ``center`` (x, y) and ``axes`` (the semi-axes along and across ``azimuth``, m) give the outline at depth ``z_ref``
    (m, positive down); ``azimuth`` is the package convention (degrees clockwise from +x, the long axis runs along
    ``(cos az, -sin az)``). ``lean`` = (dx/dz, dy/dz) shifts the outline per metre of depth, ``cot(dip)`` for a wall
    dipping at ``dip`` degrees (0: vertical); ``flare`` = d(radius)/dz changes both semi-axes per metre of depth (< 0
    an overhang, > 0 a pedestal; N3, N8); ``lobes`` (0 to 0.3) is the outline's rms irregularity as a fraction of the
    smaller semi-axis, drawn from ``seed`` (required with lobes); ``shape`` is the superellipse exponent (2 an ellipse,
    larger flatter-sided: walls). Returns a :class:`SaltBody`.
    """
    return SaltBody(center, axes, azimuth, z_ref, lean, flare, lobes, shape, seed)


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
    """Thickness (m) of the salt over a subsalt trap: log-normal fitted to the seven published values of N30 (median
    1.0 km), kept between :data:`MIN_THICKNESS` and :data:`MAX_THICKNESS`. ``rng`` a ``numpy.random.Generator``."""
    logs = np.log(_THICKNESS_M)
    return np.clip(np.exp(rng.normal(logs.mean(), logs.std(ddof=1), size)), MIN_THICKNESS, MAX_THICKNESS)


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
