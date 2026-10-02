"""Structural deformation fields applied at export time.

A :class:`Structure` is a scalar field ``f(x, y) -> dz`` giving a vertical
shift in meters for every map position, with the package's positive-down
depth convention: **negative values lift rock toward the surface, positive
values push it deeper**. Presets are written so their ``amplitude``/``throw``
arguments read naturally (``anticline(amplitude=60)`` lifts the crest 60 m).

Structures compose with ``+``, ``-``, unary ``-`` and scalar ``*``; the other
operand may be another ``Structure``, a plain callable ``f(x, y)``, or a
scalar (a uniform burial shift). The exporter accepts any of these forms
directly, plus a 2-D array resampled over the model footprint.

Every field is evaluated on physical coordinates in meters, in the model's
own frame: x in ``[0, x_len]``, y in ``[0, y_len]``. Where a preset takes a
``center`` (or ``x0``/``y0``) and it is left ``None``, the midpoint of the
coordinates being evaluated is used, which for the exporter is the middle
of the model footprint.

``azimuth`` follows the package convention (degrees clockwise from +x, the
same convention as ``LobeLayer``/``ChannelLayer``): the structure's long
axis (fold hinge, fault trace) runs along ``(cos az, -sin az)``.
"""

import heapq
import math

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from .faults import _frame, _plane, ww_profile


def _as_field(obj):
    """Coerce Structure | callable | scalar to a plain callable f(x, y)."""
    if isinstance(obj, Structure):
        return obj.fn
    if callable(obj):
        return obj
    value = float(obj)
    return lambda x, y: value


def _mid(v):
    v = np.asarray(v, dtype=float)
    return 0.5 * (v.min() + v.max())


class Structure:
    """Composable vertical-shift field ``f(x, y) -> meters`` (positive down)."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.zeros(np.broadcast(x, y).shape) + self.fn(x, y)

    def __add__(self, other):
        f, g = self.fn, _as_field(other)
        return Structure(lambda x, y: f(x, y) + g(x, y))

    __radd__ = __add__

    def __sub__(self, other):
        f, g = self.fn, _as_field(other)
        return Structure(lambda x, y: f(x, y) - g(x, y))

    def __neg__(self):
        f = self.fn
        return Structure(lambda x, y: -f(x, y))

    def __mul__(self, factor):
        f = self.fn
        factor = float(factor)
        return Structure(lambda x, y: factor * f(x, y))

    __rmul__ = __mul__


def _axes(azimuth):
    """Unit normal to the structure's long axis (its dip direction)."""
    az = np.radians(azimuth)
    return np.sin(az), np.cos(az)


def anticline(amplitude, wavelength, azimuth=0.0, center=None):
    """Cosine fold train with its hinge along ``azimuth``.

    The crest is lifted by ``amplitude`` (m) above the regional datum and
    the flanks return to the datum half a wavelength away; crests repeat
    every ``wavelength`` (m), so a single fold needs ``wavelength`` at
    least the footprint size. Negative amplitude gives a syncline.

    The fold is cylindrical (uniform along the hinge, open at both ends).
    For a trap that closes in every direction, a four-way dip closure,
    use :func:`dome` with ``aspect``.
    """
    nx_, ny_ = _axes(azimuth)

    def fn(x, y):
        cx, cy = center if center is not None else (_mid(x), _mid(y))
        t = (np.asarray(x, dtype=float) - cx) * nx_ + (np.asarray(y, dtype=float) - cy) * ny_
        return -0.5 * amplitude * (1.0 + np.cos(2.0 * np.pi * t / wavelength))

    return Structure(fn)


def syncline(amplitude, wavelength, azimuth=0.0, center=None):
    """Cosine trough: ``-anticline(...)`` with the axis sagging by ``amplitude``."""
    return -anticline(amplitude, wavelength, azimuth=azimuth, center=center)


def dome(amplitude, radius, center=None, aspect=1.0, azimuth=0.0):
    """Elliptical Gaussian dome: a four-way dip closure.

    Uplift is ``amplitude * exp(-(a / (aspect * radius))**2 - (c / radius)**2)``
    with ``a`` the distance along the ``azimuth`` axis and ``c`` across it,
    so dip falls away from the crest in every direction. ``aspect=1``
    (default) is a circular dome; ``aspect > 1`` elongates it into a
    doubly plunging (periclinal) anticline, the classic structural trap.
    ``radius`` is the cross-axis e-folding distance in meters.
    """
    nx_, ny_ = _axes(azimuth)
    az = np.radians(azimuth)
    sx_, sy_ = np.cos(az), -np.sin(az)

    def fn(x, y):
        cx, cy = center if center is not None else (_mid(x), _mid(y))
        dx = np.asarray(x, dtype=float) - cx
        dy = np.asarray(y, dtype=float) - cy
        a = dx * sx_ + dy * sy_
        c = dx * nx_ + dy * ny_
        return -amplitude * np.exp(-(a / (aspect * radius)) ** 2 - (c / radius) ** 2)

    return Structure(fn)


def ramp(dip, azimuth=0.0, center=(0.0, 0.0)):
    """Planar tilt of ``dip`` degrees, deepening along the ``azimuth`` normal.

    ``ramp(d)`` reproduces the base ``Layer(dip=d)`` plane (deepening with
    +y from y=0); any other azimuth generalizes it.
    """
    nx_, ny_ = _axes(azimuth)
    slope = np.tan(np.radians(dip))

    def fn(x, y):
        cx, cy = center
        t = (np.asarray(x, dtype=float) - cx) * nx_ + (np.asarray(y, dtype=float) - cy) * ny_
        return slope * t

    return Structure(fn)


def fault(throw, x0=None, y0=None, azimuth=None):
    """Vertical fault plane: drop one side by ``throw`` (m, positive down).

    The trace runs through ``(x0, y0)`` along ``azimuth``; the side in the
    azimuth-normal direction is the downthrown block. When ``azimuth`` is
    None it is inferred: ``x0`` alone gives a trace of constant x
    (azimuth 90), otherwise constant y (azimuth 0). Put the trace on a
    grid line (a multiple of dx/dy) for a clean vertical fault face; an
    oblique trace stair-steps, with cells the trace crosses internally
    carrying the throw as shear.
    """
    if azimuth is None:
        azimuth = 90.0 if (x0 is not None and y0 is None) else 0.0
    nx_, ny_ = _axes(azimuth)

    def fn(x, y):
        px = x0 if x0 is not None else _mid(x)
        py = y0 if y0 is not None else _mid(y)
        t = (np.asarray(x, dtype=float) - px) * nx_ + (np.asarray(y, dtype=float) - py) * ny_
        return np.where(t > 0.0, float(throw), 0.0)

    return Structure(fn)


def surface(arr, x_len, y_len):
    """Interpolate a gridded surface into a Structure.

    ``arr`` is a 2-D array shaped ``(n_x, n_y)`` in the package's ``ij``
    orientation (or a path to a whitespace-delimited text file of one),
    holding vertical shifts in meters (positive down) sampled on a regular
    grid spanning ``[0, x_len] x [0, y_len]``. Values are interpolated
    linearly and extrapolated at the edges, so any resolution works.
    """
    if isinstance(arr, (str, bytes)) or hasattr(arr, "__fspath__"):
        arr = np.loadtxt(arr)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"surface() needs a 2-D array, got shape {arr.shape}")
    xs = np.linspace(0.0, x_len, arr.shape[0])
    ys = np.linspace(0.0, y_len, arr.shape[1])
    interp = RegularGridInterpolator((xs, ys), arr, bounds_error=False, fill_value=None)

    def fn(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pts = np.stack(np.broadcast_arrays(x, y), axis=-1)
        return interp(pts)

    return Structure(fn)


def _spill_levels(depth, wall_x=None, wall_y=None):
    """Each cell's spill depth: the deepest point of the shallowest path from it to the map's edge (a priority
    flood from the edge, 4-connected). ``wall_x`` (nx-1, ny) and ``wall_y`` (nx, ny-1) give the pass level of each cell
    edge (between i and i+1, j and j+1): a path crosses it at the highest of its level so far, the next cell's depth
    and that pass level, so +inf seals the edge and -inf leaves it free (a boolean array seals its True edges). A cell
    no path reaches keeps an infinite level. A cell of infinite depth holds no rock (collapsed by a fault): no path
    crosses it, and its own level stays infinite."""
    nx, ny = depth.shape
    walls = [None if w is None else np.where(w, np.inf, -np.inf).tolist() if w.dtype == bool else w.tolist()
             for w in (wall_x, wall_y)]
    depth = depth.tolist()                              # plain lists: the loop below reads single cells
    spill = [[math.inf] * ny for _ in range(nx)]
    done = [[False] * ny for _ in range(nx)]
    heap = []
    for i in range(nx):
        for j in range(ny):
            if i in (0, nx - 1) or j in (0, ny - 1):
                spill[i][j] = depth[i][j]
                heap.append((depth[i][j], i, j))
    heapq.heapify(heap)
    while heap:
        level, i, j = heapq.heappop(heap)
        if done[i][j]:
            continue
        done[i][j] = True
        for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if not (0 <= a < nx and 0 <= b < ny) or done[a][b]:
                continue
            across_x = a != i                               # the edge lies between i and i+1, else j and j+1
            wall = walls[0] if across_x else walls[1]
            gate = -math.inf if wall is None else wall[min(a, i)][j] if across_x else wall[i][min(b, j)]
            cost = max(depth[a][b], level, gate)
            if cost < spill[a][b]:
                spill[a][b] = cost
                heapq.heappush(heap, (cost, a, b))
    return np.array(spill, dtype=float)


def closure_stats(depth, dx, dy, crest=None):
    """The trap around ``crest`` on a top-surface depth map (m, positive down, shape ``(nx, ny)``).

    Hydrocarbons fill a structural high down to its spill point: the deepest point of the
    shallowest path from the crest to the map's edge. A priority flood from the edge gives that
    depth for every cell (4-connected). Returns ``area`` (m2: the cells shallower than the spill
    depth connected to the crest), ``height`` (m: spill depth minus crest depth),
    ``spill_depth``, ``crest`` (i, j; the shallowest cell unless given) and ``mask``.
    """
    depth = np.asarray(depth, dtype=float)
    spill = _spill_levels(depth)
    if crest is None:
        crest = np.unravel_index(int(np.argmin(depth)), depth.shape)
    crest = (int(crest[0]), int(crest[1]))
    spill_depth = float(spill[crest])
    labels, _ = ndimage.label(depth < spill_depth)
    mask = (labels == labels[crest]) if labels[crest] else np.zeros(depth.shape, dtype=bool)
    return {"area": float(mask.sum() * dx * dy), "height": spill_depth - float(depth[crest]),
            "spill_depth": spill_depth, "crest": crest, "mask": mask}


def outline(trap, dx, dy, rim):
    """The columns of a simulation outline: those whose centre lies within ``rim`` m of a column of ``trap``.

    ``trap`` is the ``(nx, ny)`` bool map of a trap's columns (the ``mask`` of :func:`closure_stats`, or any other
    trap's), ``dx`` and ``dy`` the cell sizes (m) and ``rim`` (m, from centre to centre) the width kept around it: the
    trap dilated by a disc, as a company deck cuts its model at the closure plus a margin where an aquifer attaches.
    Returns an ``(nx, ny)`` bool map, the ``outline`` of :func:`resmill.export.to_grdecl`.
    """
    trap = np.asarray(trap, dtype=bool)
    if not trap.any():
        raise ValueError("outline needs a trap: no column of the map is True")
    if not rim >= 0.0:
        raise ValueError(f"rim must be zero or more, not {rim}")
    return ndimage.distance_transform_edt(~trap, sampling=(dx, dy)) <= float(rim)


def closure(area, height, aspect=1.0, azimuth=0.0, center=None, limb_ratio=1.0, tilt=0.0,
            satellites=0, warp=0.0, seed=None):
    """A four-way dip closure over ``area`` (m2) with ``height`` (m) of relief, lobate and asymmetric.

    After Wu et al. (2020, Geophysics 85(4)), culminations plus a planar tilt, with smooth
    culminations of finite footprint, (1 - rho^2)^2, instead of Gaussians so that a trap without
    tilt closes where its flanks meet the flat regional level. The main culmination is elliptical,
    ``aspect`` times longer along the ``azimuth`` axis than across it, its forelimb (the side the
    azimuth normal points to) ``limb_ratio`` times steeper than its backlimb. ``satellites``
    smaller culminations drawn from ``seed`` sit along the axis (amplitude 0.3-0.8 and size
    0.3-0.7 of the main one, 0.6-1.4 of its half-lengths from the crest), merging into it as noses
    and saddles; ``warp`` displaces the culminations' coordinates by smooth random waves
    (wavelengths of 1.8-4.2 half-widths) of that RMS amplitude, as a fraction of the half-width, so
    the outline is lobate. ``tilt`` (0 to below 1) is a regional dip deepening toward the forelimb,
    as a fraction of the backlimb's steepest slope; the spill point then lies up-dip. The shape is
    measured once on its own grid (:func:`closure_stats`) and scaled so that its trap closes over
    ``area`` with ``height`` of relief exactly; the regional tilt continues beyond it. The
    returned Structure carries ``crest_offset``, the crest's ``(dx, dy)`` (m) from ``center``: with
    a tilt the map's shallowest point may lie up-dip, not on the crest.
    """
    q = float(limb_ratio)
    w_fore, w_back = 2.0 / (1.0 + q), 2.0 * q / (1.0 + q)
    rng = np.random.default_rng(seed)
    sats = [(float(rng.choice((-1.0, 1.0)) * rng.uniform(0.6, 1.4) * aspect), float(rng.normal(0.0, 0.3)),
             float(rng.uniform(0.3, 0.8)), float(rng.uniform(0.3, 0.7))) for _ in range(int(satellites))]
    waves = [(rng.uniform(1.5, 3.5, 6), rng.uniform(0.0, 2.0 * np.pi, 6), rng.uniform(0.0, 2.0 * np.pi, 6))
             for _ in range(2)] if warp else []
    slope = float(tilt) * 8.0 / (3.0 * np.sqrt(3.0)) / w_back       # the bump's steepest slope, 8/(3 sqrt 3) / w

    def bump(rho2):
        return np.where(rho2 < 1.0, (1.0 - np.minimum(rho2, 1.0)) ** 2, 0.0)

    def uplift(u, v):
        tilted = -slope * v
        if waves:
            du, dv = (np.sqrt(2.0 / 6.0) * sum(np.cos(k * (np.cos(a) * u / aspect + np.sin(a) * v) + ph)
                                            for k, a, ph in zip(*wave)) for wave in waves)
            u, v = u + warp * aspect * du, v + warp * dv
        w = np.where(v > 0.0, w_fore, w_back)
        out = bump((u / aspect) ** 2 + (v / w) ** 2) + tilted
        for u0, v0, a0, s0 in sats:
            out = out + a0 * bump(((u - u0) / (s0 * aspect)) ** 2 + ((v - v0) / s0) ** 2)
        return out

    uu = np.linspace(-3.5 * aspect, 3.5 * aspect, 401)
    vv = np.linspace(-3.5, 3.5, 401)
    U, V = np.meshgrid(uu, vv, indexing="ij")
    unit = -uplift(U, V)
    core = np.where(bump((U / aspect) ** 2 + (V / w_back) ** 2) > 0.5, unit, np.inf)
    stats = closure_stats(unit, uu[1] - uu[0], vv[1] - vv[0],
                          crest=np.unravel_index(int(np.argmin(core)), core.shape))
    length, amplitude = np.sqrt(float(area) / stats["area"]), float(height) / stats["height"]
    nx_, ny_ = _axes(azimuth)
    az = np.radians(azimuth)
    sx_, sy_ = np.cos(az), -np.sin(az)
    uc, vc = uu[stats["crest"][0]] * length, vv[stats["crest"][1]] * length

    def fn(x, y):
        cx, cy = center if center is not None else (_mid(x), _mid(y))
        dx = np.asarray(x, dtype=float) - cx
        dy = np.asarray(y, dtype=float) - cy
        return -amplitude * uplift((dx * sx_ + dy * sy_) / length, (dx * nx_ + dy * ny_) / length)

    out = Structure(fn)
    out.crest_offset = (uc * sx_ + vc * nx_, uc * sy_ + vc * ny_)
    return out


def roughness(sd, range_m, x_len, y_len, seed=None):
    """A smooth random surface (m) with standard deviation ``sd`` over ``[0, x_len] x [0, y_len]``.

    Gaussian covariance exp(-3 (r / range_m)^2), the horizon residual of stochastic depth
    conversion (Abrahamsen 1993; COHIBA; IGEMS: SD 5-13 m, ranges 1.4-7 km). It is generated once
    by FFT on a grid of about range/8 spacing, padded by one range against wrap-around, and
    interpolated (:func:`surface`), so it is one surface wherever it is evaluated.
    """
    rng = np.random.default_rng(seed)
    step = float(range_m) / 8.0
    nxs, nys = int(np.ceil(x_len / step)) + 1, int(np.ceil(y_len / step)) + 1
    sx, sy = x_len / (nxs - 1), y_len / (nys - 1)
    px, py = int(np.ceil(range_m / sx)), int(np.ceil(range_m / sy))
    shape = (nxs + 2 * px, nys + 2 * py)
    kx = 2.0 * np.pi * np.fft.fftfreq(shape[0], sx)
    ky = 2.0 * np.pi * np.fft.fftfreq(shape[1], sy)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    field = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal(shape))
                                 * np.exp(-k2 * float(range_m) ** 2 / 24.0)))[px:px + nxs, py:py + nys]
    return surface(float(sd) * (field - field.mean()) / field.std(), x_len, y_len)


def isochore(cv, trend_share, range_m, x_len, y_len, azimuth=0.0, seed=None):
    """A zone-thickness factor field with mean 1 over ``[0, x_len] x [0, y_len]``: 1 + a l + e.

    ``l`` is a unit-SD planar ramp over the footprint, thickening along ``azimuth``
    (the package convention: its normal points that way), with a = cv sqrt(trend_share);
    ``e`` is correlated noise (:func:`roughness`, Gaussian covariance of range ``range_m``)
    of SD cv sqrt(1 - trend_share). So its coefficient of variation is ``cv``, of which the
    planar trend explains ``trend_share`` (Norne and Volve isochores: CV 0.10-0.55, trend
    0.05-0.95). Pass it to ``to_grdecl(isochore=[...])``; values below 0 pinch the zone out.
    """
    share = float(trend_share)
    nx_, ny_ = _axes(azimuth)
    corners = np.array([[0.0, 0.0], [x_len, 0.0], [0.0, y_len], [x_len, y_len]])
    t = corners @ np.array([nx_, ny_])
    t_mid, t_sd = 0.5 * (t.min() + t.max()), (t.max() - t.min()) / np.sqrt(12.0)
    noise = roughness(float(cv) * np.sqrt(max(1.0 - share, 0.0)), range_m, x_len, y_len, seed=seed)
    a = float(cv) * np.sqrt(share)

    def fn(x, y):
        ramp = ((np.asarray(x, dtype=float) * nx_ + np.asarray(y, dtype=float) * ny_) - t_mid) / t_sd
        return 1.0 + a * ramp + noise(x, y)

    return Structure(fn)


def growth(fault, expansion, width, depth):
    """A zone's thickness factor for a zone laid down at ``depth`` (m) while ``fault`` was moving (growth strata).

    1 in the footwall, ``expansion`` (the growth or expansion index: downthrown over upthrown thickness; Ewing et al. 1986,
    Xiao & Suppe 1992: 1.1-2.5 per fault) in the hanging wall. The factor rises as a smooth step from 1 at the fault's trace
    at ``depth`` (the plane's footwall cutoff there) to ``expansion`` ``width`` m toward the hanging wall (``None``: the
    horizon's heave at the fault's centre line, so that the hanging wall begins at the full factor), and its excess
    is tapered along the strike by the throw profile (the fault's tip ellipse at ``depth``, relative to its centre line):
    no growth where the fault has no throw at that depth. A listric fault (:attr:`resmill.faults.Fault.flatten`) moves its whole
    hanging wall by one heave, that of the horizon at ``z_center``, so its step spans that heave at every depth and its taper
    does not depend on the depth. The step is the model: the whole hanging wall carries the full factor, with no decay length
    and no wedge thinning away from the fault (T18 gives the index, not a wedge's shape). Pass it to ``to_grdecl(isochore=[...])``;
    it needs ``fault.z_center``, the depth scale of the tip ellipse.
    """
    if fault.z_center is None:
        raise ValueError("growth needs fault.z_center, the depth scale of the fault's tip ellipse")
    if not (width is None or width > 0.0) or not expansion > 0.0:
        raise ValueError(f"growth needs a width and an expansion above 0, got {width!r} and {expansion!r}")
    zc = float(fault.z_center)
    trace = _plane(fault, zc)[1]
    h0 = float(trace(depth))
    lx = 0.5 * fault.length
    listric = fault.flatten is not None                 # its hanging wall has one heave and no taper with depth (faults.py)
    rz = 0.0 if listric else (depth - zc) / np.sin(np.radians(fault.dip)) / (lx / fault.aspect)
    centre = float(ww_profile(abs(rz)))
    if width is None:
        ref = zc if listric else depth
        width = max(float(trace(ref + fault.throw * centre) - trace(ref)), 1.0)

    def fn(x, y):
        s, h = _frame(fault, np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        t = np.clip((h - h0) / width, 0.0, 1.0)
        taper = ww_profile(np.sqrt((s / lx) ** 2 + rz ** 2)) / centre if centre > 0.0 else 0.0
        return 1.0 + (expansion - 1.0) * t * t * (3.0 - 2.0 * t) * taper

    return Structure(fn)
