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

import numpy as np
from scipy.interpolate import RegularGridInterpolator


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
