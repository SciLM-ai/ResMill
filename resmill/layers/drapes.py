"""Mud drapes at the bases of channel storeys, as transmissibility multipliers on cell faces.

A channel element often lies on a thin mud drape (a few centimetres to a few metres of thin-bedded mudstone) that covers
part of its base: a median 60 % of it, from under 5 % to over 90 % over 17 outcrops and 154 elements (Barton et al.
2010). Where it covers the base, the element does not touch the older sand beneath it, and in Barton's sector model
recovery falls from 0.60 without drapes to 0.53 at 60 % coverage and 0.20 at 90 % (Ruetten 2021 finds the same fall at
breakthrough).

A drape is far thinner than a cell (0.1-1.5 m in cells of 0.5-5 m), so painting mud cells would remove up to fifty times
the sand the drape displaces and could not draw one thinner than a cell at all. It is put where it is, on the faces: the
transmissibility between a channel-fill cell and the older-storey sand next to it is multiplied by the thin-barrier
factor of a mud layer of thickness ``t`` and permeability ``kd`` replacing ``t`` of the rock between the two cells, each
of size ``h`` and permeability ``k1`` and ``k2`` across the face. In series, with ``ks`` the harmonic mean of ``k1`` and
``k2``::

    M = [1 + (t / h) (ks / kd - 1)]^-1        (t at most h; kept between 1e-12 and 1)

Facies, porosity and permeability stay as they were; the writers turn the multipliers into MULTX, MULTY and MULTZ. A
mud drape (``kd`` 4e-6 to 1e-3 mD) seals its face against sand of 400 mD (M below 2e-4, for any thickness from 0.1 m in
cells up to 5 m); only silty drapes of 0.2-8 mD leave a leak that depends on thickness and cell size. The sector models
of the literature do the same (Li 2008; Ruetten 2021) and Alpak & van der Vlugt (2014) lay the drape on the whole
surface and cut holes in it.

Where, per storey (a level of the layer's aggradation):

* the drape lies on the storey's fill footprint (the columns holding channel fill or lateral accretion of that level),
  on a share of columns equal to ``coverage`` (the mean over the base, Barton's measure), chosen by the ranks of a
  smooth random field of practical range ``hole_range_m``: the columns left bare are the holes;
* with ``coverage_concentration`` k each storey's own share is drawn from Beta(c k, (1 - c) k), mean c = ``coverage``,
  clipped to [0, 1]: a storey drawn at 1 seals its whole base. Without it every storey gets exactly c, and while the holes
  are at least a cell wide no interface is ever sealed, though a stack is as tight as its tightest interface. Barton et al.
  (2010) found drapes "continuous in approximately 25 % of the channel elements" and discontinuous in the rest; at their
  mean coverage of 0.5565, P(coverage >= 0.95) = 0.25 is k = 0.71 (reading "continuous" as 95 % or more of the base is
  [J], and so is treating a storey, a belt of 4-15 flow events, as one element). The spread is the sampler's choice,
  not a fact fixed here: ``None`` (default) draws nothing and gives today's maps;
* with ``margin_bias`` b the share covered is b higher at the margin than at the axis, where the axis-ness of a
  column is the engine's ``depth_norm`` of the lowest fill cell (1 at the thalweg, 0 at the banks), the mean held at
  ``coverage``;
* a face is draped when a fill cell meets older-storey sand across it and its column is draped in its own storey's map.
  A drape thus survives only where the older storey's sand still lies under the younger element, and a later storey
  that cut through an older base takes its drape away with the cells. A lateral face is the element's wall, a vertical
  one its base; levees and splays make no drape.

The maps come from a random stream of their own (seeded from the layer's seed), so drapes change no sand and no other
random number; without ``drapes`` nothing here runs.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import betainc, betaincinv

__all__ = ["COVERAGE_BETA", "COVERAGE_RANGE", "HOLE_RANGE_WIDTHS", "MARGIN_BIAS_MAX", "THICKNESS_RANGE_M", "check_drapes",
           "coverage_from_unit", "sample_drapes", "storey_map", "draped_columns", "drape_faces", "place_drapes"]

# Coverage over reservoirs: the Beta with the mean and standard deviation of Barton et al.'s 17 outcrop means (0.556 and
# 0.257), cut to the span of those outcrops (the truncation moves them to 0.53 and 0.23).
COVERAGE_BETA = (1.52, 1.21)
COVERAGE_RANGE = (0.05, 0.92)
# Margin minus axis: none (Barton: "apparently random") to Vento (2020)'s 0.33 against 0.67; uniform in between [J].
MARGIN_BIAS_MAX = 0.34
# A drape's thickness (m): an element's own drape under 0.5 m, a story set's over 1 m (Barton et al. 2010); the
# log-uniform shape between is a judgement.
THICKNESS_RANGE_M = (0.1, 1.5)
# The holes' practical range in channel widths (mCHdepth x mCHwdratio), log-uniform per reservoir: the owner's G-S11 [J], so
# that the dataset spans mild drapes (small holes, little open area) to strong ones (large windows, more open area).
HOLE_RANGE_WIDTHS = (0.25, 4.0)
MULTIPLIER_FLOOR = 1e-12            # the floor ``fault_seal.face_multipliers`` keeps too
_DEFAULTS = {"margin_bias": 0.0, "thickness": 0.5, "perm": None, "hole_range_m": None, "hole_range_widths": None,
             "coverage_concentration": None}
_STREAM = 0x44524150                # tags the drape stream: ``default_rng([seed, tag])`` is not the engine's ``seed``


def check_drapes(drapes: dict) -> dict:
    """``drapes`` with the defaults filled in; ``ValueError`` for a name that is not a setting, a missing ``coverage``
    or a value out of range (a typo would otherwise silently change the geology)."""
    if not isinstance(drapes, dict) or "coverage" not in drapes or set(drapes) - set(_DEFAULTS) - {"coverage"}:
        raise ValueError(f"drapes is a dict with 'coverage' and optionally {sorted(_DEFAULTS)}; got {drapes!r}")
    s = {**_DEFAULTS, **drapes}
    bad = [name for name, ok in (
        ("coverage", 0.0 <= s["coverage"] <= 1.0), ("margin_bias", -1.0 <= s["margin_bias"] <= 1.0),
        ("thickness", s["thickness"] > 0.0), ("perm", s["perm"] is None or s["perm"] > 0.0),
        ("hole_range_m", s["hole_range_m"] is None or s["hole_range_m"] > 0.0),
        ("hole_range_widths", s["hole_range_widths"] is None or s["hole_range_widths"] > 0.0),
        ("coverage_concentration", s["coverage_concentration"] is None or 0.0 < s["coverage_concentration"] < math.inf)
    ) if not ok]
    if bad:
        raise ValueError(f"drapes: {', '.join(bad)} out of range in {drapes!r}")
    if s["hole_range_m"] is not None and s["hole_range_widths"] is not None:
        raise ValueError(f"drapes: give the holes' range as hole_range_m or as hole_range_widths, not both: {drapes!r}")
    return s


def coverage_from_unit(u):
    """Coverage at the quantile ``u`` (a number or an array in [0, 1]) of :data:`COVERAGE_BETA` truncated to
    :data:`COVERAGE_RANGE`, by the inverse CDF (never clipped, so nothing piles up on a bound)."""
    a, b = COVERAGE_BETA
    lo, hi = (betainc(a, b, x) for x in COVERAGE_RANGE)
    return betaincinv(a, b, lo + np.asarray(u) * (hi - lo))


def _log_uniform(u, bounds):
    """The value at the quantile ``u`` of a log-uniform distribution over ``bounds``."""
    lo, hi = bounds
    return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))


def sample_drapes(rng, hole_widths=None) -> dict:
    """One reservoir's ``drapes`` settings: ``coverage`` as above, ``margin_bias`` uniform from 0 to
    :data:`MARGIN_BIAS_MAX`, ``thickness`` log-uniform over :data:`THICKNESS_RANGE_M`. Draws three numbers from the
    ``numpy.random.Generator`` ``rng``; ``perm`` and the holes' range keep their defaults, unless ``hole_widths`` is a
    ``(low, high)`` pair (:data:`HOLE_RANGE_WIDTHS`): the dict then also has ``hole_range_widths``, log-uniform over that
    range in channel widths, from a fourth number."""
    u = rng.random(3 if hole_widths is None else 4)
    out = {"coverage": float(coverage_from_unit(u[0])), "margin_bias": float(MARGIN_BIAS_MAX * u[1]),
           "thickness": _log_uniform(u[2], THICKNESS_RANGE_M)}
    if hole_widths is not None:
        out["hole_range_widths"] = _log_uniform(u[3], hole_widths)
    return out


def storey_map(mask, poro_mult_field, log_perm_offset_field, event_group) -> np.ndarray:
    """The storey of every cell of ``mask``: the level ``event_group`` gives for the cell's flow event, which the engine
    stamps into the two fields as a pair (``fluvial.event_levels``). -1 outside the mask, for an event the map does
    not know, and for every cell when ``event_group`` is empty. The map is looked up once per distinct event, not per
    cell."""
    storey = np.full(mask.shape, -1, dtype=np.int64)
    if event_group and mask.any():
        pairs = np.stack([poro_mult_field[mask], log_perm_offset_field[mask]], axis=1)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        level = np.array([event_group.get((p, q), -1) for p, q in unique], dtype=np.int64)
        storey[mask] = level[inverse.ravel()]
    return storey


def _pair(a, axis):
    """``a`` and its upper neighbour along ``axis``: two views, one cell shorter than ``a`` along it."""
    a = np.moveaxis(a, axis, 0)
    return a[:-1], a[1:]


def _local_coverage(axis_ness, coverage, margin_bias):
    """Share of each column's base to cover: ``coverage`` on average, ``margin_bias`` more at the margin (axis-ness 0)
    than at the axis (1), clipped to [0, 1] with the offset found by bisection so the mean stays ``coverage``."""
    shape = margin_bias * (axis_ness.mean() - axis_ness)
    lo, hi = -1.0 - shape.max(), 2.0 - shape.min()
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.clip(mid + shape, 0.0, 1.0).mean() < coverage else (lo, mid)
    return np.clip(0.5 * (lo + hi) + shape, 0.0, 1.0)


def _storey_coverage(coverage, concentration, rng):
    """One storey's coverage: a draw from Beta(c k, (1 - c) k), mean ``coverage`` = c and concentration k, clipped to
    [0, 1]. Coverage 0 or 1 has no spread (and a Beta parameter of 0), so it is returned as it is."""
    if not 0.0 < coverage < 1.0:
        return coverage
    return float(np.clip(rng.beta(coverage * concentration, (1.0 - coverage) * concentration), 0.0, 1.0))


def draped_columns(fill, storey, depth_norm, coverage, margin_bias, sigma, rng, concentration=None) -> np.ndarray:
    """Which columns of each storey's base are draped: bool ``(n_storeys, nx, ny)``, ``n_storeys`` the top level + 1 (at
    least 1, an undraped level 0, when no storey is known).

    ``fill`` marks the channel-fill cells, ``storey`` their levels, ``depth_norm`` the engine's position in the
    channel (its lowest fill cell in a column gives the axis-ness) and ``sigma`` the smoothing (cells along x and y) of
    the white noise behind the holes, one field per storey drawn from ``rng``. A storey's fill footprint is covered on
    exactly the share ``coverage`` (to one column) when ``margin_bias`` is 0: the columns whose rank in the field is
    below it. With a ``concentration`` each storey's share is a Beta draw from ``rng`` (:func:`_storey_coverage`), after
    that storey's field, instead; None draws nothing more.
    """
    nx, ny, _ = fill.shape
    draped = np.zeros((max(int(storey.max()), 0) + 1, nx, ny), dtype=bool)
    for level in np.unique(storey[fill]):
        if level < 0:
            continue
        mine = fill & (storey == level)
        base = mine.any(axis=2)
        lowest = np.take_along_axis(depth_norm, mine.argmax(axis=2)[..., None], axis=2)[..., 0][base]
        noise = gaussian_filter(rng.standard_normal((nx, ny)), sigma=sigma, mode="reflect")[base]
        rank = (np.argsort(np.argsort(noise)) + 0.5) / noise.size
        share = coverage if concentration is None else _storey_coverage(coverage, concentration, rng)
        draped[level][base] = rank < _local_coverage(lowest, share, margin_bias)
    return draped


def drape_faces(facies, storey, draped, perms, size, thickness, mud_perm):
    """``(mult_x, mult_y, mult_z)``, each ``(nx, ny, nz)`` float32 with k up: the multiplier across each cell's +x,
    +y and lower face (k - 1), 1 where there is no drape. A face is draped when its two cells are sand of different
    known storeys, the younger is channel fill (CH or LA) and its column is draped in ``draped[storey]``; for a z face
    the younger cell must lie above (the face is the base of its element), for a lateral face either side may (its
    wall).

    ``perms`` are the cells' permeabilities across x, y and z faces (mD), ``size`` the cell sizes (m), ``thickness``
    the drape's (m, capped at the cell size) and ``mud_perm`` its permeability (mD); the multiplier is the thin barrier
    of the module docstring between the two cells' harmonic mean, kept between 1e-12 and 1.
    """
    known = (facies >= 1) & (storey >= 0)
    i, j = np.indices(facies.shape[:2], sparse=True)
    held = known & ((facies == 3) | (facies == 4)) & draped[storey, i[:, :, None], j[:, :, None]]
    out = []
    for axis, (perm, h) in enumerate(zip(perms, size)):
        known_lo, known_hi = _pair(known, axis)
        s_lo, s_hi = _pair(storey, axis)
        held_lo, held_hi = _pair(held, axis)
        flag = known_lo & known_hi & (s_hi > s_lo) & held_hi
        if axis != 2:
            flag |= known_lo & known_hi & (s_lo > s_hi) & held_lo
        k_lo, k_hi = (k[flag].astype(float) for k in _pair(perm, axis))
        ks = 2.0 * k_lo * k_hi / (k_lo + k_hi)
        m = 1.0 / (1.0 + (min(float(thickness), h) / h) * (ks / float(mud_perm) - 1.0))
        mult = np.ones(facies.shape, dtype=np.float32)
        view = np.moveaxis(mult, axis, 0)
        (view[1:] if axis == 2 else view[:-1])[flag] = np.clip(m, MULTIPLIER_FLOOR, 1.0)
        out.append(mult)
    return tuple(out)


def place_drapes(facies, storey, depth_norm, perms, size, drapes, mud_perm, seed=None):
    """The multipliers ``(mult_x, mult_y, mult_z)`` of ``drapes`` (see :func:`check_drapes`; ``hole_range_m`` is the
    practical range in metres and must be given) on a facies cube with its storeys, :func:`drape_faces` of
    :func:`draped_columns`. ``mud_perm`` is the permeability of a drape without ``perm``; ``seed`` seeds the drape's own
    random stream (None: fresh entropy)."""
    from .channel import _RANGE_PER_SIGMA          # the range convention of noise_range_m

    s = check_drapes(drapes)
    if s["hole_range_m"] is None:
        raise ValueError("drapes needs hole_range_m here (ChannelLayer.create_geology defaults it to half a channel)")
    rng = np.random.default_rng(None if seed is None else [int(seed), _STREAM])
    sigma = tuple(s["hole_range_m"] / _RANGE_PER_SIGMA / d for d in size[:2])
    draped = draped_columns((facies == 3) | (facies == 4), storey, depth_norm, s["coverage"], s["margin_bias"], sigma,
                            rng, s["coverage_concentration"])
    return drape_faces(facies, storey, draped, perms, size, s["thickness"],
                       mud_perm if s["perm"] is None else s["perm"])
