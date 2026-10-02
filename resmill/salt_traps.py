"""Salt traps: a three-way trap against a salt flank and a trap beneath a salt sheet, composed from :mod:`resmill.salt`.

As :func:`resmill.fold_traps.fold_trap` does for the fold styles, each composer draws nothing: every value, seeds included,
is an argument, so the model is a pure function of them, and it returns ``{"kwargs": ..., "meta": ...}``, the keywords
of :func:`resmill.export.to_grdecl` and what was built (decision S15; rows N1-N39 of ``step5_salt.md``). A sampler draws the
arguments; this module only places the pieces: where the contact passes the closure, how the upturn's folding zone and
unconformity are cut from the intent (a share of the reservoir lost at the wall, or the angle of truncation), at which
depth the contact stands, and how the base of a salt sheet meets the trap beneath it.
"""
import math

import numpy as np

from . import structure as st
from .fault_patterns import fold_faults
from .fold_traps import crest_cell, trap_outline
from .salt import (MAX_DIP, base_of_salt, salt_body, salt_sequence, salt_thinning, salt_upturn, sequence_dip,
                   truncation_cut)

KINDS = ("truncated", "cover")
"""The subsalt traps: cut by the base of the salt (three-way), or a fold beneath a cover (four-way)."""


def _centres(x_len, y_len, dx):
    """``(sx, sy, X, Y)``: the columns' sizes (m) of a map of whole cells of ``dx`` m and their centres."""
    nx, ny = int(round(x_len / dx)), int(round(y_len / dx))
    sx, sy = x_len / nx, y_len / ny
    return (sx, sy, *np.meshgrid((np.arange(nx) + 0.5) * sx, (np.arange(ny) + 0.5) * sy, indexing="ij"))


def _fold(x_len, y_len, closure, roughness):
    """``(center, fold, crest)``: the closure about the map's middle unless it says otherwise, and its crest (x, y) (a tilt
    moves it off the centre); ``fold`` is the closure alone, the structure of its faults."""
    center = closure.get("center", (0.5 * x_len, 0.5 * y_len))
    fold = st.closure(**{**closure, "center": center})
    return center, fold, np.asarray(center, dtype=float) + np.asarray(fold.crest_offset)


def _at(field, x, y):
    """The value of a Structure at one point."""
    return float(np.ravel(field(np.array([x]), np.array([y])))[0])


def _reach(mask, crest, u, sx, sy):
    """Distance (m) from ``crest`` (x, y) along the unit vector ``u`` to the edge of the closure's ``mask`` ((nx, ny): the
    columns of the trap), in steps of half a column."""
    nx, ny = mask.shape
    for s in np.arange(0.0, math.hypot(nx * sx, ny * sy), 0.5 * min(sx, sy)):
        i, j = (int(v // c) for v, c in zip(crest + s * u, (sx, sy)))
        if not (0 <= i < nx and 0 <= j < ny) or not mask[i, j]:
            return float(s)
    return math.hypot(nx * sx, ny * sy)


def _upturn(width, dip, thinning, relief_cap, taper, angle, loss, thickness):
    """What the folding zone is, checked: ``(sequence, power, relief, cut)``. A plain upturn (``taper`` None) meets the contact
    at ``dip`` with the exponent that holds its relief ``width tan(dip) / power`` under ``relief_cap``. A halokinetic sequence
    (``taper`` given: hooks and wedges) has dip ``sequence_dip(taper)`` and exponent 2, and the unconformity that cuts the lifted
    beds is given as ``angle`` (the truncation angle at the contact, degrees: a hook that pinches out is cut at over 70) or
    ``loss`` (the share of the reservoir's thickness it removes at the contact: a hook that keeps the interval to the wall, a
    wedge); the cut is the share of the lift it removes (``truncation_cut``, or ``loss x thickness / relief`` up to 1)."""
    if not width > 0.0:
        raise ValueError(f"width (the folding zone, m) must be positive, got {width}")
    if taper is None:
        if dip is None or angle is not None or loss is not None:
            raise ValueError("a plain upturn needs its dip and takes neither angle nor loss (those cut a sequence, "
                             "which has a taper)")
        power = max(2.0, width * math.tan(math.radians(min(dip, MAX_DIP))) / relief_cap)
        return False, power, width * math.tan(math.radians(min(dip, MAX_DIP))) / power, None
    if dip is not None or thinning or (angle is None) == (loss is None):
        raise ValueError("a sequence (taper) has no dip or thinning of its own (the taper sets the dip, the unconformity the "
                         "thinning) and needs exactly one of angle and loss")
    relief = width * math.tan(math.radians(sequence_dip(taper))) / 2.0
    cut = truncation_cut(taper, angle) if angle is not None else min(1.0, loss * thickness / relief)
    return True, 2.0, relief, float(cut)


def salt_flank_trap(x_len, y_len, dx, top, thickness, closure, body, *, normal, wall_at, width, dip=None, thinning=0.0,
                    relief_cap=350.0, taper=None, angle=None, loss=None, neck=0.6, waist=50.0, roughness=None,
                    faults=None, n_layers=1, rim=None):
    """A three-way trap against a salt flank (decision S15): ``{"kwargs": ..., "meta": ...}`` with the keywords of
    :func:`resmill.export.to_grdecl` (``structure``, ``salt``, ``faults`` and either ``isochore`` or ``erode_above``, and
    ``outline`` with a ``rim``) and what was built.

    ``x_len`` and ``y_len`` (m, whole cells of ``dx``) are the model, ``top`` the depth (m) of the reservoir's top at the datum
    and ``thickness`` (m) that of the whole stack, ``n_layers`` the number of its layers (zones). ``closure`` (the keywords of
    :func:`resmill.structure.closure`, centred on the model's middle unless it says so) is the trap whose crest the salt
    cuts off, ``roughness`` (the keywords of :func:`resmill.structure.roughness` without the model's size, None: a smooth fold)
    its horizon error and ``faults`` (the keywords of :func:`resmill.fault_patterns.fold_faults` without the model, style,
    ``top`` and ``thickness``: ``density`` and ``seed`` are required, ``radial_rate`` is the radial faults per km of contact,
    None: no faults) the ``salt_flank`` set, drawn on the fold without its roughness.

    The contact passes ``wall_at`` (0 to 1: 0.2-0.6) of the closure's half-width from the crest, on the side away from the salt,
    so that the crest is cut off into the salt and the trap that is left lies against the wall: ``normal`` (degrees, 0 = +x,
    90 = +y) is the direction from the crest toward the salt and the half-width is that of the closure's trap in the opposite
    direction, measured on the column centres. ``body`` holds the keywords of :func:`resmill.salt.salt_body` except its
    ``center`` and ``z_ref``: ``axes`` (the semi-axes along and across ``azimuth``; a stock has equal ones, a wall a long
    one) and, as wanted, ``azimuth`` (None: the long axis runs across ``normal``), ``flare``, ``lobes``, ``rough``, ``hurst``,
    ``overhang`` (lateral extent L and height H of its underside), ``shape`` and ``seed``. The body is centred so that its
    outline, at its reference depth, passes through the contact point (its polar radius toward the salt from the crest side).
    The reference depth (the overhang's neck) is, with an ``overhang``, ``neck`` (0.3-0.9) of its height H below the lifted
    beds' top at the contact, so the beds run into the underside, and without one ``waist`` m below the middle of the reservoir
    there.

    The strata are upturned over a folding zone ``width`` m wide (cells at most half of it: the exporter checks), plain or as a
    halokinetic sequence (:func:`_upturn`): plain, at ``dip`` degrees with the exponent that keeps the relief under
    ``relief_cap`` m and ``thinning`` (0 to 1) of their thickness lost at the contact (an isochore per layer); a sequence, by
    ``taper`` degrees (the angle of the line from the fold's inflection point to its tip: hooks 40-80, wedges 8-49) and the
    unconformity that truncates the lifted beds (``erode_above``), given as the truncation ``angle`` or the ``loss``. A hook
    cut at over 70 degrees pinches the reservoir out within its zone and needs a taper above 54 degrees; one with a ``loss``
    keeps the interval to the wall.

    ``meta`` is plain numbers: the placement (``normal``, ``wall_at``, ``z_ref``, the ``contact`` point (x, y) in m, the
    closure as drawn ``closure_area_km2`` and ``closure_height_m`` and ``crest``, the crest of the trap the salt leaves,
    (x, y) in m), the upturn (``upturn`` ``"plain"`` or ``"sequence"``, ``width``, ``dip`` at the contact, ``power``,
    ``relief_m``, ``thinning``, and for a sequence its ``cut`` and ``truncation_angle_deg``), ``overhang`` (L, H) or None,
    ``n_faults``, ``fault_sets`` and with a ``rim`` its value and ``outline_columns``. The trap as the grid holds it is read
    with :func:`resmill.fault_seal.blocks_at` from ``to_grdecl(report=)``."""
    if not 0.0 < wall_at < 1.0:
        raise ValueError(f"wall_at must lie between 0 and 1, got {wall_at}")
    if len(body["axes"]) != 2:
        raise ValueError(f"body['axes'] are the two semi-axes, got {body['axes']}")
    sequence, power, relief, cut = _upturn(width, dip, thinning, relief_cap, taper, angle, loss, thickness)
    center, fold, crest = _fold(x_len, y_len, closure, roughness)
    sx, sy, X, Y = _centres(x_len, y_len, dx)
    stats = st.closure_stats(top + fold(X, Y), sx, sy, crest=(int(crest[0] // sx), int(crest[1] // sy)))
    a = math.radians(normal)
    u = np.array([math.cos(a), math.sin(a)])
    contact = crest - wall_at * _reach(stats["mask"], crest, -u, sx, sy) * u         # beyond the crest: it is cut off
    axes = tuple(float(v) for v in body["axes"])
    azimuth = body.get("azimuth")
    if azimuth is None:                                           # the long axis (cos az, -sin az) across the normal
        azimuth = math.degrees(math.atan2(-u[0], -u[1]))
    t_axis = np.array([math.cos(math.radians(azimuth)), -math.sin(math.radians(azimuth))])
    n_axis = np.array([math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth))])
    shape = float(body.get("shape", 2.0))
    polar = (abs(u @ t_axis) ** shape / axes[0] ** shape + abs(u @ n_axis) ** shape / axes[1] ** shape) ** (-1.0 / shape)
    z_top = top + _at(fold, *contact)
    lifted = z_top - (relief if cut is None else relief * (1.0 - cut))               # the beds' top at the contact
    overhang = body.get("overhang")
    z_ref = lifted + neck * overhang[1] if overhang else z_top + 0.5 * thickness + waist
    options = {k: v for k, v in body.items() if k not in ("axes", "azimuth")}
    salt = salt_body(tuple(contact + polar * u), axes, azimuth=azimuth, z_ref=z_ref, **options)
    rough = None if roughness is None else st.roughness(x_len=x_len, y_len=y_len, **roughness)
    surface = fold if rough is None else fold + rough
    if sequence:
        built = salt_sequence(salt, width, taper, cut, datum=top + surface)
        up, kwargs = built.upturn, dict(erode_above=built.truncation)
        dip, angle = built.dip, built.angle
    else:
        up = salt_upturn(salt, dip, width, power=power)
        kwargs = dict(isochore=[salt_thinning(salt, thinning, width, power=power)] * n_layers)
        angle = None
    drawn = [] if faults is None else fold_faults("salt_flank", fold, x_len, y_len, dx, top=top, thickness=thickness,
                                                  salt=salt, upturn=up, **faults)
    kwargs.update(structure=surface + up, salt=salt, faults=drawn)
    left = stats["mask"] & ~salt.inside(X, Y, top + fold(X, Y) + 0.5 * thickness)       # the trap the salt leaves
    where = np.unravel_index(int(np.argmin(np.where(left, top + fold(X, Y), np.inf))), left.shape) if left.any() \
        else stats["crest"]
    meta = dict(normal=float(normal), wall_at=float(wall_at), z_ref=float(z_ref),
                contact=(float(contact[0]), float(contact[1])),
                closure_area_km2=stats["area"] / 1e6, closure_height_m=float(stats["height"]),
                crest=(float((where[0] + 0.5) * sx), float((where[1] + 0.5) * sy)),
                upturn="sequence" if sequence else "plain", width=float(width),
                dip=float(dip), power=float(power), relief_m=float(relief),
                thinning=0.0 if sequence else float(thinning), cut=cut, truncation_angle_deg=angle,
                overhang=None if overhang is None else tuple(float(v) for v in overhang),
                n_faults=len(drawn), fault_sets=sorted({fault.kind for fault in drawn}))
    if rim is not None:
        kwargs["outline"] = trap_outline(fold, surface, closure, center, x_len, y_len, dx, top, tuple(crest), rim)
        meta.update(rim=float(rim), outline_columns=int(kwargs["outline"].sum()))
    return dict(kwargs=kwargs, meta=meta)


def subsalt_trap(x_len, y_len, dx, top, thickness, closure, *, kind, base_dip, base_azimuth, rugosity_sd, rugosity_range,
                 sheet, seed, cut=None, margin=None, roughness=None, faults=None, rim=None):
    """A subsalt trap (decision S15): a fold beneath a salt sheet whose base is :func:`resmill.salt.base_of_salt`, as
    ``{"kwargs": ..., "meta": ...}`` with the keywords of :func:`resmill.export.to_grdecl` (``structure``, ``faults``,
    ``erode_above`` and ``outline`` with a ``rim``) and what was built.

    ``x_len``, ``y_len``, ``dx``, ``top``, ``thickness``, ``closure``, ``roughness`` and ``faults`` are those of
    :func:`salt_flank_trap` (the faults are the faulted anticline's). ``kind`` ``"truncated"``: the base cuts the crest
    by ``cut`` (0.2-0.8) of the trap's relief, a three-way trap against the salt (Tahiti, Heidelberg); ``"cover"``: it lies
    ``margin`` m above the crest and cuts nothing there (a four-way fold under a cover: Mad Dog, Atlantis; the salt is only
    a label, and a dipping base can still cut the flanks far from the crest). The trap is the one at the fold's crest on the
    fold and its roughness (:func:`resmill.fold_traps.crest_cell`), before the faults (they only divide it). The base passes
    through its crest column at the depth that gives, dips ``base_dip`` degrees toward ``base_azimuth``
    (:func:`resmill.structure.ramp`'s convention) and is rugose (``rugosity_sd`` m over ``rugosity_range`` m, drawn from
    ``seed``, zero at the crest column); ``sheet`` (m) is the sheet's thickness, a label (:func:`resmill.salt.salt_thickness`).

    ``meta`` is plain numbers: ``kind``, ``cut`` and ``margin`` as given, ``base_dip``, ``base_azimuth``, ``rugosity_sd``,
    ``sheet_m``, ``depth_base_at_crest``, the trap's ``closure_area_km2``, ``closure_height_m``, ``crest_depth_m`` and ``crest``
    ((x, y) in m), ``n_faults``, ``fault_sets`` and with a ``rim`` its value and ``outline_columns``."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if kind == "truncated" and cut is None:
        raise ValueError("a truncated trap needs the share of its relief the base cuts (cut)")
    if kind == "cover" and margin is None:
        raise ValueError("a cover needs the margin (m) of its base above the crest")
    center, fold, crest = _fold(x_len, y_len, closure, roughness)
    sx, sy, X, Y = _centres(x_len, y_len, dx)
    rough = None if roughness is None else st.roughness(x_len=x_len, y_len=y_len, **roughness)
    surface = fold if rough is None else fold + rough
    depth = top + np.asarray(surface(X, Y), dtype=float)
    cell = crest_cell(depth, crest, sx, sy)
    trap = st.closure_stats(depth, sx, sy, crest=cell)
    crest_depth, height = float(depth[cell]), float(trap["height"])
    at = ((cell[0] + 0.5) * sx, (cell[1] + 0.5) * sy)
    depth_base = crest_depth + cut * height if kind == "truncated" else crest_depth - margin
    base_rough = st.roughness(rugosity_sd, rugosity_range, x_len, y_len, seed=seed)
    plane = depth_base - _at(base_rough, *at)                                  # the rugosity is zero at the crest column
    base = base_of_salt(plane, dip=base_dip, azimuth=base_azimuth, center=at, rough=base_rough, thickness=sheet)
    drawn = [] if faults is None else fold_faults("faulted_anticline", fold, x_len, y_len, dx, top=top, thickness=thickness,
                                                  **faults)
    kwargs = dict(structure=surface, faults=drawn, erode_above=base)
    meta = dict(kind=kind, cut=cut, margin=margin, base_dip=float(base_dip), base_azimuth=float(base_azimuth),
                rugosity_sd=float(rugosity_sd), sheet_m=float(sheet), depth_base_at_crest=float(depth_base),
                closure_area_km2=trap["area"] / 1e6, closure_height_m=height, crest_depth_m=crest_depth,
                crest=(float(at[0]), float(at[1])),
                n_faults=len(drawn), fault_sets=sorted({fault.kind for fault in drawn}))
    if rim is not None:
        kwargs["outline"] = trap_outline(fold, surface, closure, center, x_len, y_len, dx, top, tuple(crest), rim)
        meta.update(rim=float(rim), outline_columns=int(kwargs["outline"].sum()))
    return dict(kwargs=kwargs, meta=meta)
