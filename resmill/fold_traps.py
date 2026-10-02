"""Fold traps: a closure, its roughness and its faults, composed from ResMill's own pieces.

The six fold styles (four-way, turtle, faulted anticline, fold belt, three-way fault-bounded and low relief) are one
construction. A closure (:func:`resmill.structure.closure`) is the fold, a rough surface
(:func:`resmill.structure.roughness`) the horizon error of a depth map added to it, and the style's faults
(:func:`resmill.fault_patterns.fold_faults`) are drawn on the fold without that roughness (a rough surface has many
local highs and is slow to search). What differs between the styles is what the caller draws for them: a fold belt
a steeper forelimb, a three-way trap a tilt, a low-relief trap a few metres of relief, and the style's own fault sets
inside ``fold_faults``. :func:`fold_trap` draws nothing: every value, seeds included, is an argument, so the model is a
pure function of them, and it returns the keywords of :func:`resmill.export.to_grdecl` with what it built. With a
``rim`` it also says which columns a simulation keeps: the closure's own columns and the rim around them
(:func:`resmill.structure.outline`).
"""
import math

import numpy as np

from . import structure as st
from .fault_patterns import fold_faults

FOLD_STYLES = ("four_way", "turtle", "faulted_anticline", "fold_belt", "fault_bounded", "low_relief")
"""The styles of this composer: the fault patterns that stand on a closure alone."""


def fold_trap(style, x_len, y_len, dx, top, thickness, closure, roughness=None, faults=None, rim=None):
    """A fold-style trap as ``{"kwargs": ..., "meta": ...}``: the keywords for :func:`resmill.export.to_grdecl` and what
    was built.

    ``style`` is one of :data:`FOLD_STYLES`. ``x_len`` and ``y_len`` (m) are the model, ``dx`` (m) its cell size,
    ``top`` (m) the depth of the reservoir's top at the datum (a layer's ``top_depth``, to which the fold adds its
    shift) and ``thickness`` (m) that of the whole stack, the zones' total. The three blocks of values are dicts of the
    keywords of the primitives, so a keyword a primitive gains needs no change here:

    * ``closure``: :func:`resmill.structure.closure` (``area`` and ``height`` are required; ``center`` is the model's
      middle unless given);
    * ``roughness``: :func:`resmill.structure.roughness` without ``x_len`` and ``y_len`` (``sd``, ``range_m``,
      ``seed``); None for a smooth fold;
    * ``faults``: :func:`resmill.fault_patterns.fold_faults` without the model, ``style``, ``top`` and ``thickness``
      (``density`` and ``seed`` are required; ``over_salt``, ``regional``, ``max_faults``); None for no faults.

    Seeds are the caller's: a block without one is drawn from the system's entropy, as its primitive would be alone.
    ``rim`` (m, from column centre to column centre; ``x_len`` and ``y_len`` must be whole cells of ``dx``) cuts the
    simulation outline: the columns kept are the closure's own and every column within ``rim`` of them. The closure
    is the ellipse drawn (``area``, ``aspect`` and ``azimuth`` about the centre) joined with the trap around the crest
    (its spill contour, before the faults) of the fold as drawn, satellites and warp included, and of the fold plus
    its roughness as it came out, and the crest's column. The ellipse keeps the closure a grid cannot resolve, or one a
    tilt or the roughness leaves almost no trap of inside the map, from shrinking the model to a disc about the crest.

    ``kwargs`` is ``{"structure": the fold plus its roughness, "faults": the faults}`` (an empty list for none), for
    ``reservoir.to_grdecl(path, **result["kwargs"])``, and with a ``rim`` also ``"outline"``: the ``(nx, ny)`` bool
    columns kept, which ``to_grdecl`` writes as the only active ones (its geometry, seal and ``report`` stay those of
    the whole map). ``meta`` is plain numbers: ``style``, ``area`` (m2) and ``height`` (m) as asked, ``crest`` ((x, y)
    in m: the main culmination's crest, which a tilt moves off the centre), ``n_faults`` and ``fault_sets`` (the
    sorted kinds the faults were drawn as), and with a ``rim`` its value and ``outline_columns``. The closure of the
    model as built, with its faults and roughness, is read with :func:`resmill.fault_seal.blocks_at` from
    ``to_grdecl(report=)``.
    """
    if style not in FOLD_STYLES:
        raise ValueError(f"style must be one of {FOLD_STYLES}, got {style!r}")
    center = closure.get("center", (0.5 * x_len, 0.5 * y_len))
    fold = st.closure(**{**closure, "center": center})
    structure = fold if roughness is None else fold + st.roughness(x_len=x_len, y_len=y_len, **roughness)
    drawn = [] if faults is None else fold_faults(style, fold, x_len, y_len, dx, top=top, thickness=thickness, **faults)
    meta = dict(style=style, area=float(closure["area"]), height=float(closure["height"]),
                crest=tuple(float(c + o) for c, o in zip(center, fold.crest_offset)),
                n_faults=len(drawn), fault_sets=sorted({fault.kind for fault in drawn}))
    kwargs = dict(structure=structure, faults=drawn)
    if rim is not None:
        kwargs["outline"] = trap_outline(fold, structure, closure, center, x_len, y_len, dx, top, meta["crest"], rim)
        meta.update(rim=float(rim), outline_columns=int(kwargs["outline"].sum()))
    return dict(kwargs=kwargs, meta=meta)


def trap_outline(fold, structure, closure, center, x_len, y_len, dx, top, crest, rim):
    """The columns of the closure and ``rim`` m around it, ``(nx, ny)`` bool. The closure is the ellipse of ``closure``
    (``area``, ``aspect``, ``azimuth``) about ``center``, joined with the trap of ``top`` plus the fold and with that
    of ``structure``, the fold plus its roughness (the closure as it came out, which a low relief can move far from
    the drawn one), each read at the column centres (:func:`resmill.structure.closure_stats`) around its shallowest
    column within two of ``crest`` ((x, y) in m), and that column itself, so that a closure under a cell still leaves
    a model."""
    nx, ny = int(round(x_len / dx)), int(round(y_len / dx))
    sx, sy = x_len / nx, y_len / ny
    x, y = np.meshgrid((np.arange(nx) + 0.5) * sx, (np.arange(ny) + 0.5) * sy, indexing="ij")
    aspect, azimuth = closure.get("aspect", 1.0), math.radians(closure.get("azimuth", 0.0))
    across_half = math.sqrt(closure["area"] / (math.pi * aspect))
    along = (x - center[0]) * math.cos(azimuth) - (y - center[1]) * math.sin(azimuth)
    across = (x - center[0]) * math.sin(azimuth) + (y - center[1]) * math.cos(azimuth)
    footprint = (along / (aspect * across_half)) ** 2 + (across / across_half) ** 2 <= 1.0
    for surface in (fold,) if structure is fold else (fold, structure):
        depth = top + surface(x, y)
        cell = crest_cell(depth, crest, sx, sy)
        footprint |= st.closure_stats(depth, sx, sy, crest=cell)["mask"]
        footprint[cell] = True
    return st.outline(footprint, sx, sy, rim)


def crest_cell(depth, crest, sx, sy):
    """The shallowest column of ``depth`` ((nx, ny), the column centres' depths) within two columns of ``crest`` ((x, y) in m,
    columns ``sx`` by ``sy`` m): the crest of the trap on a rough surface, where the fold's own crest column may not be the
    shallowest."""
    nx, ny = depth.shape
    ci, cj = min(nx - 1, int(crest[0] // sx)), min(ny - 1, int(crest[1] // sy))
    i0, j0 = max(ci - 2, 0), max(cj - 2, 0)
    window = depth[i0:ci + 3, j0:cj + 3]
    return tuple(int(n + o) for n, o in zip(np.unravel_index(int(np.argmin(window)), window.shape), (i0, j0)))
