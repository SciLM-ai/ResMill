"""Fold traps: a closure, its roughness and its faults, composed from ResMill's own pieces.

The six fold styles (four-way, turtle, faulted anticline, fold belt, three-way fault-bounded and low relief) are one
construction. A closure (:func:`resmill.structure.closure`) is the fold, a rough surface
(:func:`resmill.structure.roughness`) the horizon error of a depth map added to it, and the style's faults
(:func:`resmill.fault_patterns.fold_faults`) are drawn on the fold without that roughness (a rough surface has many
local highs and is slow to search). What differs between the styles is what the caller draws for them: a fold belt
a steeper forelimb, a three-way trap a tilt, a low-relief trap a few metres of relief, and the style's own fault sets
inside ``fold_faults``. :func:`fold_trap` draws nothing: every value, seeds included, is an argument, so the model is a
pure function of them, and it returns the keywords of :func:`resmill.export.to_grdecl` with what it built.
"""
from . import structure as st
from .fault_patterns import fold_faults

FOLD_STYLES = ("four_way", "turtle", "faulted_anticline", "fold_belt", "fault_bounded", "low_relief")
"""The styles of this composer: the fault patterns that stand on a closure alone."""


def fold_trap(style, x_len, y_len, dx, top, thickness, closure, roughness=None, faults=None):
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

    ``kwargs`` is ``{"structure": the fold plus its roughness, "faults": the faults}`` (an empty list for none), for
    ``reservoir.to_grdecl(path, **result["kwargs"])``. ``meta`` is plain numbers: ``style``, ``area`` (m2) and
    ``height`` (m) as asked, ``crest`` ((x, y) in m: the main culmination's crest, which a tilt moves off the centre),
    ``n_faults`` and ``fault_sets`` (the sorted kinds the faults were drawn as). The closure of the model as built,
    with its faults, roughness and cells that hold no rock, is read with :func:`resmill.fault_seal.blocks_at` from
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
    return dict(kwargs=dict(structure=structure, faults=drawn), meta=meta)
