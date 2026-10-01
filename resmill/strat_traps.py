"""Stratigraphic traps: where a sand zone pinches out, is truncated, onlaps or changes facies.

A structural trap closes by the dip of its top surface; a stratigraphic one closes where the sand stops. Where it is
absent (pinched out, eroded, onlapped) a column is a wall, never an exit: the only way out of a trap is over the map's
edge. :func:`zone_trap` reads the traps of the top of a zone's active cells on that rule, with the flood of
:func:`resmill.structure.closure_stats`, which neither loops on nor leaks through such columns. Read the other way,
with an absent neighbour counting as an exit, a tongue of sand protruding updip from a sheet that continues downdip
has no closure at all, though it holds the dip times its length (15.4 m in the research prototype, 0 m by that
reading).

A straight pinch-out line on a plane monocline has no closure either: every point of it lies at the same depth, so the
oil spills along it. The line has to indent updip (a tongue, a lens: the pure traps, whose closure is the tangent of the
dip times their length along dip, Berg's 13-37 m columns at 1 degree and a kilometre) or to cross a nose (a combination
trap, whose closure is the nose's).
"""
import numpy as np
from scipy import ndimage

from .structure import _spill_levels


def zone_trap(zc, act, dx, dy, k=None, column=None):
    """The traps of the top of the active cells of a zone, absent columns being walls and the map's edge the only exit.

    ``zc`` (2nx, 2ny, nk + 1) is the interface stack and ``act`` (nx, ny, nk) the active cells, both k top-down as
    :func:`resmill.export._build_geometry` returns them; ``k`` (a slice of the layers, default all) picks the zone
    that is net reservoir, so that a barrier zone of active cells under or beside it is not part of the surface.
    ``column`` (m) is the oil column the updip seal holds if that is capillary; a trap holds no more than that
    below its crest. The surface is the top of the first active cell of each column.

    Returns one dict per body of connected columns that has any, the shallowest crest first: ``crest`` (i, j),
    ``crest_depth``, ``spill_depth`` (the deepest top on the best way out, None when nothing reaches the body:
    sealed all round), ``spill_point`` (the column beside the trap the oil leaves into, None when sealed),
    ``limit_depth`` (the deepest contact the trap holds: the spill, the deepest top of a sealed body, or ``column``
    below the crest, whichever is shallowest), ``limited_by`` ("spill", "sealed" or "barrier"), ``height``
    (``limit_depth`` - ``crest_depth``), and the ``area`` (m2) and ``mask`` of the columns shallower than the limit
    that join the crest (all of a sealed body: it fills to its deepest point). A body with no closure has height 0.
    """
    lo, hi, _ = (slice(None) if k is None else k).indices(act.shape[2])
    act, zc = act[:, :, lo:hi], zc[:, :, lo:hi + 1]
    alive = act.any(axis=2)
    tops = 0.25 * (zc[0::2, 0::2, :-1] + zc[1::2, 0::2, :-1] + zc[0::2, 1::2, :-1] + zc[1::2, 1::2, :-1])
    depth = np.where(alive, np.take_along_axis(tops, np.argmax(act, axis=2)[..., None], axis=2)[..., 0], np.inf)
    spill = _spill_levels(depth)
    bodies, count = ndimage.label(alive)
    traps = []
    for b in range(1, count + 1):
        body = bodies == b
        crest = np.unravel_index(int(np.argmin(np.where(body, depth, np.inf))), depth.shape)
        sealed = not np.isfinite(spill[crest])
        deepest = float(depth[body].max() if sealed else spill[crest])
        limit = deepest if column is None else min(deepest, float(depth[crest]) + float(column))

        def trapped(level):
            joined, _ = ndimage.label(body & (depth < level))
            return (joined == joined[crest]) & (joined > 0)

        full = body if sealed else trapped(deepest)
        leaves = [] if sealed else np.argwhere(ndimage.binary_dilation(full) & ~full & alive & (depth == deepest))
        mask = full if limit == deepest else trapped(limit)
        traps.append(dict(
            crest=(int(crest[0]), int(crest[1])), crest_depth=float(depth[crest]),
            spill_depth=None if sealed else deepest,
            spill_point=tuple(int(c) for c in leaves[0]) if len(leaves) else None,
            limit_depth=limit, limited_by="barrier" if limit < deepest else "sealed" if sealed else "spill",
            height=limit - float(depth[crest]), area=float(mask.sum()) * dx * dy, mask=mask))
    return sorted(traps, key=lambda trap: trap["crest_depth"])
