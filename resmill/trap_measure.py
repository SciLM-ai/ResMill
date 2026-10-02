"""The trap measure of a zone whose sand stops: pools, spill, a barrier's column, and the contact an initialisation can use.

A structural trap closes by the dip of its top surface; a stratigraphic one closes where the sand stops. Where it is
absent (pinched out, eroded, onlapped) a column is a wall, never an exit: the only way out of a trap is over the map's
edge. :func:`zone_trap` reads the traps of the top of a zone's active cells on that rule, as pools (the elder rule of
:func:`_pools`: every local minimum of the top, with the level at which it spills into a pool with a shallower crest or
leaves the map; for a body's own pool that level is the spill that :func:`resmill.structure.closure_stats` gives, the
deepest top on the best way out, which neither loops on nor leaks through such columns). Read the other way, with an
absent neighbour counting as an exit, a tongue of sand protruding updip from a sheet that continues downdip has no
closure at all, though it holds the dip times its length (15.4 m in the research prototype, 0 m by that reading).

:func:`barrier_column` is Berg's (1975) oil column that a finer barrier zone holds in a coarser reservoir sand, the
limit of a facies-change trap's fill.
"""
import math

import numpy as np
from scipy import ndimage

MIN_POOL = 1.0                      # m: a pool of less closure is not listed (the body's own always is): a cell's own rise [J]

GRAVITY = 9.81                      # m/s2
# Berg's packing of the grains as uniform spheres, rhombohedral (porosity 26 %; Graton and Fraser 1935): the pores
# between them are 0.414 of a grain diameter across, the throats connecting them 0.154.
PORE_RADIUS, THROAT_RADIUS = 0.5 * 0.414, 0.5 * 0.154


def effective_grain_size(permeability, porosity):
    """The effective grain size (m) that sets a sandstone's pore size, from its permeability (mD) and porosity (a
    fraction): D = (1.89 k n^-5.1)^0.5 cm, k in mD and the porosity n in percent (Berg 1975, eq. 33, his
    empirical relation of permeability, porosity and the 90th-percentile grain size, which he applies to sandstones of
    under 30 % porosity too)."""
    return 1e-2 * np.sqrt(1.89 * np.asarray(permeability, dtype=float) * (100.0 * np.asarray(porosity)) ** -5.1)


def barrier_column(delta_rho, d_reservoir, d_barrier, sigma=0.030):
    """The oil (or gas) column (m) a finer barrier holds in a coarser reservoir, as the barrier's capillary entry
    pressure less the reservoir's against the buoyancy of the column: z = 2 sigma (1 / r_t - 1 / r_p) / (g delta_rho)
    (Berg 1975, eq. 16). The throats r_t are those of the barrier and the pores r_p those of the reservoir, each a fixed
    share of its effective grain size (:data:`THROAT_RADIUS`, :data:`PORE_RADIUS`; :func:`effective_grain_size`).
    ``delta_rho`` is the water minus hydrocarbon density (kg/m3), ``d_reservoir`` and ``d_barrier`` the grain sizes
    (m) and ``sigma`` the interfacial tension (N/m; 30-35 mN/m for oil). Zero for a barrier no finer than the
    reservoir: Berg's formula would still give 3 m for the same sand and 0.6 m for one twice as coarse, the height
    for oil to migrate through such a stringer, a threshold of migration and no seal. It is the column above the
    contact in the reservoir sand; the free-water level lies below that contact by the sand's own entry-pressure
    head. Berg's calculated columns (14-64 ft, Table 1) are lower bounds of the 25-120+ ft he observed at three of his
    four fields."""
    d_barrier, d_reservoir = np.asarray(d_barrier, dtype=float), np.asarray(d_reservoir, dtype=float)
    term = 1.0 / (THROAT_RADIUS * d_barrier) - 1.0 / (PORE_RADIUS * d_reservoir)
    return np.where(d_barrier < d_reservoir, np.maximum(2.0 * sigma * term / (GRAVITY * np.asarray(delta_rho, dtype=float)),
                                                        0.0), 0.0)


def zone_top(zc, act, k=None):
    """The depth map (nx, ny) of the top of the active cells of a zone: that of the first active cell of each column,
    infinite where the zone is absent. ``zc`` (2nx, 2ny, nk + 1) and ``act`` (nx, ny, nk) are as
    :func:`resmill.export._build_geometry` returns them (k top-down); ``k``, a slice of the layers, picks the zone
    (default all)."""
    lo, hi, _ = (slice(None) if k is None else k).indices(act.shape[2])
    act, zc = act[:, :, lo:hi], zc[:, :, lo:hi + 1]
    tops = 0.25 * (zc[0::2, 0::2, :-1] + zc[1::2, 0::2, :-1] + zc[0::2, 1::2, :-1] + zc[1::2, 1::2, :-1])
    first = np.take_along_axis(tops, np.argmax(act, axis=2)[..., None], axis=2)[..., 0]
    return np.where(act.any(axis=2), first, np.inf)


def _pools(depth):
    """The pools of a depth map, as the elder rule finds them by raising the water over the finite cells (4-connected;
    a plateau is one pool): every local minimum is a pool, and it ends at the level where its water meets that of a pool
    with a shallower crest (it then spills into that pool) or first holds a cell of the map's edge (it leaves the map).
    Returns ``(crest, spill, into)`` for each: the crest (i, j), the level it ends at (infinite for the body's own
    pool when nothing reaches the edge: sealed) and the crest of the pool it spills into (None where it leaves the
    map)."""
    nx, ny = depth.shape
    level = depth.ravel().tolist()
    cells = [c for c in np.argsort(depth.ravel(), kind="stable").tolist() if level[c] < math.inf]
    # union-find over the cells the water has reached; per root: the crest of its pool (None: it has none) and whether
    # its water holds a cell of the edge
    parent, crest, opened, pools = {}, {}, {}, []

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    def ends(c, at, into):                              # the pool with crest cell c stops being one at the level ``at``
        pools.append((divmod(c, ny), at, None if into is None else divmod(into, ny)))

    def join(a, b, at):
        a, b = find(a), find(b)
        if a == b:
            return
        if crest[a] is not None and crest[b] is not None and opened[a] == opened[b]:
            if not opened[a]:                           # two pools meet: the one with the deeper crest spills into the other
                young, old = sorted((crest[a], crest[b]), key=lambda c: (level[c], c), reverse=True)
                ends(young, at, old)
        elif crest[a] is not None and crest[b] is not None:              # a pool meets water that holds the edge
            closed = b if opened[a] else a
            ends(crest[closed], at, crest[a if closed == b else b])
        elif crest[a] is not None and opened[b] and not opened[a]:
            ends(crest[a], at, None)
        elif crest[b] is not None and opened[a] and not opened[b]:
            ends(crest[b], at, None)
        parent[b] = a
        opened[a] = opened[a] or opened[b]
        crest[a] = min((c for c in (crest[a], crest[b]) if c is not None), key=lambda c: (level[c], c), default=None)

    k = 0
    while k < len(cells):
        at, group = level[cells[k]], []
        while k < len(cells) and level[cells[k]] == at:
            group.append(cells[k])
            k += 1
        steps = lambda c: [m for m, ok in ((c - ny, c >= ny), (c + ny, c < (nx - 1) * ny), (c - 1, c % ny > 0),
                                          (c + 1, c % ny < ny - 1)) if ok and m in parent]
        for c in group:
            i, j = divmod(c, ny)
            parent[c], crest[c], opened[c] = c, c, i in (0, nx - 1) or j in (0, ny - 1)
        for c in group:                                  # cells of one level that touch are one plateau: one pool, not many
            for m in steps(c):
                if level[m] == at:
                    a, b = find(c), find(m)
                    if a != b:
                        parent[b], opened[a], crest[a] = a, opened[a] or opened[b], min(crest[a], crest[b])
        for r in {find(c) for c in group}:
            lower = [m for c in group if find(c) == r for m in steps(c) if level[m] < at]
            if not lower:                                # nothing shallower is joined to it: a new pool is born
                if opened[r]:
                    ends(crest[r], at, None)             # on the map's edge from its birth: no closure
            else:
                crest[r] = None                          # it lies beside shallower water: no pool of its own
                for m in lower:
                    join(r, m, at)
    for r in {find(c) for c in parent}:
        if not opened[r]:
            pools.append((divmod(crest[r], ny), math.inf, None))
    return pools


def _traps(depth, dx, dy, column=None, barrier_top=None, min_height=MIN_POOL):
    """The traps of a map of depths (nx, ny), infinite where the zone is absent (:func:`zone_trap`, whose result it is,
    on any such map: the top of a zone's cells, or an analytic surface): one for each pool (:func:`_pools`) of at least
    ``min_height`` closure and the shallowest of each connected body, whatever its closure. A barrier holds its
    ``column`` below the shallower of the crest and ``barrier_top``, the shallowest top of its cells (an initialisation
    by contacts puts oil in any cell above its entry pressure, joined to the trap or not)."""
    if column is not None and column < 0.0:
        raise ValueError(f"column must not be negative, not {column}")
    alive = np.isfinite(depth)
    bodies, count = ndimage.label(alive)
    edge = np.zeros(depth.shape, dtype=bool)
    edge[[0, -1], :] = edge[:, [0, -1]] = True
    edge_top = float(depth[edge].min())                    # the shallowest column on the edge: the oil above it leaks
    own = {b: np.unravel_index(int(np.argmin(np.where(bodies == b, depth, np.inf))), depth.shape) for b in
           range(1, count + 1)}
    traps = []
    for crest, spill, into in _pools(depth):
        b = bodies[crest]
        primary = own[b] == crest
        if not primary and spill - depth[crest] < min_height:
            continue
        body = bodies == b
        sealed = not np.isfinite(spill)
        deepest = float(depth[body].max() if sealed else spill)
        reach = float(depth[crest]) if barrier_top is None else min(float(depth[crest]), barrier_top)
        limit = deepest if column is None else max(min(deepest, reach + float(column)), float(depth[crest]))

        def trapped(level):
            joined, _ = ndimage.label(body & (depth < level))
            return (joined == joined[crest]) & (joined > 0)

        full = body if sealed else trapped(deepest)
        leaves = [] if sealed else np.argwhere(ndimage.binary_dilation(full) & ~full & alive & (depth == deepest))
        mask = full if limit == deepest else trapped(limit)
        traps.append(dict(
            crest=(int(crest[0]), int(crest[1])), crest_depth=float(depth[crest]), body=int(b), primary=primary,
            into=into, spill_depth=None if sealed else deepest,
            spill_point=tuple(int(c) for c in leaves[0]) if len(leaves) else None,
            limit_depth=limit, limited_by="barrier" if limit < deepest else "sealed" if sealed else "spill",
            closure=deepest - float(depth[crest]), height=limit - float(depth[crest]),
            contact_limit=max(min(limit, edge_top), float(depth[crest])), area=float(mask.sum()) * dx * dy, mask=mask,
            barrier_top=barrier_top))
    return sorted(traps, key=lambda trap: trap["crest_depth"])


def zone_trap(zc, act, dx, dy, k=None, column=None, barrier=None, min_height=MIN_POOL):
    """The traps of the top of the active cells of a zone, absent columns being walls and the map's edge the only exit.

    ``zc`` (2nx, 2ny, nk + 1) is the interface stack and ``act`` (nx, ny, nk) the active cells, both k top-down as
    :func:`resmill.export._build_geometry` returns them; ``k`` (a slice of the layers, default all) picks the zone
    that is net reservoir, so that a barrier zone of active cells under or beside it is not part of the surface
    (:func:`zone_top`). ``column`` (m) is the oil column the updip seal holds if that is capillary; a trap holds no more
    than that below its crest, or below the shallowest top of the barrier's cells, ``barrier`` (a slice of the layers),
    if they reach updip of it (``barrier_top`` in each trap, None without a barrier).

    Returns one dict per pool, the shallowest crest first: every local minimum of the zone's top is one (the elder rule,
    :func:`_pools`), listed if it closes by ``min_height`` (a cell's own rise: 1 m) or is the shallowest of its body.
    A rough edge leaves a median 12-25 pools (up to 90) over a pinch-out, truncation, onlap or nose of the plan's ranges.
    The keys: ``crest`` (i, j), ``crest_depth``, ``body`` (the connected body of columns it lies in) and ``primary``
    (the body's shallowest); ``spill_depth`` (the level where the pool ends: it meets a pool with a shallower crest, or
    the edge; None when nothing reaches the body: sealed all round), ``into`` (the crest of the pool it spills into,
    None where it leaves the map), ``spill_point`` (the column beside the trap the oil leaves into, None when sealed),
    ``closure`` (the spill, or the deepest top of a sealed body, less the crest), ``limit_depth`` (the deepest contact
    the trap holds: the spill, the deepest top of a sealed body, or ``column`` below the crest, whichever is
    shallowest), ``limited_by`` ("spill", "sealed" or "barrier"), ``height`` (``limit_depth`` - ``crest_depth``), and
    the ``area`` (m2) and ``mask`` of the columns shallower than the limit that join the crest (all of a sealed
    body: it fills to its deepest point). A body with no closure has height 0.

    ``contact_limit`` is the deepest contact that an initialisation by contacts can use with this pool holding the
    oil: the limit, or the shallowest column on the map's edge if that is shallower (the crest at the least). An
    initialisation puts oil in every cell above the contact, joined to the trap or not, and the water of a pool that
    holds a cell of the edge leaks: at the old rule, the main limit less 1 m, 5 % of the oil-bearing sand of a
    pinch-out (mean; 8-9 % of a truncation or onlap, 20-22 % of a nosed one) lay in water that leaks; at
    ``contact_limit`` none does. A contact that keeps every pool closed costs the main trap 12-16 % of its closure
    (median; P90 29-55 %).
    """
    top = None if barrier is None else float(zone_top(zc, act, barrier).min())
    return _traps(zone_top(zc, act, k), dx, dy, column, top, min_height)
