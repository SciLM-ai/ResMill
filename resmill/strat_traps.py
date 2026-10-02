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

Real sand limits are not lines. A pinch-out, a facies change, a truncation or an onlap is irregular at every scale from
a cell up to the trap, with several tongues (lobate or digitate) and embayments, and the sand thins to its limit
smoothly: the Berea Sandstone's thickness contour (USGS PP 259), the subcrops of Flanders and the Brussels Sands, the
House Creek sand ridge (USGS DDS-33). :func:`strat_trap` draws such an edge from a taper and a multi-scale relief
(:func:`resmill.structure.taper`, :func:`resmill.structure.relief`), lets an erosion surface's relief make a
truncation's or an onlap's edge, shapes a lens as a mound on a flat base and gives the tops a relief of their own.
The closure of such a geometry is whatever it holds, read from the cells by :func:`trap_report`; the tangent of the dip
times the length of the drawn tongue is the nominal one.
"""
import numpy as np
from scipy import ndimage

from . import structure as st
from .export import _build_geometry
from .structure import _spill_levels

KINDS = ("pinchout", "facies_change", "lens", "truncation", "onlap", "pinchout_nose", "truncation_nose")
NOSED = KINDS[-2:]                  # the combination kinds
MAX_NOSE = 335.0                    # m: the closure of Kuparuk (Carman and Hardwick 1983), the largest [J]

MIN_THICKNESS = 6e-3                # m: a cell thinner than this is collapsed, as the deck's PINCH threshold takes it
EDGE_MARGIN = 0.02                  # of the model's size: a trap this close to its edge is taken to touch it [J]
COLLAPSED = 0.2                     # of the nominal closure: a wandering edge that leaves less has destroyed the trap [J]
WANDER_TONGUE, WANDER_TAPER = 0.2, 0.25     # the most an edge may wander, of the main tongue's length (depositional) or of
#                                             the taper (an erosion surface, or across a nose): more breaks the sand into
#                                             pieces that hold no trap (the owner's range, 2026-10-02, judgement [J])

RIM_SHARE = 0.25                    # of column / tan(dip), the widest rim that leaves the oil a column: the rim's width [J]

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
    (m) and ``sigma`` the interfacial tension (N/m; 30-35 mN/m for oil). Zero where the barrier's throats are no
    narrower than the reservoir's pores (a barrier 2.7 times as coarse). It is the column above the contact in the
    reservoir sand; the free-water level lies below that contact by the sand's own entry-pressure head."""
    term = 1.0 / (THROAT_RADIUS * np.asarray(d_barrier, dtype=float)) - 1.0 / (PORE_RADIUS * np.asarray(d_reservoir))
    return np.maximum(2.0 * sigma * term / (GRAVITY * np.asarray(delta_rho, dtype=float)), 0.0)


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


def _traps(depth, dx, dy, column=None, barrier_top=None):
    """The traps of a map of depths (nx, ny), infinite where the zone is absent (:func:`zone_trap`, whose result it is,
    on any such map: the top of a zone's cells, or an analytic surface). A barrier holds its ``column`` below the
    shallower of the crest and ``barrier_top``, the shallowest top of its cells (an initialisation by contacts puts oil
    in any cell above its entry pressure, joined to the trap or not)."""
    if column is not None and column < 0.0:
        raise ValueError(f"column must not be negative, not {column}")
    alive = np.isfinite(depth)
    spill = _spill_levels(depth)
    bodies, count = ndimage.label(alive)
    traps = []
    for b in range(1, count + 1):
        body = bodies == b
        crest = np.unravel_index(int(np.argmin(np.where(body, depth, np.inf))), depth.shape)
        sealed = not np.isfinite(spill[crest])
        deepest = float(depth[body].max() if sealed else spill[crest])
        reach = float(depth[crest]) if barrier_top is None else min(float(depth[crest]), barrier_top)
        limit = deepest if column is None else max(min(deepest, reach + float(column)), float(depth[crest]))

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
            closure=deepest - float(depth[crest]), height=limit - float(depth[crest]), area=float(mask.sum()) * dx * dy,
            mask=mask, barrier_top=barrier_top))
    return sorted(traps, key=lambda trap: trap["crest_depth"])


def zone_trap(zc, act, dx, dy, k=None, column=None, barrier=None):
    """The traps of the top of the active cells of a zone, absent columns being walls and the map's edge the only exit.

    ``zc`` (2nx, 2ny, nk + 1) is the interface stack and ``act`` (nx, ny, nk) the active cells, both k top-down as
    :func:`resmill.export._build_geometry` returns them; ``k`` (a slice of the layers, default all) picks the zone
    that is net reservoir, so that a barrier zone of active cells under or beside it is not part of the surface
    (:func:`zone_top`). ``column`` (m) is the oil column the updip seal holds if that is capillary; a trap holds no more
    than that below its crest, or below the shallowest top of the barrier's cells, ``barrier`` (a slice of the layers),
    if they reach updip of it (``barrier_top`` in each trap, None without a barrier).

    Returns one dict per body of connected columns that has any, the shallowest crest first: ``crest`` (i, j),
    ``crest_depth``, ``spill_depth`` (the deepest top on the best way out, None when nothing reaches the body:
    sealed all round), ``spill_point`` (the column beside the trap the oil leaves into, None when sealed),
    ``limit_depth`` (the deepest contact the trap holds: the spill, the deepest top of a sealed body, or ``column``
    below the crest, whichever is shallowest), ``limited_by`` ("spill", "sealed" or "barrier"), ``height``
    (``limit_depth`` - ``crest_depth``), and the ``area`` (m2) and ``mask`` of the columns shallower than the limit
    that join the crest (all of a sealed body: it fills to its deepest point). A body with no closure has height 0.
    """
    top = None if barrier is None else float(zone_top(zc, act, barrier).min())
    return _traps(zone_top(zc, act, k), dx, dy, column, top)


def _packed(widths, rng):
    """Offsets (m) along strike that set bodies of the given full ``widths`` side by side in a random order with gaps
    of 10-50 % of their mean width, the row centred on 0."""
    widths = np.asarray(widths, dtype=float)
    order = rng.permutation(len(widths))
    w = widths[order]
    start = np.concatenate([[0.0], np.cumsum(w[:-1] + rng.uniform(0.1, 0.5, len(w) - 1) * w.mean())])
    centres = start + 0.5 * w
    offsets = np.empty(len(w))
    offsets[order] = centres - 0.5 * (start[-1] + w[-1])
    return offsets


def _check(kind, barrier, column, mound, dip, taper_angle, area, tongues, thicknesses, stagger, nose):
    """Refuse what :func:`strat_trap` cannot build; returns the number of sand layers (the thicknesses less the
    barrier's)."""
    nosed, lens = kind in NOSED, kind == "lens"
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    if not 0.0 < taper_angle < 90.0:
        raise ValueError(f"taper_angle must lie between 0 and 90 degrees, not {taper_angle}")
    if bool(barrier) != (kind == "facies_change") and not (lens and barrier):
        raise ValueError(f"a barrier zone is what a facies change needs and a lens may have, not a {kind}")
    if bool(barrier) != (column is not None):
        raise ValueError("a barrier zone needs the oil column it holds (column, m: Berg's barrier_column) and only a "
                         "barrier zone has one")
    if mound and kind not in ("lens", "pinchout", "facies_change"):
        raise ValueError(f"a convex-up top is the shape of a lens, a pinch-out tongue or a facies change, not of a {kind}")
    if lens and area is None:
        raise ValueError("a lens needs an area")
    if nosed != (nose is not None):
        raise ValueError(f"a nose (its area, height and aspect) is what {', '.join(NOSED)} need, not a {kind}")
    if nosed and not {"area", "height", "aspect"} <= set(nose):
        raise ValueError(f"a nose needs its area, height and aspect, not {sorted(nose)}")
    if nosed and not 0.0 < nose["height"] <= MAX_NOSE:
        raise ValueError(f"the nose's height must lie between 0 and {MAX_NOSE:.0f} m, the closure of Kuparuk")
    if nosed and nose.get("tilt", 0.0):
        raise ValueError("a nose takes no tilt: the plane is the dip")
    if kind.startswith("truncation") and taper_angle > dip:
        raise ValueError(f"the erosion surface must dip the same way as the beds, less steeply: the discordance "
                         f"taper_angle ({taper_angle}) cannot exceed the bed dip ({dip})")
    if len(tongues) and (nosed or lens or area is None):
        raise ValueError(f"tongues are further lobes on the edge of a pinch-out, truncation or onlap with an area, "
                         f"not on a {kind}{' without one' if area is None else ''}")
    if not all(t > 0.0 for t in thicknesses) or not all(a > 0.0 and r > 0.0 for a, r in tongues):
        raise ValueError(f"thicknesses and the (area, aspect) of the tongues must be positive, not {thicknesses} "
                         f"and {list(tongues)}")
    n_net = len(thicknesses) - bool(barrier)
    if n_net < 1:
        raise ValueError("thicknesses needs a sand layer as well as the barrier")
    if stagger and (kind not in ("pinchout", "facies_change") or n_net < 2):
        raise ValueError(f"a stagger (the sand layers ending each farther downdip) needs a pinch-out or a facies "
                         f"change of more than one sand layer, not a {kind} of {n_net}")
    return n_net


def _check_numbers(**given):
    """Refuse a number that is not a size: ``warp``, ``stagger``, ``wander`` and ``relief_sd`` may be 0, the others
    (``dip``, ``area``, ``aspect``, the ranges, ``floor_m``, ``cell``) are positive where given."""
    for name, value in given.items():
        zero_ok = name in ("warp", "stagger", "wander", "relief_sd")
        if value is not None and not (value >= 0.0 if zero_ok else value > 0.0):
            raise ValueError(f"{name} must be {'at least 0' if zero_ok else 'positive'}, not {value}")


def _shapes(nose, area, aspect, tongues, lens):
    """The footprints of the shapes: their (area, aspect) as drawn, their lengths along dip and, for each, how far it
    reaches updip and downdip of its line (of its middle, for a lens) and half across it."""
    if nose is not None:
        reach = np.sqrt(nose["area"] / (np.pi * nose["aspect"]))                 # the nose's half-length along dip
        return [], [0.0], [(reach, reach, nose["aspect"] * reach)]
    if area is None:
        return [], [0.0], [(0.0, 0.0, 0.0)]                                      # a straight line: a point to fit
    parts = [(float(area), float(aspect))] + [(float(a), float(r)) for a, r in tongues]
    lengths = [2.0 * np.sqrt(a / (np.pi * r)) for a, r in parts]                 # each trap's length along dip
    return parts, lengths, [(0.5 * n, 0.5 * n, 0.5 * r * n) if lens else (n, 0.0, 0.5 * r * n)
                            for n, (_, r) in zip(lengths, parts)]


def _line_shift(sizes, offsets, x_len, y_len, dip_dir, strike, lens):
    """How far along dip from the model's centre (m) the line goes (the middle of a lens: 0): as far updip as the rims
    of all the shapes fit within 80 % of the model's size; a ValueError where they do not fit at all."""
    with np.errstate(divide="ignore"):                       # the model's half-length along dip, through its centre
        half = float(min(0.5 * x_len / abs(dip_dir[0]), 0.5 * y_len / abs(dip_dir[1])))
    phi = np.linspace(0.0, 2.0 * np.pi, 73)                                    # the rim of a shape, its base too

    def fits(shift):                                   # the rims, the line ``shift`` m from the centre along dip
        for (up, down, wide), v in zip(sizes, offsets):
            rim = (shift + np.where(np.cos(phi) < 0.0, up, down) * np.cos(phi))[:, None] * dip_dir \
                + (v + wide * np.sin(phi))[:, None] * strike
            if not (np.abs(rim) <= 0.4 * np.array([x_len, y_len])).all():
                return False
        return True

    grid = np.linspace(-half, half, 401)               # the line goes as far updip as the shapes fit (a lens: centred)
    first = [0] if lens else [k for k, shift in enumerate(grid) if fits(shift)][:1]
    if not first or (lens and not fits(0.0)):
        raise ValueError(f"the trap ({sum(max(u + d, 2 * w) for u, d, w in sizes[:1]):.0f} m across, with "
                         f"{max(len(sizes) - 1, 0)} more tongues) does not fit the model: make the model larger or "
                         f"the area smaller")
    if lens:
        return 0.0
    lo, shift = grid[max(first[0] - 1, 0)], grid[first[0]]
    for _ in range(40):                                # the fitting shifts are an interval: bisect to its updip end
        mid = 0.5 * (lo + shift)
        lo, shift = (lo, mid) if fits(mid) else (mid, shift)
    return shift


def _floors(wander, range_m, relief_sd, relief_range, floor_m, cell):
    """The finest wavelength (m) of the edge's relief and of the top's: ``floor_m`` if it is given, else a 32nd of the
    range but not under two cells (a ValueError if there is neither a ``floor_m`` nor a ``cell`` to read it from, or
    if a range is shorter than that)."""
    if floor_m is None and cell is None and (wander or relief_sd):
        raise ValueError("a sand edge or top with relief needs the cell width (cell, m: the floor of its octaves is "
                         "two cells) or the floor itself (floor_m)")
    floors = [floor_m if floor_m is not None else max(r / 32.0, 2.0 * (cell or 0.0)) for r in (range_m, relief_range)]
    for used, r, f, name in ((wander, range_m, floors[0], "range_m"), (relief_sd, relief_range, floors[1],
                                                                       "relief_range")):
        if used and f > r:
            raise ValueError(f"{name} ({r:g} m) is below the finest wavelength of its relief ({f:g} m: two cells): "
                             f"use a longer range or a finer grid")
    return floors


def _check_wander(kind, wander, length, taper_m):
    """Refuse an edge that wanders more than it may: a fifth of the main tongue's length (a tongue of sand: depositional
    edges) and, where the edge is the relief of an erosion surface or lies across a nose, a quarter of the taper."""
    erosion = kind in ("truncation", "onlap") or kind in NOSED
    limit = min(WANDER_TONGUE * length if length and not erosion else np.inf, WANDER_TAPER * taper_m if erosion else
                np.inf)
    if wander > limit * (1.0 + 1e-9):
        what = f"{WANDER_TAPER:g} of the taper" if erosion else f"{WANDER_TONGUE:g} of the main tongue's length"
        raise ValueError(f"wander ({wander:.0f} m) must be at most {limit:.0f} m, {what}: more breaks the sand into "
                         f"pieces that hold no trap")


def _footprint(nose, parts, offsets, at, dip_dir, strike, azimuth, warp, lens, seeds):
    """What outlines the sand: ``(fold, outline, line)``, the nose's fold (None without one), the footprint of the
    tongues (a Structure, negative inside; None for a straight line or a nose) and where the line lies along dip
    (m from the origin; None for a lens)."""
    fold = outline = line = None
    if nose is not None:
        fold = st.closure(azimuth=azimuth, center=tuple(at), seed=seeds[0], **nose)
        line = float((at + fold.crest_offset) @ dip_dir)
    elif parts:
        outlines = [st.closure(a if lens else 2.0 * a, 1.0, aspect=r if lens else 0.5 * r, azimuth=azimuth,
                               center=tuple(at + v * strike), warp=warp, seed=k)
                    for (a, r), v, k in zip(parts, offsets, seeds[1].spawn(len(parts)))]
        outline = outlines[0] if len(outlines) == 1 else st.Structure(
            lambda x, y: np.minimum.reduce([o(x, y) for o in outlines]))
    if not lens and line is None:
        line = float(at @ dip_dir)
    return fold, outline, line


def _plan(x_len, y_len):
    """The plan grid (301 x 301 points over the model) that the refusals read the drawn sand on."""
    return np.meshgrid(np.linspace(0.0, x_len, 301), np.linspace(0.0, y_len, 301), indexing="ij")


def _refuse_edge(foot, lens, nosed, line, dip_dir, plan, margin):
    """Refuse the footprint of the lobes (a Structure, negative inside) if it comes within ``margin`` m (along x and
    along y) of the model's edge where the trap is, over which it would leak: updip of the line, all round a lens,
    downdip of it for a nose."""
    (gx, gy), (mx, my) = plan, margin
    x_len, y_len = gx.max(), gy.max()
    inside, along_dip = foot(gx, gy) < 0.0, gx * dip_dir[0] + gy * dip_dir[1]
    edge = (np.minimum(gx, x_len - gx) < mx) | (np.minimum(gy, y_len - gy) < my)
    watched = edge if lens else edge & (along_dip > line if nosed else along_dip < line)   # where the trap is
    if (watched & inside).any():
        raise ValueError(f"the lobes of the trap come within {max(mx, my):.0f} m of the model's edge, over which it "
                         f"would leak: use another seed, a smaller warp or a larger model")


def _refuse_collapse(present, depth, nominal, plan, margin):
    """Refuse a draw whose wandering edge has destroyed the trap: on the plan grid, the sand's top ``depth`` where
    ``present`` (a mask) has no closure of a fifth of the ``nominal`` one, the margin of the model's edge being
    taken as the edge over which it leaks. The sand reaches the model's edge or breaks up (3 % of the pinch-outs and
    1 % of the truncations at the plan's ranges, 0 m against 34-40 m nominal, nearly all with the crest on the edge)."""
    (gx, gy), (mx, my) = plan, margin
    dx, dy = gx.max() / 300.0, gy.max() / 300.0
    kx, ky = int(np.ceil(mx / dx)), int(np.ceil(my / dy))
    inner = (slice(kx, -kx or None), slice(ky, -ky or None))
    traps = _traps(np.where(present, depth, np.inf)[inner], dx, dy)
    best = max((trap["height"] for trap in traps), default=0.0)
    if best < COLLAPSED * nominal:
        raise ValueError(f"the wandering edge has destroyed the trap: it closes by {best:.1f} m on the plan grid "
                         f"against {nominal:.1f} m nominal: use another seed or a smaller wander")


def _barrier(thicknesses, layers_f, flat, rim_m, line, outline, azimuth, x_len, y_len, rough):
    """The thickness factor of the barrier zone: under the sand and a rim of ``rim_m`` m beyond its edge, thinning to
    nothing over the rim (walls beyond it), so that no barrier cell lies updip of the trap, where an initialisation by
    contacts would fill it with oil. It takes the thickness the sand loses (``layers_f``, the factors of the sand's
    layers) unless it is a ``flat`` slab under a mound."""
    rim = st.taper(line, rim_m, azimuth, outline=outline, x_len=x_len, y_len=y_len, edge=rough, grow=rim_m)
    if flat:
        return rim
    return st.Structure(lambda x, y: rim(x, y) * (1.0 + sum(t * (1.0 - g(x, y)) for t, g in zip(thicknesses, layers_f))
                                                  / thicknesses[-1]))


def _thickness(line, outline, taper_m, azimuth, x_len, y_len, rough, stagger, n_net):
    """The thickness factor of each sand layer: ``factor(shift)`` is that of a layer ending ``shift`` m downdip of the
    first (the line and every tongue's outline moved), and the list has one per sand layer."""
    dip_dir = np.array([np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))])

    def factor(shift):
        moved = outline if outline is None or not shift else st.Structure(
            lambda x, y: outline(x - shift * dip_dir[0], y - shift * dip_dir[1]))
        return st.taper(None if line is None else line + shift, taper_m, azimuth, outline=moved, x_len=x_len,
                        y_len=y_len, edge=rough)

    f = factor(0.0)
    return f, ([f] + [factor(k * stagger) for k in range(1, n_net)] if stagger else [f] * n_net)


def strat_trap(kind, x_len, y_len, top, thicknesses, seed, barrier=False, dip=1.0, azimuth=0.0, taper_angle=0.3,
               area=3.4e6, aspect=2.2, warp=0.3, tongues=(), stagger=0.0, wander=0.0, range_m=1000.0, hurst=0.75,
               floor_m=None, cell=None, relief_sd=0.0, relief_range=2000.0, mound=None, nose=None, column=None):
    """Build one stratigraphic trap on a plane monocline: the arguments for :func:`resmill.export.to_grdecl`
    (``**result["kwargs"]``) and what was drawn and expected (``result["meta"]``). ``kind`` is one of :data:`KINDS`:

    * ``"pinchout"``: a sand wedge that thins to nothing updip. Every layer thins together (the thickness factor of
      :func:`resmill.structure.taper` as ``isochore``) and the columns where the sand has gone are inactive: walls.
      Its tongues hang from the bedding plane (the top is the plane, the base a bowl) or, with ``mound=True``, have a
      convex-up top on a flat base as a lens does: the top sinks toward the tongue's edge by T (1 - f), deepest (by the
      sand's thickness) at the tip. The closure is then that of the flat top to 2 m, the crest and the spill 2-10 m
      deeper (measured over dips of 0.5-1.1 degrees and tapers of 0.3-1.2): the sampler draws half of the tongues
      each way [J] (the owner, 2026-10-02: Sussex ridges have convex tops, Berea tongues are bedded).
    * ``"facies_change"``: the same, with a barrier zone, the last of ``thicknesses``, whose isochore is the
      complement of the sand's so that the interval keeps its thickness. The barrier's cells are the seal, by their
      capillary entry pressure (:func:`barrier_column`), under the sand and in a rim round it (see ``column``).
    * ``"lens"``: a lens of sand enclosed all round (by walls, or with ``barrier=True`` by a barrier zone). By
      default (``mound``: None means True for a lens, False for the others) a convex-up mound on a flat base, the base
      being the plane of the beds: the structure sinks by T (1 - f) where the sand thins, and a barrier zone below
      stays a flat slab (in its rim); ``mound=False`` hangs the lens from a flat top, convex down, as the fill of a
      channel is, the barrier taking the thickness the sand loses. A truncation, an onlap and the nosed kinds have no
      such top (``mound=True`` is refused).
    * ``"truncation"``: beds cut from above by an erosion surface that dips the same way less steeply, the older beds
      reaching farthest updip (``erode_above``: the top of the sand is the erosion surface in the subcrop strip, the
      sand's base is the bed's). ``taper_angle`` is the discordance, so it may not exceed ``dip``.
    * ``"onlap"``: layers that follow the top and end against an older surface that dips more steeply
      (``erode_below``), the younger layers reaching farthest updip.
    * ``"pinchout_nose"``, ``"truncation_nose"``: the same edges, a line, across the crest of a nose
      (a combination trap).

    ``x_len``, ``y_len`` (m) is the model, ``top`` the depth (m) of its stack's top at its centre, ``thicknesses`` the
    layers' thicknesses (m, top to bottom: the model's ``z_len``), ``dip`` the plane's dip (degrees) deepening along
    ``azimuth``'s normal (as :func:`resmill.structure.ramp`) and ``seed`` fixes the lobes, their places and the
    roughness. The sand's thickness ``T`` tapers to nothing over ``T / tan(taper_angle)`` m as the wedge of
    :func:`resmill.structure.taper`, at ``taper_angle`` on average and twice that at its edge (a wedge 10 m thick at
    0.3 degrees thins over 1.9 km: outcrop and field slopes of 0.006-0.6 degrees, step 6 research P9-P12); that is also
    the width of the subcrop strip of a truncation (the discordance) and the length over which layers onlap (the
    onlap angle). A lens tapers over that or its half-width, whichever is less, so that its middle is as thick as the
    sand.

    A straight updip line has no closure, so the line has a tongue of sand ``area`` (m2) and ``aspect`` (strike over
    dip length) protruding updip from it, a lobate half-ellipse (``warp``, :func:`resmill.structure.closure`): its
    length along dip is L = 2 sqrt(area / (pi aspect)) and the closure is tan(dip) L, derived, not drawn
    (``meta["closure_expected"]``): the oil spills over the sheet's updip edge, ``meta["spill_expected"]``.
    ``area=None`` gives a straight line, no closure. ``tongues`` are further tongues, a sequence of (area, aspect)
    that the caller draws (digitate ones have an aspect under 1), set side by side along the line in an order and with
    gaps drawn from ``seed``; a set that does not fit within 80 % of the model raises a ValueError. A lens has the
    area and aspect as a whole ellipse in the middle of the model. A pinch-out or facies change of several sand layers
    (``thicknesses``) interfingers in section if ``stagger`` (m) is given: each layer ends that much farther downdip
    than the one above, along the line and round every tongue, so the sand's top is the first layer's and the layers
    below it step back (the stacked, offset ridges of USGS DDS-33 fig. 8).

    The sand limit is irregular from ``floor_m`` up to ``range_m``: ``wander`` (m) is the rms
    displacement of the limit, of the line and of every tongue's outline, a :func:`resmill.structure.relief` surface
    of Hurst exponent ``hurst`` (the exponent of the edge's own structure function, which the test reads back; the
    realized edge's dimension is about 1.5 - hurst / 2 where the relief dominates, the 1.02-1.25 of coasts for 0.5-0.96,
    Kondev and Henley 1995, Mandelbrot 1967). It fades out over the taper, so the thick sand is smooth. More than
    about 0.2 of the main tongue's length, or of the taper for an erosional edge, breaks the sand into pieces that hold
    no trap. In a truncation or an onlap the wander is the relief of the erosion surface: tan(taper_angle) times it,
    ``meta["erosion_relief_m"]`` (None where there is no erosion surface); a valley of that depth preserves sand
    farther updip by its depth over the discordance. ``relief_sd`` (m) is the low-amplitude relief of the zone's top
    and base together, a :func:`resmill.structure.relief` surface of range ``relief_range``: the closure is that of
    the geometry with it. Neither relief is finer than two cells of the model, ``cell`` (m, its widest horizontal
    cell), unless ``floor_m`` says otherwise (a 32nd of the range at the finest, never under two cells): a sand edge
    that wanders on a scale below the cells opens and shuts necks between them, so that the closure the cells read
    would depend on the grid (outliers of 90 m in 2 of 24 draws with a floor of 31 m on cells of 100-200 m), and its
    octaves cost memory that nothing sees (6 GB for the plan's largest model). A relief without ``cell`` or
    ``floor_m`` is refused.

    A barrier zone needs ``column`` (m), the oil column it holds (:func:`barrier_column`): it lies under the sand and in
    a rim beyond its edge, ``meta["rim_m"]`` = :data:`RIM_SHARE` x column / tan(dip) wide, where it thins to nothing,
    and the columns beyond have no active cell, walls. A slab under the whole model, updip of the trap too, would be
    filled with oil by an initialisation by contacts (EQUIL puts oil in every cell above its entry pressure, joined to
    the trap or not): 84-100 % of the oil sat in it at the limit, and the shallowest barrier cell lay above the crest by
    more than the column in 37 of 40 draws, so that no contact kept it dry. With the rim the crest lies a quarter of the
    column, a cell's rise and the relief below the barrier's shallowest top, and :func:`trap_report` gives the contact
    that keeps the barrier dry.

    A combination trap takes its lateral closure from a ``nose``, the keywords of :func:`resmill.structure.closure`
    (``area``, ``height`` and ``aspect`` are required, ``height`` at most :data:`MAX_NOSE`; give no ``tilt``: the
    plane is the dip) centred where the line passes through its crest, so the trap closes by the nose's ``height``.

    The shapes sit on the dip direction through the middle of the model, as far updip as their rims fit within 80 % of
    the model's size (a lens in the middle); a shape that does not fit, or lobes that the warp pushes within 2 % of the
    edge, raise a ValueError for the caller to draw again. ``meta["net_layers"]`` counts the layers that are sand, for
    :func:`trap_report`. ``meta["closure_expected"]`` is that of the main tongue with a smooth edge and a flat top; with
    a rough edge, tongues, relief or a mound the closure is what the geometry has, which :func:`trap_report` reads.
    """
    nosed, lens = kind in NOSED, kind == "lens"
    thicknesses = [float(t) for t in thicknesses]
    _check_numbers(dip=dip, area=area, aspect=aspect, warp=warp, stagger=stagger, wander=wander, range_m=range_m,
                   floor_m=floor_m, cell=cell, relief_sd=relief_sd, relief_range=relief_range, column=column)
    mound = lens if mound is None else bool(mound)        # a lens is a mound unless it is the fill of a channel
    n_net = _check(kind, barrier, column, mound, dip, taper_angle, area, tongues, thicknesses, stagger, nose)
    t_sand = sum(thicknesses[:n_net])
    taper_m = t_sand / np.tan(np.radians(taper_angle))
    az = np.radians(azimuth)
    dip_dir, strike = np.array([np.sin(az), np.cos(az)]), np.array([np.cos(az), -np.sin(az)])
    centre = np.array([0.5 * x_len, 0.5 * y_len])
    ss = np.random.SeedSequence(seed).spawn(5)
    parts, lengths, sizes = _shapes(nose, area, aspect, tongues, lens)
    _check_wander(kind, wander, lengths[0], taper_m)
    offsets = _packed([2.0 * s[2] for s in sizes], np.random.default_rng(ss[0]))
    length = lengths[0]
    at = centre + _line_shift(sizes, offsets, x_len, y_len, dip_dir, strike, lens) * dip_dir
    fold, outline, line = _footprint(nose, parts, offsets, at, dip_dir, strike, azimuth, warp, lens, (ss[3], ss[4]))
    structure = st.ramp(dip, azimuth, center=tuple(centre))
    structure = structure if fold is None else structure + fold
    plan = _plan(x_len, y_len)
    margin = [max(EDGE_MARGIN * size, 2.0 * (cell or 0.0)) for size in (x_len, y_len)]           # two cells at least
    if fold is not None or outline is not None:
        _refuse_edge(fold if nosed else outline, lens, nosed, line, dip_dir, plan, margin)
    floors = _floors(wander, range_m, relief_sd, relief_range, floor_m, cell)
    if relief_sd:
        structure = structure + st.relief(relief_sd, relief_range, x_len, y_len, hurst, floors[1], seed=ss[2])
    rough = st.relief(wander, range_m, x_len, y_len, hurst, floors[0], seed=ss[1]) if wander else None
    if lens and mound:
        taper_m = min(taper_m, 0.5 * lengths[0] * min(1.0, aspect))                       # a whole mound
    f, layers_f = _thickness(line, outline, taper_m, azimuth, x_len, y_len, rough, stagger, n_net)
    sink = st.Structure(lambda x, y: t_sand * (1.0 - f(x, y)))        # what the top lies below the base's plane
    if mound:
        structure = structure + sink                                 # the base stays flat: the top is the mound's
    rim_m = RIM_SHARE * column / np.tan(np.radians(dip)) if barrier else None
    kwargs = dict(structure=structure)
    if kind.startswith("truncation"):                                  # the sand cut from above: the top is the surface
        kwargs["erode_above"] = st.Structure(lambda x, y: top + structure(x, y) + sink(x, y))
    elif kind == "onlap":                                              # the layers cut from below: the base is
        kwargs["erode_below"] = st.Structure(lambda x, y: top + structure(x, y) + t_sand * f(x, y))
    else:
        kwargs["isochore"] = layers_f + ([_barrier(thicknesses, layers_f, mound, rim_m, line, outline,
                                                   azimuth, x_len, y_len, rough)] if barrier else [])
    nominal = nose["height"] if nosed else np.tan(np.radians(dip)) * length
    if wander or relief_sd:                                            # the roughness may have destroyed the trap
        gx, gy = plan
        surface = kwargs["erode_above"] if "erode_above" in kwargs else st.Structure(
            lambda x, y: top + structure(x, y))
        _refuse_collapse(thicknesses[0] * f(gx, gy) > MIN_THICKNESS, surface(gx, gy), nominal, plan, margin)
    tan_dip, along = np.tan(np.radians(dip)), float(centre @ dip_dir)
    cut = t_sand if kind.startswith("truncation") else 0.0              # a truncation's top is its sand's base
    level = top + tan_dip * ((along if lens else line) - along) + cut   # the depth of the line (of the middle)
    meta = dict(kind=kind, x_len=x_len, y_len=y_len, top=top, thicknesses=thicknesses, dip=dip, azimuth=azimuth,
                taper_angle=taper_angle, taper_m=taper_m, thickness=t_sand,
                area=None if nosed else area, aspect=None if nosed else aspect, warp=warp, stagger=stagger,
                wander=wander, range_m=range_m, hurst=hurst, floor_m=floors[0], cell=cell, relief_sd=relief_sd,
                mound=mound,
                tongues=[dict(area=a, aspect=r, length=n, offset=float(v)) for (a, r), n, v in
                         zip(parts, lengths, offsets)] if area is not None and not nosed else [],
                length=length, line=line, nose=nose, barrier=bool(barrier), column=column, rim_m=rim_m, net_layers=n_net, seed=seed,
                erosion_relief_m=np.tan(np.radians(taper_angle)) * wander if kind in ("truncation", "onlap",
                                                                                      "truncation_nose") else None,
                spill_expected=None if lens else level,
                closure_expected=nominal,
                crest_expected=level - (nose["height"] if nosed else tan_dip * (length if not lens else 0.5 * length)))
    return dict(kwargs=kwargs, meta=meta)


def trap_report(model, built):
    """The traps of the sand of ``model`` (a layer, a reservoir or a list of layers) as :func:`strat_trap` shaped it
    (:func:`zone_trap`, the surface being the top of the sand's cells: a barrier zone is no part of it), the one with
    the most closure first: a rough edge leaves pieces of sand that hold small traps of their own, some of them
    shallower than the main one. With a barrier zone the limit is the admissible one: the barrier holds ``column`` (what
    ``built`` was given) below the shallower of the crest and the top of its own shallowest cell, ``barrier_top``, so
    that a contact at ``limit_depth`` puts oil in no barrier cell."""
    layers = list(model) if isinstance(model, (list, tuple)) else list(getattr(model, "layers", [model]))
    meta = built["meta"]
    made = [meta["x_len"], meta["y_len"], meta["top"], *meta["thicknesses"]]
    given = [layers[0].x_len, layers[0].y_len, layers[0].top_depth, *(layer.z_len for layer in layers)]
    if len(made) != len(given) or not np.allclose(made, given) or any(layer.dip for layer in layers):
        raise ValueError(f"the layers (size {given[:2]}, top {given[2]}, thicknesses {given[3:]}, flat) are not the "
                         f"ones the trap was built for ({made[:2]}, {made[2]}, {made[3:]}): a model's own dip or "
                         f"thickness changes the trap")
    _, _, zc, act = _build_geometry(layers, **built["kwargs"])
    cells = sum(layer.nz for layer in layers[:meta["net_layers"]])
    traps = zone_trap(zc, act, layers[0].dx, layers[0].dy, k=slice(0, cells), column=meta["column"],
                      barrier=slice(cells, None) if meta["barrier"] else None)
    return sorted(traps, key=lambda trap: -trap["height"])
