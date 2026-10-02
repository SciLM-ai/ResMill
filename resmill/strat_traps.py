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


def _traps(depth, dx, dy, column=None):
    """The traps of a map of depths (nx, ny), infinite where the zone is absent (:func:`zone_trap`, whose result it is,
    on any such map: the top of a zone's cells, or an analytic surface)."""
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


def zone_trap(zc, act, dx, dy, k=None, column=None):
    """The traps of the top of the active cells of a zone, absent columns being walls and the map's edge the only exit.

    ``zc`` (2nx, 2ny, nk + 1) is the interface stack and ``act`` (nx, ny, nk) the active cells, both k top-down as
    :func:`resmill.export._build_geometry` returns them; ``k`` (a slice of the layers, default all) picks the zone
    that is net reservoir, so that a barrier zone of active cells under or beside it is not part of the surface
    (:func:`zone_top`). ``column`` (m) is the oil column the updip seal holds if that is capillary; a trap holds no more
    than that below its crest.

    Returns one dict per body of connected columns that has any, the shallowest crest first: ``crest`` (i, j),
    ``crest_depth``, ``spill_depth`` (the deepest top on the best way out, None when nothing reaches the body:
    sealed all round), ``spill_point`` (the column beside the trap the oil leaves into, None when sealed),
    ``limit_depth`` (the deepest contact the trap holds: the spill, the deepest top of a sealed body, or ``column``
    below the crest, whichever is shallowest), ``limited_by`` ("spill", "sealed" or "barrier"), ``height``
    (``limit_depth`` - ``crest_depth``), and the ``area`` (m2) and ``mask`` of the columns shallower than the limit
    that join the crest (all of a sealed body: it fills to its deepest point). A body with no closure has height 0.
    """
    return _traps(zone_top(zc, act, k), dx, dy, column)


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


def strat_trap(kind, x_len, y_len, top, thicknesses, seed, barrier=False, dip=1.0, azimuth=0.0, taper_angle=0.3,
               area=3.4e6, aspect=2.2, warp=0.3, tongues=(), stagger=0.0, wander=0.0, range_m=1000.0, hurst=0.75,
               floor_m=None, relief_sd=0.0, relief_range=2000.0, mound=True, nose=None):
    """Build one stratigraphic trap on a plane monocline: the arguments for :func:`resmill.export.to_grdecl`
    (``**result["kwargs"]``) and what was drawn and expected (``result["meta"]``). ``kind`` is one of :data:`KINDS`:

    * ``"pinchout"``: a sand wedge that thins to nothing updip. Every layer thins together (the thickness factor of
      :func:`resmill.structure.taper` as ``isochore``) and the columns where the sand has gone are inactive: walls.
    * ``"facies_change"``: the same, with a barrier zone, the last of ``thicknesses``, whose isochore is the
      complement of the sand's so that the interval keeps its thickness. The barrier's cells stay active and are the
      seal, by their capillary entry pressure (:func:`barrier_column`).
    * ``"lens"``: a lens of sand enclosed all round (by walls, or with ``barrier=True`` by a barrier zone). By
      default (``mound``) a convex-up mound on a flat base, the base being the plane of the beds: the structure sinks
      by T (1 - f) where the sand thins, and a barrier zone below stays a flat slab; ``mound=False`` hangs the lens
      from a flat top, convex down, as the fill of a channel is, the barrier taking the thickness the sand loses.
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

    The sand limit is irregular from ``floor_m`` up to ``range_m`` (1/32 of it unless given): ``wander`` (m) is the rms
    displacement of the limit, of the line and of every tongue's outline, a :func:`resmill.structure.relief` surface
    of Hurst exponent ``hurst`` (the exponent of the edge's own structure function, which the test reads back; the
    realized edge's dimension is about 1.5 - hurst / 2 where the relief dominates, the 1.02-1.25 of coasts for 0.5-0.96,
    Kondev and Henley 1995, Mandelbrot 1967). It fades out over the taper, so the thick sand is smooth. More than
    about 0.2 of the main tongue's length, or of the taper for an erosional edge, breaks the sand into pieces that hold
    no trap. In a truncation or an onlap the wander is the relief of the erosion surface: tan(taper_angle) times it,
    ``meta["erosion_relief_m"]`` (a valley of that depth preserves sand farther updip by its depth over the
    discordance). ``relief_sd`` (m) is the low-amplitude relief of the zone's top and base together, a
    :func:`resmill.structure.relief` surface of range ``relief_range``: the closure is that of the geometry with it.

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
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    if not 0.0 < taper_angle < 90.0:
        raise ValueError(f"taper_angle must lie between 0 and 90 degrees, not {taper_angle}")
    if bool(barrier) != (kind == "facies_change") and not (lens and barrier):
        raise ValueError(f"a barrier zone is what a facies change needs and a lens may have, not a {kind}")
    if lens and area is None:
        raise ValueError("a lens needs an area")
    if nosed != (nose is not None):
        raise ValueError(f"a nose (its area, height and aspect) is what {', '.join(NOSED)} need, not a {kind}")
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
    thicknesses = [float(t) for t in thicknesses]
    n_net = len(thicknesses) - bool(barrier)
    if n_net < 1:
        raise ValueError("thicknesses needs a sand layer as well as the barrier")
    if stagger and (kind not in ("pinchout", "facies_change") or n_net < 2):
        raise ValueError(f"a stagger (the sand layers ending each farther downdip) needs a pinch-out or a facies "
                         f"change of more than one sand layer, not a {kind} of {n_net}")
    t_sand = sum(thicknesses[:n_net])
    taper_m = t_sand / np.tan(np.radians(taper_angle))
    az = np.radians(azimuth)
    dip_dir, strike = np.array([np.sin(az), np.cos(az)]), np.array([np.cos(az), -np.sin(az)])
    centre = np.array([0.5 * x_len, 0.5 * y_len])
    with np.errstate(divide="ignore"):                       # the model's half-length along dip, through its centre
        half = float(min(0.5 * x_len / abs(dip_dir[0]), 0.5 * y_len / abs(dip_dir[1])))
    ss = np.random.SeedSequence(seed).spawn(5)
    # the shapes: how far each reaches updip and downdip of its line (of its middle, for a lens), half across it, and
    # where along the line it sits
    if nosed:
        reach = np.sqrt(nose["area"] / (np.pi * nose["aspect"]))             # the nose's half-length along dip
        sizes = [(reach, reach, nose["aspect"] * reach)]
        lengths = [0.0]
    elif area is None:
        sizes, lengths = [(0.0, 0.0, 0.0)], [0.0]                            # a straight line: a point to fit
    else:
        parts = [(float(area), float(aspect))] + [(float(a), float(r)) for a, r in tongues]
        lengths = [2.0 * np.sqrt(a / (np.pi * r)) for a, r in parts]         # each trap's length along dip
        sizes = [(0.5 * n, 0.5 * n, 0.5 * r * n) if lens else (n, 0.0, 0.5 * r * n)
                 for n, (_, r) in zip(lengths, parts)]
    offsets = _packed([2.0 * s[2] for s in sizes], np.random.default_rng(ss[0]))
    length = lengths[0]
    t = np.linspace(0.0, 2.0 * np.pi, 73)                                      # the rim of a shape, its base too

    def fits(shift):                                   # the rims, the line ``shift`` m from the centre along dip
        for (up, down, wide), v in zip(sizes, offsets):
            rim = (shift + np.where(np.cos(t) < 0.0, up, down) * np.cos(t))[:, None] * dip_dir \
                + (v + wide * np.sin(t))[:, None] * strike
            if not (np.abs(rim) <= 0.4 * np.array([x_len, y_len])).all():
                return False
        return True

    grid = np.linspace(-half, half, 401)               # the line goes as far updip as the shapes fit (a lens: centred)
    first = [0] if lens else [k for k, shift in enumerate(grid) if fits(shift)][:1]
    if not first or (lens and not fits(0.0)):
        raise ValueError(f"the trap ({sum(max(u + d, 2 * w) for u, d, w in sizes[:1]):.0f} m across, with "
                         f"{max(len(sizes) - 1, 0)} more tongues) does not fit the model: make the model larger or "
                         f"the area smaller")
    shift = 0.0
    if not lens:
        lo, shift = grid[max(first[0] - 1, 0)], grid[first[0]]
        for _ in range(40):                            # the fitting shifts are an interval: bisect to its updip end
            mid = 0.5 * (lo + shift)
            lo, shift = (lo, mid) if fits(mid) else (mid, shift)
    at = centre + shift * dip_dir
    ramp = st.ramp(dip, azimuth, center=tuple(centre))
    outline, line, structure = None, None, ramp
    if nosed:
        fold = st.closure(azimuth=azimuth, center=tuple(at), seed=ss[3], **nose)
        structure, line = ramp + fold, float((at + fold.crest_offset) @ dip_dir)
    elif area is not None:
        outlines = [st.closure(a if lens else 2.0 * a, 1.0, aspect=r if lens else 0.5 * r, azimuth=azimuth,
                               center=tuple(at + v * strike), warp=warp, seed=k)
                    for (a, r), v, k in zip(parts, offsets, ss[4].spawn(len(parts)))]
        outline = outlines[0] if len(outlines) == 1 else st.Structure(
            lambda x, y: np.minimum.reduce([o(x, y) for o in outlines]))
    if not lens:
        line = float(at @ dip_dir) if line is None else line
    gx, gy = np.meshgrid(np.linspace(0.0, x_len, 301), np.linspace(0.0, y_len, 301), indexing="ij")
    foot = fold if nosed else outline
    if foot is not None:
        inside, along_dip = foot(gx, gy) < 0.0, gx * dip_dir[0] + gy * dip_dir[1]
        edge = (np.minimum(gx, x_len - gx) < 0.02 * x_len) | (np.minimum(gy, y_len - gy) < 0.02 * y_len)
        watched = edge if lens else edge & (along_dip > line if nosed else along_dip < line)   # where the trap is
        if (watched & inside).any():
            raise ValueError("the lobes of the trap come within 2 % of the model's edge, over which it would leak: use "
                             "another seed, a smaller warp or a larger model")
    if relief_sd:
        structure = structure + st.relief(relief_sd, relief_range, x_len, y_len, hurst, floor_m, seed=ss[2])
    rough = st.relief(wander, range_m, x_len, y_len, hurst, floor_m, seed=ss[1]) if wander else None
    if lens:
        taper_m = min(taper_m, 0.5 * lengths[0] * min(1.0, aspect)) if mound else taper_m   # a whole mound
    def factor(shift):                                     # the thickness factor of a layer ending ``shift`` m downdip
        moved = outline if not shift else st.Structure(
            lambda x, y: outline(x - shift * dip_dir[0], y - shift * dip_dir[1]))
        return st.taper(None if line is None else line + shift, taper_m, azimuth, outline=moved, x_len=x_len,
                        y_len=y_len, edge=rough)

    f = factor(0.0)
    layers_f = [f] + [factor(k * stagger) for k in range(1, n_net)] if stagger else [f] * n_net
    sink = st.Structure(lambda x, y: t_sand * (1.0 - f(x, y)))        # what the top lies below the base's plane
    if lens and mound:
        structure = structure + sink                                 # the base stays flat: the top is the mound's
    kwargs = dict(structure=structure)
    if kind.startswith("truncation"):                                  # the sand cut from above: the top is the surface
        kwargs["erode_above"] = st.Structure(lambda x, y: top + structure(x, y) + sink(x, y))
    elif kind == "onlap":                                              # the layers cut from below: the base is
        kwargs["erode_below"] = st.Structure(lambda x, y: top + structure(x, y) + t_sand * f(x, y))
    else:
        complement = None if lens and mound else st.Structure(      # the barrier takes what the sand loses
            lambda x, y: 1.0 + sum(t * (1.0 - g(x, y)) for t, g in zip(thicknesses, layers_f)) / thicknesses[-1])
        kwargs["isochore"] = layers_f + ([complement] if barrier else [])
    tan_dip, along = np.tan(np.radians(dip)), float(centre @ dip_dir)
    cut = t_sand if kind.startswith("truncation") else 0.0              # a truncation's top is its sand's base
    level = top + tan_dip * ((along if lens else line) - along) + cut   # the depth of the line (of the middle)
    meta = dict(kind=kind, dip=dip, azimuth=azimuth, taper_angle=taper_angle, taper_m=taper_m, thickness=t_sand,
                area=None if nosed else area, aspect=None if nosed else aspect, warp=warp, stagger=stagger,
                wander=wander, range_m=range_m, hurst=hurst, floor_m=floor_m, relief_sd=relief_sd,
                mound=bool(lens and mound),
                tongues=[dict(area=a, aspect=r, length=n, offset=float(v)) for (a, r), n, v in
                         zip(parts, lengths, offsets)] if area is not None and not nosed else [],
                length=length, line=line, nose=nose, barrier=bool(barrier), net_layers=n_net, seed=seed,
                erosion_relief_m=np.tan(np.radians(taper_angle)) * wander,
                spill_expected=None if lens else level,
                closure_expected=nose["height"] if nosed else tan_dip * length,
                crest_expected=level - (nose["height"] if nosed else tan_dip * (length if not lens else 0.5 * length)))
    return dict(kwargs=kwargs, meta=meta)


def trap_report(model, built, column=None):
    """The traps of the sand of ``model`` (a layer, a reservoir or a list of layers) as :func:`strat_trap` shaped it
    (:func:`zone_trap`, the surface being the top of the sand's cells: a barrier zone is no part of it). With a barrier
    zone each trap also has ``barrier_top``, the depth of the top of the barrier's shallowest cell (None otherwise): an
    initialisation by contacts (EQUIL) puts oil in every cell above the contact whose capillary pressure exceeds its
    entry pressure, joined to the trap or not, so a barrier holds its ``column`` below that top, not below the crest, if
    it reaches updip of it; cut the model's outline there, or keep the contact above ``barrier_top + column``."""
    layers = list(model) if isinstance(model, (list, tuple)) else list(getattr(model, "layers", [model]))
    _, _, zc, act = _build_geometry(layers, **built["kwargs"])
    cells = sum(layer.nz for layer in layers[:built["meta"]["net_layers"]])
    traps = zone_trap(zc, act, layers[0].dx, layers[0].dy, k=slice(0, cells), column=column)
    top = float(zone_top(zc, act, slice(cells, None)).min()) if built["meta"]["barrier"] else None
    return [dict(trap, barrier_top=top) for trap in sorted(traps, key=lambda trap: -trap["height"])]
