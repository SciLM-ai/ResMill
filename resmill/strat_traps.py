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

from . import structure as st
from .export import _build_geometry
from .structure import _spill_levels

KINDS = ("pinchout", "facies_change", "lens")

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


def strat_trap(kind, x_len, y_len, top, thicknesses, seed, barrier=False, dip=1.0, azimuth=0.0, taper_angle=0.3,
               area=3.4e6, aspect=2.2, warp=0.3, wander=0.0, range_m=1000.0):
    """Build one stratigraphic trap on a plane monocline: the arguments for :func:`resmill.export.to_grdecl`
    (``**result["kwargs"]``) and what was drawn and expected (``result["meta"]``). ``kind`` is one of :data:`KINDS`:

    * ``"pinchout"``: a sand wedge that thins to nothing updip. Every layer thins together (the thickness factor of
      :func:`resmill.structure.taper` as ``isochore``) and the columns where the sand has gone are inactive: walls.
    * ``"facies_change"``: the same, with a barrier zone, the last of ``thicknesses``, whose isochore is the
      complement of the sand's so that the interval keeps its thickness. The barrier's cells stay active and are the
      seal, by their capillary entry pressure (:func:`barrier_column`).
    * ``"lens"``: a lens of sand enclosed all round (by walls, or with ``barrier=True`` by a barrier zone).

    ``x_len``, ``y_len`` (m) is the model, ``top`` the depth (m) of its stack's top at its centre, ``thicknesses`` the
    layers' thicknesses (m, top to bottom: the model's ``z_len``), ``dip`` the plane's dip (degrees) deepening along
    ``azimuth``'s normal (as :func:`resmill.structure.ramp`) and ``seed`` fixes the lobes and the wander. The sand's
    thickness ``T`` tapers to nothing over ``T / tan(taper_angle)`` m (a wedge 10 m thick at 0.3 degrees thins over
    1.9 km: outcrop and field slopes of 0.006-0.6 degrees, step 6 research P9-P12).

    A straight updip line has no closure, so the line has a tongue of sand ``area`` (m2) and ``aspect`` (strike over
    dip length) protruding updip from it, a lobate half-ellipse (``warp``, :func:`resmill.structure.closure`), the
    line wandering ``wander`` m rms with correlation ``range_m`` (``area=None``: a straight line, no closure). The
    tongue's length along dip is L = 2 sqrt(area / (pi aspect)) and the closure is tan(dip) L, derived, not drawn
    (``meta["closure_expected"]``): the oil spills over the sheet's updip edge, ``meta["spill_expected"]``. A lens has
    that area and aspect as a whole ellipse in the middle of the model and closes by tan(dip) times its length along
    dip, filling to its deepest point. ``meta["net_layers"]`` counts the layers that are sand, for
    :func:`trap_report`.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    if not 0.0 < taper_angle < 90.0:
        raise ValueError(f"taper_angle must lie between 0 and 90 degrees, not {taper_angle}")
    if bool(barrier) != (kind == "facies_change") and not (kind == "lens" and barrier):
        raise ValueError(f"a barrier zone is what a facies change needs and a lens may have, not a {kind}")
    if kind == "lens" and area is None:
        raise ValueError("a lens needs an area")
    thicknesses = [float(t) for t in thicknesses]
    n_net = len(thicknesses) - bool(barrier)
    if n_net < 1:
        raise ValueError("thicknesses needs a sand layer as well as the barrier")
    t_sand = sum(thicknesses[:n_net])
    taper_m = t_sand / np.tan(np.radians(taper_angle))
    az = np.radians(azimuth)
    dip_dir, strike = (np.sin(az), np.cos(az)), (np.cos(az), -np.sin(az))
    centre = (0.5 * x_len, 0.5 * y_len)
    half = 0.5 * (abs(x_len * dip_dir[0]) + abs(y_len * dip_dir[1]))          # half the model's extent along dip
    half_strike = 0.5 * (abs(x_len * strike[0]) + abs(y_len * strike[1]))
    lens = kind == "lens"
    length = 0.0 if area is None else 2.0 * np.sqrt(area / (np.pi * aspect))     # the trap's length along dip
    shift = 0.0 if lens else -0.8 * half + length                      # the line, from the centre along dip
    if 0.5 * aspect * length > 0.95 * half_strike or (length > 1.8 * half if lens else shift > 0.8 * half):
        raise ValueError(f"the trap (length {length:.0f} m along dip, {aspect * length:.0f} m along strike) does "
                         f"not fit the model: make the model larger or the area smaller")
    seeds = [int(v) for v in np.random.default_rng(seed).integers(2 ** 31, size=2)]
    at = (centre[0] + shift * dip_dir[0], centre[1] + shift * dip_dir[1])
    outline = None if area is None else st.closure(
        area if lens else 2.0 * area, 1.0, aspect=aspect if lens else 0.5 * aspect, azimuth=azimuth, center=at,
        warp=warp, seed=seeds[0])
    line = None if lens else at[0] * dip_dir[0] + at[1] * dip_dir[1]
    f = st.taper(line, taper_m, azimuth, wander=0.0 if lens else wander, range_m=range_m, seed=seeds[1],
                 outline=outline, x_len=x_len, y_len=y_len)
    ratio = t_sand / thicknesses[-1]
    isochore = [f] * n_net + ([st.Structure(lambda x, y: 1.0 + ratio * (1.0 - f(x, y)))] if barrier else [])
    tan_dip = np.tan(np.radians(dip))
    meta = dict(kind=kind, dip=dip, azimuth=azimuth, taper_angle=taper_angle, taper_m=taper_m, thickness=t_sand,
                area=area, aspect=aspect, warp=warp, wander=wander, range_m=range_m, length=length, line=line,
                barrier=bool(barrier), net_layers=n_net, closure_expected=tan_dip * length, seed=seed,
                crest_expected=top + tan_dip * (shift - (0.0 if lens else length) - (0.5 * length if lens else 0.0)),
                spill_expected=None if lens else top + tan_dip * shift)
    return dict(kwargs=dict(structure=st.ramp(dip, azimuth, center=centre), isochore=isochore), meta=meta)


def trap_report(model, built, column=None):
    """The traps of the sand of ``model`` (a layer, a reservoir or a list of layers) as :func:`strat_trap` shaped it
    (:func:`zone_trap`, the surface being the top of the sand's cells: a barrier zone is no part of it)."""
    layers = list(model) if isinstance(model, (list, tuple)) else list(getattr(model, "layers", [model]))
    _, _, zc, act = _build_geometry(layers, **built["kwargs"])
    cells = sum(layer.nz for layer in layers[:built["meta"]["net_layers"]])
    return zone_trap(zc, act, layers[0].dx, layers[0].dy, k=slice(0, cells), column=column)
