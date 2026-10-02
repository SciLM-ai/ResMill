"""Tilted fault blocks and rollovers on listric growth faults (structure step 4).

Two trap styles, each returning a :class:`BlockModel` (the structure, the faults, the growth isochores and every drawn
value) that ``to_grdecl`` takes as it is. Every argument left ``None`` is drawn from its range with
``numpy.random.default_rng(seed)``; a value pins it (explicit mode), and pinning one argument leaves the others as drawn.
Ranges are those of ``design_notes/structure_research/step4_blocks_rollovers.md`` (rows T1-T28); values tagged [J] are
judgement values, labelled so. A trap is measured, never fitted: its area and relief are read off the faulted top, every
fault taken as sealing (``labels["trap"]``; ``labels["structure_trap"]`` is the one before the faults inside it). The model goes
to the exporter as it is, one layer per zone for a rollover's ``zones``::

    m = rollover(16000.0, 12000.0, 100.0, 2500.0, 120.0, seed=7, zones=3)       # the top at 2,500 m, 120 m thick
    to_grdecl(reservoir, "m.grdecl", structure=m.structure, faults=m.faults, isochore=m.isochore)

:func:`tilted_blocks`: a domino system, N parallel planar faults dipping 25-35 degrees whose hanging walls lie on the up-dip
side of beds tilted 3-20 degrees (10 % of cases up to 30). Each block is a rigid plate: the structure is a tilt, and each fault
displaces the whole of one side (``hw_share`` 0.5, no drag) by the column offset ``tan(tilt) x`` (fault spacing), so that the
horizon is a sawtooth with no net dip: blocks ``width`` wide at the horizon, a throw (cutoff to cutoff along the plane) of
``width tan(tilt)``, a heave of throw / tan(dip) and an extension beta = 1 + tan(tilt) / tan(dip) of the blocks' horizontal
widths, sin(alpha + theta) / sin(theta) of their lengths along the beds (Fossen & Hesthammer 1998; T6, T9). Tilt and throw taper
along the strike like the faults' tip ellipse, so the blocks plunge toward the fault tips and each footwall crest closes
laterally. ``kind="horst_graben"``: alternating hanging walls, dips 55-70 degrees, a tilt under 5 degrees (T11). Transfer
faults (0-2) cross the blocks; the faults inside the trap are drawn by :func:`resmill.fault_patterns.fold_faults`.

:func:`rollover`: a growth fault whose hanging wall rolls over toward it. ``kind="frio"`` (60 % [J]): one sinuous listric
master fault (a ramp dipping 50-75 degrees, then tan(dip) falling by 1/e every 1.2-5 km of depth,
:attr:`resmill.faults.Fault.flatten`: the shapes of six published faults, whose fits give 0.6-3.1 km), the ramp ending 0-2 km
above the reservoir (the published bends lie 1.0-2.7 km down, so the cutoff is on the curved part and the roll starts at it),
maximum displacement (at the tip ellipse's centre, as ``fold_faults`` draws a fault's) from the clastic displacement-length law, a
regional dip toward the basin; ``kind="wilcox"`` (40 % [J]): two or three
nearly straight, closely spaced planar faults with little rollover and a higher expansion (Ewing et al. 1986). The strata laid
down while the faults moved thicken into their hanging walls by an expansion index of 1.1-2.5
(:func:`resmill.structure.growth`). The faults inside the trap, with the keystone graben half the time and 40-70 % antithetic,
come from ``fold_faults``. The vertical shear that rolls the hanging wall over a listric fault (one heave for the whole block, so
that no zone changes thickness and the growth strata are the isochore's alone) drags it down by the throw at the fault and lets it
rise as the plane flattens, toward the regional dip: the crest, where the roll's dip falls to the regional dip, lies a few km from
the fault (the throw and the regional dip set it, the flattening length hardly), and a draw whose masters leave no trap of 1 km2
(the P10 of the Gulf's rollover traps) is drawn again (:func:`rollover`).
"""
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

from . import structure as st
from .export import _build_geometry
from .fault_patterns import fold_faults
from .faults import Fault, _plane, ww_profile
from .layers.base import Layer
from .structure import Structure, _spill_levels

PLAN_CELLS = 300                    # the planning grid (a trap is measured and framed on it) has at most this many cells a side
PLAN_THICKNESS = 4.0                # m: the layer a trap is measured on
TRIES = 32                          # a rollover whose masters leave no trap to frame its faults on is redrawn this often
MIN_AREA = 1.0e6                    # m2: the trap that frames them is at least the P10 of the Gulf's rollover traps (BOEM, T25: 1.4 km2)
MIN_CELLS = 9                       # and spans at least 3 x 3 planning cells, the least a fold can frame faults on
TIP_ASPECT = 2.15                   # tip-line length / height of a fault (Nicol et al. 1996; Fault.aspect's default)
D_OVER_L = 0.1                      # upper bound of displacement / length (Lathrop et al. 2022)
RAMP_DIP = (50.0, 75.0)             # a Frio master's dip down to its bend, degrees: uniform (T14 50-60; six published faults 46-77) [J]
FLATTEN = (1200.0, 5000.0)          # m: depth over which its tan(dip) falls by 1/e below the bend, log-uniform; the median, 2.45 km, is
#                                     the published one (six faults fit 0.6-3.1 km; Bruce 1973: 60 to 15 degrees over about 4 km is 2.2) [J]
RAMP_BASE = (0.0, 2000.0)           # m above the reservoir where that ramp ends, uniform (the six faults bend 1.0-2.7 km down, median
#                                     1.9, the Frio sands they cut lie at 1-3.5 km) [J]


@dataclass
class BlockModel:
    """A block-style model: what ``to_grdecl`` takes (``structure``, ``faults``, ``isochore``) and what was drawn.

    ``isochore`` is one growth thickness-factor field per zone (or None); ``labels`` holds every drawn value, the faults'
    counts and the measured traps, plain numbers for the episode record.
    """

    structure: Structure
    faults: list
    isochore: list | None
    labels: dict = field(default_factory=dict)


def _log_uniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _centre_dip(fault):
    """The dip (degrees) of ``fault``'s plane at its tip ellipse's centre: ``dip`` for a planar fault, and for a listric one tan(dip)
    has fallen by 1/e per ``flatten`` m of depth below the ramp's base."""
    base = fault.z_center if fault.ramp_base is None else fault.ramp_base
    below = 0.0 if fault.flatten is None else max(fault.z_center - base, 0.0) / fault.flatten
    return math.degrees(math.atan(math.tan(math.radians(fault.dip)) * math.exp(-below)))


def _axes(azimuth):
    """The dip direction n (package convention: the ramp deepens along it) and the strike direction t of a fault set."""
    a = math.radians(azimuth)
    return np.array([math.sin(a), math.cos(a)]), np.array([math.cos(a), -math.sin(a)])


def _azimuth_of(v):
    """The ``fold_faults`` azimuth (degrees clockwise from +x, the direction (cos a, -sin a)) of the vector ``v``."""
    return float(-math.degrees(math.atan2(v[1], v[0])) % 360.0)


def _measure(x_len, y_len, dx, top, structure, faults):
    """The trap of largest volume on the top of the model, every fault sealing, or None.

    The map is the depth of the top (mean of a cell's four corners) on cells of about ``dx``, from a layer 4 m thick, so
    that the faults' planes cut the layer only where they cut the horizon. Where a plane cuts a cell, or the cell lies in
    the cut-out between a hanging-wall and a footwall cutoff, a corner has lost part of its thickness and the cell's top is
    the fault plane, not the horizon: such a cell (a corner thinner than 0.8 of the layer) holds no rock here, and bounds
    nothing. A cell is in a trap where its depth lies above its spill depth (the deepest point of the shallowest path to the
    map's edge that crosses no fault: :func:`resmill.structure._spill_levels`); a trap is a connected set of such cells.
    Returns ``area`` (m2), ``height`` (m, the relief at the crest), ``crest`` (i, j) and its ``crest_xy`` (m),
    ``crest_depth``, ``spill_depth``, ``mask``, the cell size ``cell`` and the ``depth`` map (infinite where a cell holds no rock).
    """
    nx, ny = max(int(round(x_len / dx)), 3), max(int(round(y_len / dx)), 3)
    cell = (x_len / nx, y_len / ny)
    faces = []
    _, _, zc, _ = _build_geometry([Layer(nx, ny, 1, x_len, y_len, PLAN_THICKNESS, top_depth=top)], structure=structure,
                                  faults=faults, _faces=faces)
    z = zc[:, :, 0]
    thin = (zc[:, :, 1] - z).reshape(nx, 2, ny, 2).min(axis=(1, 3))
    depth = np.where(thin >= 0.8 * PLAN_THICKNESS, 0.25 * (z[0::2, 0::2] + z[1::2, 0::2] + z[0::2, 1::2] + z[1::2, 1::2]),
                     np.inf)
    wx, wy = np.zeros((nx - 1, ny), dtype=bool), np.zeros((nx, ny - 1), dtype=bool)
    for _, side in faces:
        s0 = side[:, :, 0]
        wx |= s0[:-1] * s0[1:] == -1
        wy |= s0[:, :-1] * s0[:, 1:] == -1
    spill = _spill_levels(depth, wx, wy)
    relief = np.zeros(depth.shape)
    np.subtract(spill, depth, out=relief, where=np.isfinite(spill) & np.isfinite(depth))
    up = relief > 0.0
    if not up.any():
        return None
    idx = np.arange(nx * ny).reshape(nx, ny)
    ex, ey = up[:-1] & up[1:] & ~wx, up[:, :-1] & up[:, 1:] & ~wy
    rows = np.concatenate([idx[:-1][ex], idx[:, :-1][ey]])
    cols = np.concatenate([idx[1:][ex], idx[:, 1:][ey]])
    _, lab = csgraph.connected_components(sparse.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(nx * ny,) * 2),
                                          directed=False)
    lab = lab.reshape(nx, ny)
    best = int(np.argmax(np.bincount(lab[up], weights=relief[up], minlength=lab.max() + 1)))
    mask = up & (lab == best)
    crest = np.unravel_index(int(np.argmax(np.where(mask, relief, -1.0))), relief.shape)
    crest = (int(crest[0]), int(crest[1]))
    return dict(area=float(mask.sum()) * cell[0] * cell[1], height=float(relief[crest]), crest=crest, mask=mask, cell=cell,
                crest_xy=((crest[0] + 0.5) * cell[0], (crest[1] + 0.5) * cell[1]), crest_depth=float(depth[crest]),
                spill_depth=float(spill[crest]), depth=depth)


def _trap_label(trap):
    """A measured trap as plain numbers for the record (None when there is none)."""
    return None if trap is None else dict(
        area_km2=trap["area"] / 1e6, height_m=trap["height"], spill_depth_m=trap["spill_depth"],
        crest_depth_m=trap["crest_depth"], crest_xy_m=[float(v) for v in trap["crest_xy"]])


def _population(style, trap, x_len, y_len, dx, top, thickness, density, seed, regional, basinward):
    """The faults inside a block model's trap, from ``fold_faults`` on a smooth fold of that trap (none without a trap).

    The fold has the measured trap's area, relief, aspect and axis and its crest at the trap's real depth, so that the faults
    are drawn on a closure that frames it. The fold is smooth and flat around the trap, while the real top has the tilt, the roll
    or the steps of the model's own faults, so each fault's tip ellipse is then carried by the difference between the real top's depth
    at its centre (the planning map; over a cut-out, the nearest cell with rock) and the fold's, which puts it where it was drawn against the reservoir
    (only 66 % of a tilted-block model's faults and 72 % of a rollover's reached the 5 m the density counts at the real top before).
    """
    if trap is None:
        return []
    m, (cx, cy) = trap["mask"], trap["cell"]
    xy = (np.argwhere(m) + 0.5) * np.array([cx, cy])
    centre = xy.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(xy.T)) if len(xy) > 2 else (np.ones(2), np.eye(2))
    aspect = float(np.clip(math.sqrt(max(vals[1], 1e-12) / max(vals[0], 1e-12)), 1.0, 8.0))
    fold = st.closure(area=trap["area"], height=trap["height"], aspect=aspect, azimuth=_azimuth_of(vecs[:, 1]),
                      center=(float(centre[0]), float(centre[1])))
    datum = trap["crest_depth"] + trap["height"]
    faults = fold_faults(style, fold, x_len, y_len, dx, density, datum, thickness, seed, regional=regional, basinward=basinward)
    rock = np.isfinite(trap["depth"])                                      # a centre over a cell with no rock (a cut-out) takes the nearest one's
    top = trap["depth"][tuple(ndimage.distance_transform_edt(~rock, return_distances=False, return_indices=True))]
    for f in faults:
        i, j = (int(np.clip(f.center[k] // c, 0, n - 1)) for k, (c, n) in enumerate(zip((cx, cy), top.shape)))
        f.z_center += float(top[i, j]) - (datum + float(np.ravel(fold(np.array([f.center[0]]), np.array([f.center[1]])))[0]))
    return faults


def _finish(faults, structure, isochore, x_len, y_len, grid, top, labels, structure_trap):
    """Name the faults in order, measure the trap on the finished model and label both traps."""
    for n, f in enumerate(faults):
        f.name = f"F{n + 1:03d}"
    kinds = {}
    for f in faults:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    labels.update(n_faults=len(faults), fault_kinds=kinds, planning_dx_m=float(grid), trap_assumes_sealing_faults=True,
                  structure_trap=_trap_label(structure_trap),
                  trap=_trap_label(_measure(x_len, y_len, grid, top, structure, faults)))
    return BlockModel(structure, faults, isochore, labels)


def tilted_blocks(x_len, y_len, dx, top, thickness, seed, *, fold=None, kind=None, azimuth=None, tilt=None, dip=None,
                  width=None, density=None, scatter=0.1):
    """A tilted-fault-block model (see the module docstring).

    ``x_len``, ``y_len`` the model's size and ``dx`` its cell size (m); ``top`` the reservoir top's datum depth and
    ``thickness`` its thickness (m); ``fold`` an optional extra structure (a :func:`resmill.structure.closure` culmination)
    added to the tilt. Drawn unless pinned: ``kind`` ("domino" 70 %, "horst_graben" 30 % [J]); ``azimuth`` of the beds'
    dip direction (degrees clockwise from +x; any [J]); ``tilt`` (degrees: log-uniform 3-20, 10 % of cases 20-30, T3, T4;
    horst-graben 0.5-5, T11); ``dip`` of the faults (25-35, T5; horst-graben 55-70, T11); ``width`` of a block at the
    horizon (m, log-uniform 1,500-5,000, T1; capped so that the throw stays under 1.5 km); ``density`` of the faults inside
    the trap with a throw of 5 m or more (per km2: log-normal, median 1.7, 1-3, T28). ``scatter``: log10 sd of the faults'
    throws about ``width tan(tilt)`` (0.1 [J]; 0 for exact blocks).
    """
    rng = np.random.default_rng(seed)
    u = dict(kind=rng.uniform(), az=rng.uniform(0.0, 360.0), tail=rng.uniform(), tilt=rng.uniform(), dip=rng.uniform(),
             width=rng.uniform(), density=rng.normal())
    kind = kind or ("domino" if u["kind"] < 0.7 else "horst_graben")
    if kind not in ("domino", "horst_graben"):
        raise ValueError(f"kind must be 'domino' or 'horst_graben', got {kind!r}")
    domino = kind == "domino"
    azimuth = u["az"] if azimuth is None else float(azimuth)
    if tilt is None:
        if not domino:
            tilt = 0.5 + 4.5 * u["tilt"]
        else:
            tilt = 20.0 + 10.0 * u["tilt"] if u["tail"] < 0.1 else 3.0 * (20.0 / 3.0) ** u["tilt"]
    if dip is None:
        dip = (25.0 + 10.0 * u["dip"]) if domino else (55.0 + 15.0 * u["dip"])
        dip = min(dip, 75.0 - tilt)                                       # the beds' initial dip, tilt + dip, stays under 75
    tan_t, tan_d, sin_d = math.tan(math.radians(tilt)), math.tan(math.radians(dip)), math.sin(math.radians(dip))
    if width is None:
        width = min(1500.0 * (5000.0 / 1500.0) ** u["width"], 1500.0 / tan_t if domino else 5000.0)
    density = float(10.0 ** (math.log10(1.7) + 0.3 * u["density"])) if density is None else float(density)
    n, t = _axes(azimuth)
    extent_n, extent_t = abs(x_len * n[0]) + abs(y_len * n[1]), abs(x_len * t[0]) + abs(y_len * t[1])
    diagonal = math.hypot(x_len, y_len)
    beta = 1.0 + tan_t / tan_d if domino else 1.0
    pitch = width * beta                                                  # the fault spacing at the horizon
    count = int(np.clip(round(extent_n / pitch), 1, 6))
    place = (np.arange(count) - 0.5 * (count - 1)) * pitch + rng.uniform(-0.1, 0.1, count) * pitch
    gaps = np.diff(place, prepend=place[0] - pitch)
    wobble = 10.0 ** rng.normal(0.0, scatter, count) if scatter > 0.0 else np.ones(count)
    if domino:
        offset = np.minimum(gaps * tan_t * wobble, 1500.0 * beta)         # each fault's column offset, its spacing x tan(tilt)
        cutoff = offset / beta                                            # cutoff to cutoff along the plane: width tan(tilt)
    else:
        cutoff = offset = np.exp(rng.uniform(math.log(50.0), math.log(500.0), count)) * wobble     # T8: 50-500 m
    length = max(1.2 * diagonal, float(cutoff.max()) / sin_d / D_OVER_L)  # from displacement / length at most 0.1
    lx = 0.5 * length
    zspan = 0.5 * extent_n * tan_t + thickness + float(offset.max())
    aspect = lx * sin_d * 0.03 / zspan          # tall tip ellipse: the throw does not change with the horizon's depth (< 0.3 %)
    mid = np.array([0.5 * x_len, 0.5 * y_len])
    faults, shift = [], 0.0
    for k in range(count):
        hw = -1 if domino or k % 2 == 0 else 1
        faults.append(Fault(
            center=tuple(float(v) for v in mid + place[k] * n), strike=azimuth + float(rng.normal(0.0, 1.0)), length=length,
            throw=float(offset[k]), dip=dip, hanging_wall=hw, hw_share=0.5, drag=(0.0, 0.0), aspect=aspect,
            z_center=top + 0.5 * thickness + tan_t * float(place[k]) + shift, bends=float(rng.uniform(0.004, 0.01)),
            seed=int(rng.integers(2 ** 31)), kind="block"))
        shift += 0.5 * float(offset[k]) * (1.0 if hw == 1 else -1.0)     # the next fault's column sits on this one's footwall
    tilt_field = Structure(lambda x, y: tan_t * ww_profile(np.abs(
        (np.asarray(x, float) - mid[0]) * t[0] + (np.asarray(y, float) - mid[1]) * t[1]) / lx)
        * ((np.asarray(x, float) - mid[0]) * n[0] + (np.asarray(y, float) - mid[1]) * n[1]))
    structure = tilt_field if fold is None else tilt_field + fold
    median = float(np.median(offset))
    for _ in range(int(rng.integers(0, 3))):                              # transfer faults across the blocks (T12)
        across = _log_uniform(rng, 0.4, 1.0) * extent_n
        centre = mid + rng.uniform(-0.3, 0.3) * extent_n * n + rng.uniform(-0.3, 0.3) * extent_t * t
        faults.append(Fault(center=(float(centre[0]), float(centre[1])), strike=azimuth + 90.0 + float(rng.normal(0.0, 8.0)),
                            length=across, throw=median * float(rng.uniform(0.1, 0.3)), dip=float(rng.uniform(80.0, 90.0)),
                            hanging_wall=int(rng.choice((-1, 1))), hw_share=0.5, drag=(0.0, 0.0),
                            z_center=top + 0.5 * thickness, bends=float(rng.uniform(0.004, 0.01)),
                            seed=int(rng.integers(2 ** 31)), kind="transfer"))
    grid = max(dx, max(x_len, y_len) / PLAN_CELLS)
    frame = _measure(x_len, y_len, grid, top, structure, faults)
    hw_dir = (-n if domino else n)                                        # the side the main faults throw down toward
    ratio = float(rng.uniform(1.5, 3.0))                                  # R: 60-75 % of the population strikes with the blocks [J]
    antithetic = float(rng.uniform(0.1, 0.2))                             # T8
    faults += _population("tilted_blocks", frame, x_len, y_len, grid, top, thickness, density, int(rng.integers(2 ** 31)),
                          (ratio, _azimuth_of(hw_dir)), 1.0 - antithetic)
    labels = dict(style="tilted_blocks", kind=kind, seed=int(seed), azimuth_deg=azimuth, tilt_deg=float(tilt), dip_deg=float(dip),
                  width_m=float(width), pitch_m=float(pitch), beta=float(beta), n_main=count, density_per_km2=density,
                  throws_m=[float(c) for c in cutoff], heave_m=[float(c) / tan_d for c in cutoff],
                  length_m=float(length), antithetic_share=antithetic, regional_ratio=ratio)
    return _finish(faults, structure, None, x_len, y_len, grid, top, labels, frame)


def rollover(x_len, y_len, dx, top, thickness, seed, *, fold=None, kind=None, azimuth=None, flatten=None, ramp_base=None,
             dip=None, throw=None, regional_dip=None, expansion=None, density=None, length=None, zones=1):
    """A growth-fault rollover model (see the module docstring).

    ``x_len``, ``y_len``, ``dx``, ``top``, ``thickness`` as for :func:`tilted_blocks`, ``zones`` the number of layers of
    the stack (every zone is syn-kinematic and gets the growth isochore); ``fold`` an optional extra structure (a
    culmination for lateral closure). Drawn unless pinned: ``kind`` ("frio" 60 %, "wilcox" 40 % [J]); ``azimuth`` of the
    basinward direction (any [J]); ``dip`` of the master fault down to its bend (degrees: Frio 50-75, six published faults
    46-77; for the Wilcox the dip of its planar faults, 50-60, T14); ``flatten`` of the Frio master fault, the depth over which
    its tan(dip) falls by 1/e below the bend (m, log-uniform 1,200-5,000: six published faults fit 600-3,100, median 2,450;
    Bruce 1973: 60 to 15 degrees over about 4 km, 2,200); ``ramp_base`` the depth where that ramp ends (m: 0-2,000 above the
    reservoir where the fault is [J]); ``length`` of a master fault (m, log-uniform 3-25 km, T17); ``throw`` of the master
    fault in the reservoir (m). Unpinned, the clastic displacement-length law gives the fault's maximum displacement, at its tip
    ellipse's centre, 0.11 L^0.84 with a log10 sd of 0.27 about it (T10, T19), and the throw there is that times the sine of the
    plane's dip there (:attr:`resmill.faults.Fault.throw`, as ``fold_faults`` draws it); the throw in the reservoir follows from
    the fault, 0.56-0.86 of it for a Wilcox fault (the tip line's profile) and, for a Frio master, from its plane (one heave for
    the whole hanging wall), usually more, as the plane is steeper higher up (``labels["masters"]``). A pinned throw is the one
    in the reservoir and the throw at the centre follows from it. ``regional_dip`` toward the basin (degrees, 0.5-3 [J]); ``expansion`` index of the fault zone (downthrown over upthrown
    thickness, log-uniform 1.1-2.5, 5 % of cases 2.5-5, T18; Wilcox 1.3-2.5), shared among its faults as the count-th root;
    ``density`` of the faults inside the trap with 5 m of throw or more (per km2: log-normal, median 1, 0.5-2, T28).

    The faults inside a trap are drawn on a closure that frames it, and a drag that never turns the regional dip leaves
    the master faults none: drawn values that leave the masters and the regional dip no trap of at least 1 km2 (and nine
    planning cells) are drawn again (``kind`` and ``azimuth`` stay), up to :data:`TRIES` times (``labels["tries"]``).
    Pinned values are never changed.
    """
    rng = np.random.default_rng(seed)
    draw = lambda: dict(kind=rng.uniform(), az=rng.uniform(0.0, 360.0), length=rng.uniform(), dip=rng.uniform(),
                        flat=rng.uniform(), scatter=rng.normal(), alpha=rng.uniform(), tail=rng.uniform(), ei=rng.uniform(),
                        density=rng.normal(), where=rng.uniform(), reach=rng.uniform())
    u = draw()
    kind = kind or ("frio" if u["kind"] < 0.6 else "wilcox")
    if kind not in ("frio", "wilcox"):
        raise ValueError(f"kind must be 'frio' or 'wilcox', got {kind!r}")
    frio = kind == "frio"
    azimuth = u["az"] if azimuth is None else float(azimuth)
    n, _ = _axes(azimuth)
    extent_n = abs(x_len * n[0]) + abs(y_len * n[1])
    mid = np.array([0.5 * x_len, 0.5 * y_len])
    z_res = top + 0.5 * thickness
    grid = max(dx, max(x_len, y_len) / PLAN_CELLS)
    lo = 1.1 if frio else 1.3

    def masters_of(u):
        """The master faults of one set of draws, with the regional dip and the numbers they were built from."""
        alpha = 0.5 + 2.5 * u["alpha"] if regional_dip is None else float(regional_dip)
        tan_a = math.tan(math.radians(alpha))
        long = 3000.0 * (25000.0 / 3000.0) ** u["length"] if length is None else float(length)
        steep = (RAMP_DIP[0] + (RAMP_DIP[1] - RAMP_DIP[0]) * u["dip"] if frio else 50.0 + 10.0 * u["dip"]) if dip is None \
            else float(dip)
        sin_d = math.sin(math.radians(steep))
        bend = (FLATTEN[0] * (FLATTEN[1] / FLATTEN[0]) ** u["flat"] if flatten is None else float(flatten)) if frio else None
        count = 1 if frio else int(rng.integers(2, 4))
        spacing = float(rng.uniform(1500.0, 3000.0))
        first = (0.12 + 0.13 * u["where"] - 0.5) * extent_n                   # landward of the middle: room for the roll
        places = first + np.concatenate([[0.0], np.cumsum(spacing * rng.uniform(0.9, 1.1, count - 1))])
        faults, masters = [], []
        for k in range(count):
            L = long * (1.0 if frio else float(rng.uniform(0.8, 1.2)))
            ly = 0.5 * L / TIP_ASPECT
            z_k = z_res + tan_a * float(places[k])                            # the reservoir where the fault is
            displacement = 0.11 * L ** 0.84 * 10.0 ** (0.27 * (u["scatter"] if k == 0 else rng.normal()))   # T10, T19: Dmax
            reach = 0.25 * (1.0 + u["reach"])                                 # the tip ellipse's centre 0.25-0.5 half-heights
            zc = z_k + reach * ly * sin_d                                     # below the reservoir, as the population's faults
            r = (zc - z_k) / (sin_d * ly)                                     # the reservoir on the tip ellipse, from its centre
            base = (z_k - RAMP_BASE[0] - (RAMP_BASE[1] - RAMP_BASE[0]) * float(rng.uniform()) if ramp_base is None
                    else float(ramp_base)) if frio else None                  # the ramp ends above the reservoir, the bend there
            fault = Fault(
                center=tuple(float(v) for v in mid + places[k] * n), strike=azimuth, length=float(L),
                throw=float(displacement), dip=float(steep), hanging_wall=1, z_center=float(zc), flatten=bend, ramp_base=base,
                radius=float((_log_uniform(rng, 2.0, 10.0) if frio else _log_uniform(rng, 10.0, 30.0)) * L),
                hw_share=1.0 if frio else float(rng.uniform(0.6, 0.9)),
                drag=(0.0, 0.0) if frio else (float(rng.uniform(0.1, 0.3)), float(rng.uniform(0.1, 0.2))),
                bends=float(rng.uniform(0.02, 0.04) if frio else rng.uniform(0.004, 0.012)),
                seed=int(rng.integers(2 ** 31)), kind="master")               # (the throw is set below, once the plane is there)
            plane, trace = _plane(fault, zc)
            c = displacement * math.sin(math.radians(_centre_dip(fault)))     # Dmax is the displacement at the tip ellipse's centre
            if frio:                                  # one heave moves the whole hanging wall (faults.py): that of the throw c at z_center
                if throw is not None:                                         # a pinned throw is the reservoir's: c follows from the plane
                    c = plane(trace(z_k + throw) - trace(z_k)) - zc
                t_res = plane(trace(z_k) + trace(zc + c)) - z_k
            else:
                if throw is not None:
                    c = throw / ww_profile(r)
                t_res = c * ww_profile(r)
            faults.append(replace(fault, throw=float(c)))                     # (checked again)
            t_res = float(t_res if throw is None else throw)
            masters.append(dict(length_m=float(L), displacement_m=float(displacement), throw_m=t_res,
                                centre_throw_m=faults[-1].throw, dip_deg=float(steep), flatten_m=bend, ramp_base_m=base,
                                z_center_m=float(zc)))
        ramp = st.ramp(alpha, azimuth=azimuth, center=(float(mid[0]), float(mid[1])))
        return dict(faults=faults, masters=masters, places=places, tan_a=tan_a, alpha=alpha, steep=steep, bend=bend,
                    structure=ramp if fold is None else ramp + fold)

    for tries in range(1, TRIES + 1):
        built = masters_of(u)
        frame = _measure(x_len, y_len, grid, top, built["structure"], built["faults"])
        if frame is not None and frame["area"] >= max(MIN_AREA, MIN_CELLS * frame["cell"][0] * frame["cell"][1]):
            break
        u = {**draw(), "kind": u["kind"], "az": u["az"]}                  # no trap: the same flavour and direction, drawn again
    faults, structure, count = built["faults"], built["structure"], len(built["faults"])
    ratio = float(rng.uniform(0.5, 2.0))                                  # R of fault_patterns' Gulf anticline
    basinward = float(rng.uniform(0.3, 0.6))                              # 40-70 % antithetic (T23, T24)
    dens = float(10.0 ** (0.3 * u["density"])) if density is None else float(density)
    faults += _population("rollover", frame, x_len, y_len, grid, top, thickness, dens, int(rng.integers(2 ** 31)),
                          (ratio, _azimuth_of(n)), basinward)
    ei = float((2.5 + 2.5 * u["ei"]) if u["tail"] < 0.05 else lo * (2.5 / lo) ** u["ei"]) if expansion is None \
        else float(expansion)
    each = ei ** (1.0 / count)                                            # the zone's index, shared by its faults
    isochore = []
    for z in range(int(zones)):
        depth = top + (z + 0.5) * thickness / zones
        fields = [st.growth(f, each, None, depth + built["tan_a"] * float(built["places"][i]))
                  for i, f in enumerate(faults[:count])]
        isochore.append(Structure(lambda x, y, fs=fields: np.prod([g(x, y) for g in fs], axis=0)))
    labels = dict(style="rollover", kind=kind, seed=int(seed), tries=tries, azimuth_deg=azimuth, dip_deg=built["steep"],
                  flatten_m=built["bend"], ramp_base_m=built["masters"][0]["ramp_base_m"],
                  regional_dip_deg=float(built["alpha"]), expansion=ei, expansion_per_fault=each, zones=int(zones),
                  n_masters=count, density_per_km2=dens, regional_ratio=ratio, basinward_share=basinward, masters=built["masters"])
    return _finish(faults, structure, isochore, x_len, y_len, grid, top, labels, frame)
