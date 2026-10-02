"""Tilted fault blocks (structure step 4).

A trap style returning a :class:`BlockModel` (the structure, the faults, the growth isochores and every drawn
value) that ``to_grdecl`` takes as it is. Every argument left ``None`` is drawn from its range with
``numpy.random.default_rng(seed)``; a value pins it (explicit mode), and pinning one argument leaves the others as drawn.
Ranges are those of ``design_notes/structure_research/step4_blocks_rollovers.md`` (rows T1-T28); values tagged [J] are
judgement values, labelled so. A trap is measured, never fitted: its area and relief are read off the faulted top, every
fault taken as sealing (``labels["trap"]``; ``labels["structure_trap"]`` is the one before the faults inside it).

:func:`tilted_blocks`: a domino system, N parallel planar faults dipping 25-35 degrees whose hanging walls lie on the up-dip
side of beds tilted 3-20 degrees (10 % of cases up to 30). Each block is a rigid plate: the structure is a tilt, and each fault
displaces the whole of one side (``hw_share`` 0.5, no drag) by the column offset ``tan(tilt) x`` (fault spacing), so that the
horizon is a sawtooth with no net dip: blocks ``width`` wide at the horizon, a throw (cutoff to cutoff along the plane) of
``width tan(tilt)``, a heave of throw / tan(dip) and an extension beta = 1 + tan(tilt) / tan(dip) of the blocks' horizontal
widths, sin(alpha + theta) / sin(theta) of their lengths along the beds (Fossen & Hesthammer 1998; T6, T9). Tilt and throw taper
along the strike like the faults' tip ellipse, so the blocks plunge toward the fault tips and each footwall crest closes
laterally. ``kind="horst_graben"``: alternating hanging walls, dips 55-70 degrees, a tilt under 5 degrees (T11). Transfer
faults (0-2) cross the blocks; the faults inside the trap are drawn by :func:`resmill.fault_patterns.fold_faults`.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from . import structure as st
from .export import _build_geometry
from .fault_patterns import fold_faults
from .faults import Fault, ww_profile
from .layers.base import Layer
from .structure import Structure, _spill_levels

PLAN_CELLS = 300                    # the planning grid (a trap is measured and framed on it) has at most this many cells a side
PLAN_THICKNESS = 4.0                # m: the layer a trap is measured on
D_OVER_L = 0.1                      # upper bound of displacement / length (Lathrop et al. 2022)


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
    ``crest_depth``, ``spill_depth``, ``mask`` and the cell size ``cell``.
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
                spill_depth=float(spill[crest]))


def _trap_label(trap):
    """A measured trap as plain numbers for the record (None when there is none)."""
    return None if trap is None else dict(
        area_km2=trap["area"] / 1e6, height_m=trap["height"], spill_depth_m=trap["spill_depth"],
        crest_depth_m=trap["crest_depth"], crest_xy_m=[float(v) for v in trap["crest_xy"]])


def _population(style, trap, x_len, y_len, dx, top, thickness, density, seed, regional, basinward):
    """The faults inside a block model's trap, from ``fold_faults`` on a smooth fold of that trap (none without a trap).

    The fold has the measured trap's area, relief, aspect and axis and its crest at the trap's real depth, so that the faults
    are drawn on a closure that frames it.
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
    return fold_faults(style, fold, x_len, y_len, dx, density, trap["crest_depth"] + trap["height"], thickness, seed,
                       regional=regional, basinward=basinward)


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
