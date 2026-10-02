"""Fault seal per cell face, and fault blocks.

A fault seals or leaks by what lies on its plane. Per face of a fault (the faces ``FAULTS`` lists), the shale gouge
ratio SGR (Yielding, Freeman & Needham 1997) is the clay of the beds that slid past the face, over the throw: at a face
at depth z, in the upthrown column the beds from z - throw to z and, a throw lower, in the downthrown column those from
z to z + throw, each column's thickness-weighted clay fraction over the part of its window it holds, the two averaged
(as Lyon et al. 2005 do; Dincau 1998 takes the geometric mean; a tread, a Z face, takes the SGR of the column pair it
counts for). It sets the fault rock's permeability, log10 k_f = -4 SGR - 1/4 log10(D) (1 - SGR)^5 (mD; D the
displacement in m, the throw over sin(dip)), and its thickness t_f = D / 66, the median ratio (Manzocchi, Walsh, Nell &
Yielding 1999). The face's transmissibility multiplier is then T = [1 + t_f (2/k_f - 1/k_i - 1/k_j) / (L_i/k_i +
L_j/k_j)]^-1 between the cells on either side (permeabilities k_i and k_j across the face, L_i and L_j the whole cells'
lengths across it), at most 1, so every face lies somewhere between open and sealed. The throw at a column pair is read
off the grid, as the largest offset of an interface across it, so the throws of several faults add. The physics shapes
each fault but explains little of the spread of real ones (the history-matched multipliers of Norne were uncorrelated
with an earlier version of this prediction), so all faces of a fault also share one log-normal factor 10^(``offset`` +
N(0, ``scatter``)). Those two values are a calibration against history-matched multipliers and hold only for the
formula and the window they were fitted with: fits made with half cell lengths, or a window centred on the face, do not
carry over. A share of faults is left open (``p_open``: every face 1) and a share raises flow (``p_enhance``: every face
one log-uniform value of ``enhance``, above 1), as Norne's history match has (5 of its 36 faults with throw at 1 or
above, up to 3.9, and a fault without throw at 20). A fault draws in this order, so a seed reproduces a tree: one
uniform for its mode when either share is above 0, then one log-uniform value (enhancing) or one normal (seal, with a
``scatter``); with the options off nothing is drawn.

Juxtaposition needs nothing here: the simulator connects only the cells that touch. Over geological time a fault holds
oil back only up to the column its capillary seal supports, however low its multiplier (which sets flow in production),
and :func:`fault_blocks` takes the contacts of the fault blocks from that. A face between two net cells (horizontal
permeability from ``Capillary.net_perm``) holds the pressure :func:`seal_capacity` gives from its SGR and its burial
below the mudline: the upper envelope of Bretan, Yielding & Jones (2003), :func:`bretan_pressure`, 10^(100 SGR/27 - C)
bar with C = 0.5, 0.25 and 0 below 3, at 3-3.5 and above 3.5 km, capped at a plateau (oil: near 3 bar in their data, 6
bar or more in Childs et al. 2009; above 3.5 km oil is uncalibrated, so the plateau holds there), with a floor below an
SGR onset (sand on sand, 0.15-0.25: the juxtaposition leak) or without the membrane (juxtaposition only, as Murray et
al. 2019 back-analyse). A bed against a non-net one seals whatever the SGR. That pressure is an oil column H = 1e5 P /
(g delta_rho) m, so a face at depth z leaks once the contact lies below z + H, and an edge between two map columns below
the lowest such level of its faces (a wall when it has none: a throw beyond the reservoir, or only sand against shale).
A priority flood from the map's edge crosses a faulted edge at the higher of that level and the next column's depth, so
the level it reaches at a block's crest is the block's contact, the shallower of its spill and its leak point (the
weakest window: Bretan et al. 2003, after Gibson 1994, Skerlec 1999 and Childs et al. 2009), and blocks the flood joins
below their common level are one accumulation with one contact. The values of :class:`Capillary` are the research's
central values (SGR onset 0.2, floor 0.5 bar, plateau 4 bar for oil); the dataset draws them per tree.

:func:`fault_blocks` is two steps: :func:`block_inputs` takes what the geometry gives (once per model; ``to_grdecl``
hands it out through ``report=``) and :func:`blocks_at` runs the flood for any ``Capillary``, as a fluid drawn after the
model needs; :func:`block_labels` numbers the blocks, with or without a trap.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .faults import face_records
from .structure import _spill_levels

GRAVITY = 9.81                                  # m/s2
PA_PER_BAR = 1e5
_BURIAL_SHALLOW, _BURIAL_DEEP = 3000.0, 3500.0  # m: the edges of Bretan et al.'s burial classes (C = 0.5, 0.25, 0)
_MIN_DISPLACEMENT = 1e-3                        # m: the least displacement k_f is evaluated at
_MIN_SIN_DIP = 1e-3
_MIN_PERM = 1e-9                                # mD: a cell's permeability in the formulas, at least
_MIN_LENGTH = 1e-6                              # m: a cell's length across a face, at least
_MIN_MULT = 1e-12                               # a sealed face's multiplier (not 0)

FACE_RECORD = [("fault", int), ("axis", int), ("i", int), ("j", int), ("depth", float), ("sgr", float), ("perm", float)]


@dataclass
class Seal:
    """How faults seal (see the module docstring)."""

    vsh: object = None              # clay fraction: per cell (nx, ny, nz, k top-down) or per facies code {code: value}
    dt_ratio: float = 66.0          # displacement over fault-rock thickness (Manzocchi 1999: median 66, harmonic 170)
    scatter: float = 0.0            # per-fault spread (log10) of its face multipliers (calibration)
    seed: int | None = None
    offset: float = 0.0             # per-fault median shift (log10) of its face multipliers (calibration)
    p_open: float = 0.0             # share of faults left open: every face 1
    p_enhance: float = 0.0          # share of faults that raise flow: every face one value of ``enhance``
    enhance: tuple = (1.0, 20.0)    # that multiplier's range, drawn log-uniform

    def __post_init__(self):
        problems = [msg for bad, msg in (
            (not 0.0 <= self.p_open <= 1.0, "p_open must lie in [0, 1]"),
            (not 0.0 <= self.p_enhance <= 1.0, "p_enhance must lie in [0, 1]"),
            (self.p_open + self.p_enhance > 1.0, "p_open + p_enhance must not exceed 1"),
            (not 0.0 < self.enhance[0] <= self.enhance[1], "enhance must be a range from above 0 upwards"),
        ) if bad]
        if problems:
            raise ValueError("Seal: " + "; ".join(problems))


@dataclass
class Capillary:
    """The capillary seal of the fault faces (see the module docstring); defaults are the research's central values."""

    delta_rho: float                # water minus hydrocarbon density at the trap (kg/m3)
    mudline: float = 0.0            # depth of the seabed or ground (m): burial = depth - mudline
    onset: float = 0.2              # [J] SGR below which a face holds only the floor (research J1: core 0.15-0.25)
    floor: float = 0.5              # [J] what a face holds below the onset or without the membrane, bar (J2: 0.3-0.8)
    plateau: float = 4.0            # [J] the most a face holds, bar (J3: core 3-6 for oil; Bretan 3, Childs 6 or more)
    membrane: bool = True           # False: juxtaposition only, every face holds the floor
    net_perm: float = 1.0           # horizontal permeability (geometric mean of PERMX and PERMY) of a net cell (mD)

    def __post_init__(self):
        problems = [msg for bad, msg in (
            (not self.delta_rho > 0.0, "delta_rho must be positive"),
            (not 0.0 <= self.onset <= 1.0, "onset must lie in [0, 1]"),
            (not 0.0 <= self.floor <= self.plateau, "floor and plateau must satisfy 0 <= floor <= plateau"),
        ) if bad]
        if problems:
            raise ValueError("Capillary: " + "; ".join(problems))


def fault_rock_permeability(sgr, displacement):
    """Fault-rock permeability (mD) from the shale gouge ratio and the displacement (m) (Manzocchi et al. 1999)."""
    d = np.maximum(np.asarray(displacement, dtype=float), _MIN_DISPLACEMENT)
    return 10.0 ** (-4.0 * np.asarray(sgr, dtype=float) - 0.25 * np.log10(d) * (1.0 - np.asarray(sgr)) ** 5)


def fault_transmissibility(sgr, displacement, ki, kj, li, lj, dt_ratio=66.0):
    """Transmissibility of a face with the fault rock in it over that without (Manzocchi et al. 1999), from the shale
    gouge ratio and the displacement (m), the permeabilities (mD) and the whole lengths (m) of the cells ``i`` and ``j``
    across it: T = [1 + t_f (2/k_f - 1/k_i - 1/k_j) / (L_i/k_i + L_j/k_j)]^-1, the fault rock of thickness t_f =
    displacement / ``dt_ratio`` and permeability k_f replacing t_f/2 of each cell. Above 1 where the fault rock conducts
    better than the cells; infinite where it would replace more than the cell with a better conductor (the bracket is
    zero or negative): an open face. Arrays or numbers."""
    tf = np.asarray(displacement, dtype=float) / dt_ratio
    bracket = 1.0 + tf * (2.0 / fault_rock_permeability(sgr, displacement) - 1.0 / ki - 1.0 / kj) / (li / ki + lj / kj)
    with np.errstate(divide="ignore"):
        return np.where(bracket > 0.0, 1.0 / bracket, np.inf)


def column_height(pressure, delta_rho):
    """Oil column (m) that a pressure (bar) holds back: H = P / (g delta_rho), delta_rho the water minus hydrocarbon
    density (kg/m3) (Bretan et al. 2003, eq. 2)."""
    return PA_PER_BAR * pressure / (GRAVITY * delta_rho)


def bretan_pressure(sgr, burial):
    """Pressure (bar) a fault face supports by capillary seal: the upper envelope of Bretan, Yielding & Jones (2003,
    eq. 1), 10^(100 SGR/27 - C) with SGR a fraction and burial in m, C = 0.5 below 3,000 m, 0.25 to 3,500 m and 0 deeper
    (cementation above about 90 C, 3 km, stiffens the fault rock)."""
    burial = np.asarray(burial, dtype=float)
    c = np.where(burial < _BURIAL_SHALLOW, 0.5, np.where(burial <= _BURIAL_DEEP, 0.25, 0.0))
    return 10.0 ** (100.0 * np.asarray(sgr, dtype=float) / 27.0 - c)


def seal_capacity(sgr, burial, cap):
    """Pressure (bar) a fault face between two net cells holds, from its SGR (a fraction) and burial (m): the floor of
    ``cap`` below its onset or without the membrane, else :func:`bretan_pressure` up to the plateau, which is what a
    face holds above 3,500 m burial, where oil is uncalibrated (the envelope's C = 0 class is gas'). Never below the
    floor: eq. 1 at SGR 0 is 0.32 bar below 3 km, under a floor of 0.5 bar."""
    sgr, burial = np.asarray(sgr, dtype=float), np.asarray(burial, dtype=float)
    held = np.where(burial > _BURIAL_DEEP, cap.plateau, np.minimum(bretan_pressure(sgr, burial), cap.plateau))
    return np.where((sgr < cap.onset) | (not cap.membrane), cap.floor, np.maximum(held, cap.floor))


def _cell_depths(zc):
    """The interface depths of each cell, the mean of its four corners: (nx, ny, nz + 1)."""
    return 0.25 * (zc[0::2, 0::2] + zc[1::2, 0::2] + zc[0::2, 1::2] + zc[1::2, 1::2])


def _column_pair(zc, i, j, face):
    """The two columns' depths on a shared face: (2 corners, nk) for the lower-index and the upper-index column."""
    if face == "X":
        return zc[2 * i + 1, 2 * j:2 * j + 2], zc[2 * i + 2, 2 * j:2 * j + 2]
    return zc[2 * i:2 * i + 2, 2 * j + 1], zc[2 * i:2 * i + 2, 2 * j + 2]


def _window_clay(column, lo, hi):
    """Mean clay fraction of a column over the depth window [lo, hi] (m; arrays or numbers), and the thickness of the
    window the column holds. ``column`` is (interface depths, clay thickness above each interface)."""
    depth, clay = column
    lo, hi = np.clip(lo, depth[0], depth[-1]), np.clip(hi, depth[0], depth[-1])
    thick = hi - lo
    held = np.interp(hi, depth, clay) - np.interp(lo, depth, clay)
    return np.divide(held, thick, out=np.zeros_like(thick), where=thick > 0.0), thick


class _Slip:
    """What slid past the faces of two neighbouring columns: their throw (m) and the clay of the beds either side."""

    def __init__(self, a, b, vsh_a, vsh_b):
        """``a``, ``b``: the interface depths (2 pillars, nk + 1) of the lower- and the upper-index column on their
        shared face (:func:`_column_pair`); ``vsh_a``, ``vsh_b``: their clay fraction per cell."""
        gap = np.abs(a - b).mean(axis=0)                   # a dipping plane offsets only some layers
        self.throw = float(gap.max())
        depths = (a.mean(axis=0), b.mean(axis=0))
        columns = [(d, np.concatenate(([0.0], np.cumsum(v * np.diff(d))))) for d, v in zip(depths, (vsh_a, vsh_b))]
        a_up = depths[0][gap.argmax()] < depths[1][gap.argmax()]
        self.up, self.down = columns if a_up else columns[::-1]

    def sgr(self, z):
        """The shale gouge ratio at depth z (m) of a face of this pair: the clay of the beds that slid past that point
        (Yielding, Freeman & Needham 1997). In the upthrown column they lie between z - throw and z, and, a throw lower,
        in the downthrown column between z and z + throw; each column's thickness-weighted clay fraction over the part
        of its window it holds, the two averaged (the practice of Lyon et al. 2005 and Dincau 1998 for lateral
        changes)."""
        up, held_up = _window_clay(self.up, z - self.throw, z)
        down, held_down = _window_clay(self.down, z, z + self.throw)
        return np.where((held_up > 0.0) & (held_down > 0.0), 0.5 * (up + down), np.where(held_up > 0.0, up, down))


def _slips(records, zc, vsh):
    """The :class:`_Slip` of every pair of columns a fault's lateral face ``records`` lie on, keyed (face, I, J) as in
    the records (1-based; the pair is the column I, J and its next neighbour in x or y)."""
    slips = {}
    for _, i1, _, j1, _, _, _, face in records:
        if face in "XY" and (face, i1, j1) not in slips:
            i, j = i1 - 1, j1 - 1
            slips[(face, i1, j1)] = _Slip(*_column_pair(zc, i, j, face), vsh[i, j],
                                          vsh[i + (face == "X"), j + (face == "Y")])
    return slips


def _near_edge(slips, face, i1, j1):
    """The pair of columns a face counts for: its own, or for a tread (a Z face, inside one column) the nearest lateral
    pair of its fault; None when the fault has no lateral face."""
    if face != "Z":
        return (face, i1, j1)
    return min(slips, key=lambda edge: abs(edge[1] - i1) + abs(edge[2] - j1)) if slips else None


class _Grid:
    """The grid the faults cut, as their faces need it: depths, active cells, permeabilities, cell sizes."""

    def __init__(self, zc, act, perms, dx, dy):
        self.zc, self.act, self.perms, self.dx, self.dy = zc, act, perms, dx, dy
        self.depth = _cell_depths(zc)

    def rest_against(self, face, i, j, ks):
        """What the cells ``ks`` of column (i, j) meet across a face of kind ``face`` ("X", "Y" or "Z"), as ``(i2, j2,
        k2, z, area, li, lj)``: the indices of the cells they rest against, the depth of each face (m: the cell's middle
        at a lateral face, its base at a tread), its area (m2) and the whole lengths (m) of the two cells across it. A
        lateral face is against the cell of the next column that the cell overlaps most (its own layer if none); a
        tread, one face, against the next active cell below it (None if there is none)."""
        d = self.depth
        if face == "Z":
            k = int(ks[0])
            if not self.act[i, j, k + 1:].any():
                return None
            k2 = k + 1 + int(np.argmax(self.act[i, j, k + 1:]))
            return (i, j, np.array([k2]), d[i, j, [k + 1]], self.dx * self.dy,
                    d[i, j, k + 1] - d[i, j, k], d[i, j, k2 + 1] - d[i, j, k2])
        a, b = _column_pair(self.zc, i, j, face)
        ta, ba, tb, bb = a[:, :-1].mean(axis=0), a[:, 1:].mean(axis=0), b[:, :-1].mean(axis=0), b[:, 1:].mean(axis=0)
        overlap = np.clip(np.minimum(bb, ba[ks, None]) - np.maximum(tb, ta[ks, None]), 0.0, None)   # (cells, layers)
        k2 = np.where(overlap.max(axis=1) > 0.0, overlap.argmax(axis=1), ks)
        width, length = (self.dy, self.dx) if face == "X" else (self.dx, self.dy)
        return ((i + 1, j) if face == "X" else (i, j + 1)) + (k2, 0.5 * (ta + ba)[ks], overlap.sum(axis=1) * width,
                                                              length, length)

    def permeability(self, face, cells):
        """The permeability (mD) of ``cells`` = (i, j, ks) across a face of kind ``face``, at least ``_MIN_PERM``."""
        return np.maximum(np.asarray(self.perms["XYZ".index(face)][cells], dtype=float), _MIN_PERM)

    def net_permeability(self, cells, others):
        """The lower of the horizontal permeabilities (the geometric mean of PERMX and PERMY, mD) of two cell sets."""
        horizontal = [np.sqrt(np.asarray(self.perms[0][c], dtype=float) * np.asarray(self.perms[1][c], dtype=float))
                      for c in (cells, others)]
        return np.maximum(np.minimum(*horizontal), _MIN_PERM)


def _draw_mode(seal, rng):
    """A fault's ``(mode, fixed, factor)``: "seal" (``fixed`` None, the physics times ``factor``), "open" (every face 1)
    or "enhancing" (every face ``fixed``), by one uniform when either share is above 0, then one normal (seal, with a
    ``scatter``) or one log-uniform value (enhancing); nothing is drawn with the options off."""
    mode, fixed, shift = "seal", None, seal.offset
    if seal.p_open + seal.p_enhance > 0.0:
        u = rng.random()
        if u < seal.p_open:
            mode, fixed = "open", 1.0
        elif u < seal.p_open + seal.p_enhance:
            mode, fixed = "enhancing", 10.0 ** rng.uniform(*np.log10(seal.enhance))
    if mode == "seal" and seal.scatter > 0.0:
        shift += rng.normal(0.0, seal.scatter)
    return mode, fixed, 10.0 ** shift


def _capped(transmissibility, factor):
    """The physics multiplier scaled by a fault's factor and held between ``_MIN_MULT`` and 1; an open face (infinite)
    is 1 whatever the factor."""
    with np.errstate(invalid="ignore"):
        return np.where(np.isinf(transmissibility), 1.0, np.clip(transmissibility * factor, _MIN_MULT, 1.0))


def _record_array(rows):
    """The ``face_records`` structured array from per-face-batch tuples of columns."""
    out = np.empty(sum(len(r[0]) for r in rows), dtype=FACE_RECORD)
    for name, column in zip(out.dtype.names, zip(*rows)):
        out[name] = np.concatenate(column)
    return out


def face_multipliers(faces, zc, act, vsh, perms, dx, dy, seal):
    """Transmissibility multipliers per cell face from the faults' shale gouge ratio.

    ``faces`` holds ``(fault, side)`` pairs (from the geometry builder), ``zc`` the final interface stack, ``act`` the
    active cells, ``vsh`` the clay fraction per cell and ``perms`` (PERMX, PERMY, PERMZ), all (nx, ny, nz) in K-down
    order. Returns ``MULTX``, ``MULTY``, ``MULTZ`` (nx, ny, nz; a cell's + face, as the GRDECL keywords mean), and per
    fault its ``mode`` ("seal", "open" or "enhancing"), ``sgr`` and ``mult`` per face and ``effective``, the faces' mean
    weighted by the transmissibility each would have without the fault: the one multiplier for the whole fault (a
    MULTFLT) that passes the same flow. ``face_records`` has one row per face with a multiplier, for
    :func:`fault_blocks`: its ``fault`` (index in ``faces``), the map edge it counts for (``axis`` 0 for X, 1 for Y, and
    the lower-index column ``i``, ``j``; a Z face, a tread inside one column, counts for the nearest lateral edge of its
    fault, as for its throw), its ``depth`` (m, mid), its ``sgr`` and ``perm``, the lower of the two cells' horizontal
    permeabilities (mD; the geometric mean of PERMX and PERMY, the same on a Z face as on a lateral one: a cell is net
    reservoir or not for all its faces).
    """
    grid, rng = _Grid(zc, act, perms, dx, dy), np.random.default_rng(seal.seed)
    out = {key: np.ones(act.shape) for key in ("MULTX", "MULTY", "MULTZ")}
    info, rows = [], []
    for n, (fault, side) in enumerate(faces):
        records = face_records("F", side, zc, act)
        slips = _slips(records, zc, vsh)
        mode, fixed, factor = _draw_mode(seal, rng)
        sin_dip = max(math.sin(math.radians(fault.dip)), _MIN_SIN_DIP)
        sgrs, mults, weights = [], [], []
        for _, i1, _, j1, _, k1, k2, face in records:
            near = _near_edge(slips, face, i1, j1)
            if near is None or slips[near].throw <= 0.0:
                continue
            cells = (i1 - 1, j1 - 1, np.arange(k1 - 1, k2))
            rest = grid.rest_against(face, *cells)
            if rest is None:
                continue
            i2, j2, k_other, z, area, li, lj = rest
            slip = slips[near]
            ki, kj = grid.permeability(face, cells), grid.permeability(face, (i2, j2, k_other))
            li, lj = np.maximum(li, _MIN_LENGTH), np.maximum(lj, _MIN_LENGTH)
            sgr = slip.sgr(z)
            physics = fault_transmissibility(sgr, slip.throw / sin_dip, ki, kj, li, lj, seal.dt_ratio)
            mult = _capped(physics, factor) if fixed is None else np.full(sgr.shape, fixed)
            out["MULT" + face][cells] *= mult
            sgrs.append(sgr)
            mults.append(mult)
            weights.append(area / (0.5 * li / ki + 0.5 * lj / kj))        # the flow without the fault, centre to face
            m = len(sgr)
            rows.append((np.full(m, n), np.full(m, "XY".index(near[0])), np.full(m, near[1] - 1),
                         np.full(m, near[2] - 1), z, sgr, grid.net_permeability(cells, (i2, j2, k_other))))
        sgr, mult, weight = (np.concatenate(p) if p else np.zeros(0) for p in (sgrs, mults, weights))
        info.append(dict(name=fault.name, mode=mode, sgr=sgr.tolist(), mult=mult.tolist(),
                         effective=float(weight @ mult / weight.sum()) if weight.sum() > 0.0 else 1.0))
    out["faults"], out["face_records"] = info, _record_array(rows)
    return out


_EDGE_SLICES = ((np.s_[:-1], np.s_[1:]), (np.s_[:, :-1], np.s_[:, 1:]))   # the cells either side of an x edge, a y edge


def block_inputs(zc, act, faces, mults):
    """What :func:`blocks_at` needs of the geometry, none of it dependent on the fluid or the seal's capillary values: the
    map of column tops (``depth``, NaN where a column holds no rock; ``alive`` says which do), the cell edges a fault
    splits (``split``: ``[x edges (nx-1, ny), y edges (nx, ny-1)]``, True where a fault puts the tops of the two columns
    on different sides) and ``mults["face_records"]`` (:func:`face_multipliers`). Small arrays of map size, so a model can
    keep them and draw its blocks again for another fluid."""
    nx, ny = act.shape[:2]
    k_top = np.argmax(act, axis=2)
    alive = act.any(axis=2)
    depth = np.where(alive, np.take_along_axis(_cell_depths(zc), k_top[..., None], axis=2)[..., 0], np.nan)
    split = [np.zeros((nx - 1, ny), dtype=bool), np.zeros((nx, ny - 1), dtype=bool)]
    for _, side in faces:
        s0 = np.take_along_axis(side, k_top[..., None], axis=2)[..., 0]
        for walls, (lo, hi) in zip(split, _EDGE_SLICES):
            walls |= s0[lo] * s0[hi] == -1
    return dict(depth=depth, alive=alive, split=split, face_records=mults["face_records"])


def _pass_levels(inputs, capillary):
    """The depth below which oil crosses each edge between neighbouring map columns, as arrays for the x edges (nx-1,
    ny) and the y edges (nx, ny-1). An edge nothing splits (the tops of its two columns lie on one side of every fault)
    is free, -inf. Where a fault splits the tops it is a wall, +inf, until a face between two net cells leaks: the
    lowest of their levels, depth + H, with H the oil column ``capillary`` says the face holds."""
    gates = [np.where(walls, np.inf, -np.inf) for walls in inputs["split"]]
    rec = inputs["face_records"]
    rec = rec[rec["perm"] >= capillary.net_perm]
    leak = rec["depth"] + column_height(seal_capacity(rec["sgr"], rec["depth"] - capillary.mudline, capillary),
                                        capillary.delta_rho)
    for axis, gate in enumerate(gates):
        here = rec["axis"] == axis
        np.minimum.at(gate, (rec["i"][here], rec["j"][here]), leak[here])
    return gates


def _free_graph(inputs):
    """The graph of the map's columns joined across the edges no fault splits, and each column's component (a column
    without rock is a component of its own)."""
    alive = inputs["alive"]
    nx, ny = alive.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    free = [~walls & alive[lo] & alive[hi] for walls, (lo, hi) in zip(inputs["split"], _EDGE_SLICES)]
    rows = np.concatenate([idx[lo][edge] for edge, (lo, hi) in zip(free, _EDGE_SLICES)])
    cols = np.concatenate([idx[hi][edge] for edge, (lo, hi) in zip(free, _EDGE_SLICES)])
    graph = sparse.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()
    return graph, csgraph.connected_components(graph, directed=False)[1].reshape(nx, ny)


def block_labels(inputs):
    """The fault block of every map column, ``(nx, ny)`` int from 0: the columns joined through edges no fault splits
    (the blocks of :func:`blocks_at`, whether or not they hold a trap). -1 where a column holds no rock."""
    alive = inputs["alive"]
    out = np.full(alive.shape, -1)
    out[alive] = np.unique(_free_graph(inputs)[1][alive], return_inverse=True)[1]
    return out


def _contact_exit(trap, level, spill, depth, gates):
    """How the contact ``level`` of ``trap`` is set: ``(spills, point)``. A flood enters the trap over a cell edge at
    the higher of the edge's pass level and the level of the cell outside; the edges where that is ``level`` set the
    contact. If one of them is unfaulted the block spills over its own rim (``spills``), otherwise the oil leaves across
    a fault. ``point`` is the shallowest cell outside the trap over those edges (the unfaulted ones when there are any),
    None when there are none: nothing reaches the trap."""
    hits = []
    for gate, (lo, hi) in zip(gates, _EDGE_SLICES):
        outside = np.where(trap[lo] & ~trap[hi], spill[hi], np.where(trap[hi] & ~trap[lo], spill[lo], np.inf))
        hits.append(np.maximum(outside, gate) == level)
    free = [hit & (gate == -np.inf) for hit, gate in zip(hits, gates)]
    spills = free[0].any() or free[1].any()
    ring = np.zeros_like(trap)
    for edges, (lo, hi) in zip(free if spills else hits, _EDGE_SLICES):
        ring[lo] |= trap[hi] & edges
        ring[hi] |= trap[lo] & edges
    ring &= ~trap
    point = np.unravel_index(int(np.argmin(np.where(ring, depth, np.inf))), depth.shape) if ring.any() else None
    return bool(spills), point


def _share_contacts(blocks, gates):
    """Number the accumulations in place (``group``): blocks whose traps meet across an edge that leaks at or above both
    their contacts are joined by the flood, so they share one contact and one number (counted from the shallowest crest;
    ``blocks`` is sorted by it)."""
    where = np.full(blocks[0]["mask"].shape, -1)
    for b, blk in enumerate(blocks):
        where[blk["mask"]] = b
    contact = np.array([blk["contact_depth"] for blk in blocks])
    pairs = []
    for gate, (lo, hi) in zip(gates, _EDGE_SLICES):
        a, b = where[lo], where[hi]
        join = (a >= 0) & (b >= 0) & (a != b) & (gate <= np.minimum(contact[a], contact[b]))
        pairs.append(np.stack([a[join], b[join]]))
    a, b = np.concatenate(pairs, axis=1)
    _, comp = csgraph.connected_components(sparse.coo_matrix((np.ones(a.size), (a, b)), shape=(len(blocks),) * 2),
                                           directed=False)
    first = {}
    for blk, c in zip(blocks, comp):
        blk["group"] = first.setdefault(c, len(first))


def blocks_at(inputs, dx, dy, capillary):
    """The top surface's fault blocks and each block's contact for ``capillary``, from :func:`block_inputs` (see
    :func:`fault_blocks`, which is this on the inputs of a model's own geometry): one fluid's blocks, then another's,
    from the same inputs."""
    depth, alive = inputs["depth"], inputs["alive"]
    nx, ny = alive.shape
    gates = _pass_levels(inputs, capillary)
    spill = _spill_levels(np.where(alive, depth, np.inf), *gates)
    graph, labels = _free_graph(inputs)
    blocks = []
    for lab in np.unique(labels[alive]):
        cells = (labels == lab) & alive
        crest = np.unravel_index(int(np.argmin(np.where(cells, depth, np.inf))), depth.shape)
        level = spill[crest] if np.isfinite(spill[crest]) else float(depth[cells].max())
        below = cells & (depth < level)
        keep = below.ravel()
        _, sublab = csgraph.connected_components(graph[keep][:, keep], directed=False)
        full = np.full(nx * ny, -1)
        full[keep] = sublab
        trap = full.reshape(nx, ny) == full.reshape(nx, ny)[crest] if below[crest] else np.zeros_like(cells)
        if not trap.any():
            continue
        spills, point = _contact_exit(trap, level, spill, depth, gates)
        blocks.append(dict(crest=(int(crest[0]), int(crest[1])), crest_depth=float(depth[crest]),
                           contact_depth=float(level),
                           limited_by="sealed" if not np.isfinite(spill[crest]) else "spill" if spills else "leak",
                           point=None if point is None else (int(point[0]), int(point[1])),
                           area=float(trap.sum()) * dx * dy, height=float(level - depth[crest]), mask=trap))
    blocks.sort(key=lambda blk: blk["crest_depth"])
    if blocks:
        _share_contacts(blocks, gates)
    return blocks


def fault_blocks(zc, act, faces, mults, dx, dy, capillary):
    """The top surface's fault blocks, each block's contact, and the blocks that share one (see the module docstring).

    Two neighbouring columns lie in different blocks where a fault separates their tops. Each such edge leaks below the
    lowest of its net-on-net faces' levels, depth + H (``capillary``, and ``mults["face_records"]`` from
    :func:`face_multipliers`), a wall where it has none. The spill flood from the map's edge crosses it at the higher of
    that level and the next column's depth; the level it reaches at a block's crest is the block's contact, and the
    cells shallower than it that join the crest are its trap (a block nothing reaches fills to its deepest point).
    Returns one dict per block holding a trap, shallowest crest first: ``crest`` (i, j), ``crest_depth``,
    ``contact_depth``, ``limited_by`` ("spill": the block's own rim holds the contact; "leak": the oil leaves across a
    fault, into another block or past its weakest window; "sealed": nothing reaches the block), ``point`` (i, j: the
    column beside the trap the oil leaves into, None when sealed), ``area`` (m2), ``height`` (m), ``mask`` and
    ``group``: blocks whose traps meet across a fault that leaks at or above their contact are one accumulation, with
    one contact and one group number. :func:`block_inputs` and :func:`blocks_at` are its two halves.
    """
    return blocks_at(block_inputs(zc, act, faces, mults), dx, dy, capillary)
