"""Fault seal per cell face, and fault blocks.

A fault seals or leaks by what lies on its plane. Per face of a fault (the faces ``FAULTS`` lists), the shale gouge
ratio SGR (Yielding, Freeman & Needham 1997) is the clay of the beds that slid past the face, over the throw: at a face
at depth z, in the upthrown column the beds from z - throw to z and, a throw lower, in the downthrown column those from
z to z + throw, each column's thickness-weighted clay fraction over the part of its window it holds, the two averaged
(Lyon et al. 2005 and Dincau 1998 do so for lateral changes; a tread, a Z face, takes the SGR of the column pair it
counts for). It sets the fault rock's permeability, log10 k_f = -4 SGR - 1/4 log10(D) (1 - SGR)^5 (mD; D the
displacement in m, the throw over sin(dip)), and its thickness t_f = D / 66, the median ratio (Manzocchi, Walsh, Nell &
Yielding 1999). The face's transmissibility multiplier is then T = [1 + t_f (2/k_f - 1/k_i - 1/k_j) / (L_i/k_i +
L_j/k_j)]^-1 between the cells on either side (permeabilities k_i and k_j across the face, L_i and L_j the whole cells'
lengths across it), at most 1, so every face lies somewhere between open and sealed. The throw at a column pair is read
off the grid, as the largest offset of an interface across it, so the throws of several faults add. The physics shapes
each fault but explains none of the spread of real ones (on Norne the predicted and the history-matched multipliers are
uncorrelated), so all faces of a fault also share one log-normal factor 10^(``offset`` + N(0, ``scatter``)), calibrated
on Norne's history match (``scatter`` 0.9 and ``offset`` -0.6 on its own grid, the mean of three fits to its 36 faults
with throw). A share of faults is left open (``p_open``: every face 1) and a share raises flow (``p_enhance``: every
face one log-uniform value of ``enhance``, above 1), as Norne's history match has (5 of its 36 faults with throw at 1 or
above, up to 3.9, and a fault without throw at 20). The share of faults that seal by Dn = throw / gross reservoir
thickness in Knott's (1993) North Sea counts (:func:`knott_seal_probability`) is a check on the calibration. A fault
draws in this order, so a seed reproduces a tree: one uniform for its mode when either share is above 0, then one
log-uniform value (enhancing) or one normal (seal, with a ``scatter``); with the options off nothing is drawn.

Juxtaposition needs nothing here: the simulator connects only the cells that touch. Over geological time a fault holds
oil back only up to the column its capillary seal supports, however low its multiplier (which sets flow in production),
and :func:`fault_blocks` takes the contacts of the fault blocks from that. A face between two net cells (permeability
from ``Capillary.net_perm``) holds the pressure :func:`seal_capacity` gives from its SGR and its burial below the
mudline: the upper envelope of Bretan, Yielding & Jones (2003), :func:`bretan_pressure`, 10^(100 SGR/27 - C) bar with C
= 0.5, 0.25 and 0 below 3, at 3-3.5 and above 3.5 km, capped at a plateau (oil: near 3 bar in their data, 6 bar or more
in Childs et al. 2009; above 3.5 km oil is uncalibrated, so the plateau holds there), with a floor below an SGR onset
(sand on sand, 0.15-0.25: the juxtaposition leak) or without the membrane (juxtaposition only, as Murray et al. 2019
back-analyse). A bed against a non-net one seals whatever the SGR. That pressure is an oil column H = 1e5 P / (g
delta_rho) m, so a face at depth z leaks once the contact lies below z + H, and an edge between two map columns below
the lowest such level of its faces (a wall when it has none: a throw beyond the reservoir, or only sand against shale).
A priority flood from the map's edge crosses a faulted edge at the higher of that level and the next column's depth, so
the level it reaches at a block's crest is the block's contact, the shallower of its spill and its leak point (the
weakest window: Bretan et al. 2003, after Gibson 1994, Skerlec 1999 and Childs et al. 2009), and blocks the flood joins
below their common level are one accumulation with one contact. The values of :class:`Capillary` are the research's
central values (SGR onset 0.2, floor 0.5 bar, plateau 4 bar for oil); the dataset draws them per tree.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .faults import face_records
from .structure import _spill_levels

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
    onset: float = 0.2              # SGR below which a face holds only the floor
    floor: float = 0.5              # what a face holds below the onset or without the membrane (bar)
    plateau: float = 4.0            # the most a face holds (bar)
    membrane: bool = True           # False: juxtaposition only, every face holds the floor
    net_perm: float = 1.0           # permeability from which a cell is net reservoir (mD)

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
    d = np.maximum(np.asarray(displacement, dtype=float), 1e-3)
    return 10.0 ** (-4.0 * np.asarray(sgr, dtype=float) - 0.25 * np.log10(d) * (1.0 - np.asarray(sgr)) ** 5)


def knott_seal_probability(dn):
    """Share of faults that seal, by throw over gross reservoir thickness: Knott (1993), the northern and southern North
    Sea counts averaged per class (over 0.9 and all for Dn >= 1; 1 in 2 and 82 % for 0.5-1; 1 in 3 to 1 in 2 and 66 %
    for 0.25-0.5; 1 in 3 and 14 % below 0.25)."""
    return 0.92 if dn >= 1.0 else 0.66 if dn >= 0.5 else 0.5 if dn >= 0.25 else 0.24                  # [J] (averages)


def bretan_pressure(sgr, burial):
    """Pressure (bar) a fault face supports by capillary seal: the upper envelope of Bretan, Yielding & Jones (2003,
    eq. 1), 10^(100 SGR/27 - C) with SGR a fraction and burial in m, C = 0.5 below 3,000 m, 0.25 to 3,500 m and 0 deeper
    (cementation above about 90 C, 3 km, stiffens the fault rock)."""
    burial = np.asarray(burial, dtype=float)
    c = np.where(burial < 3000.0, 0.5, np.where(burial <= 3500.0, 0.25, 0.0))
    return 10.0 ** (100.0 * np.asarray(sgr, dtype=float) / 27.0 - c)


def seal_capacity(sgr, burial, cap):
    """Pressure (bar) a fault face between two net cells holds, from its SGR (a fraction) and burial (m): the floor of
    ``cap`` below its onset or without the membrane, else :func:`bretan_pressure` up to the plateau, which is what a
    face holds above 3,500 m burial, where oil is uncalibrated (the envelope's C = 0 class is gas')."""
    sgr, burial = np.asarray(sgr, dtype=float), np.asarray(burial, dtype=float)
    held = np.where(burial > 3500.0, cap.plateau, np.minimum(bretan_pressure(sgr, burial), cap.plateau))
    return np.where((sgr < cap.onset) | (not cap.membrane), cap.floor, held)


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
        """``a``, ``b``: the interface depths (2 pillars, nk + 1) of the lower- and the upper-index column on their shared
        face (:func:`_column_pair`); ``vsh_a``, ``vsh_b``: their clay fraction per cell."""
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
        of its window it holds, the two averaged (the practice of Lyon et al. 2005 and Dincau 1998 for lateral changes)."""
        up, held_up = _window_clay(self.up, z - self.throw, z)
        down, held_down = _window_clay(self.down, z, z + self.throw)
        return np.where((held_up > 0.0) & (held_down > 0.0), 0.5 * (up + down), np.where(held_up > 0.0, up, down))


def face_multipliers(faces, zc, act, vsh, perms, dx, dy, seal, thickness=None):
    """Transmissibility multipliers per cell face from the faults' shale gouge ratio.

    ``faces`` holds ``(fault, side)`` pairs (from the geometry builder), ``zc`` the final interface stack, ``act`` the
    active cells, ``vsh`` the clay fraction per cell and ``perms`` (PERMX, PERMY, PERMZ), all (nx, ny, nz) in K-down
    order. Returns ``MULTX``, ``MULTY``, ``MULTZ`` (nx, ny, nz; a cell's + face, as the GRDECL keywords mean), the
    faces each fault listed (``listed``), and per fault its ``mode`` ("seal", "open" or "enhancing"), ``dn``, ``sgr``
    and ``mult`` per face and ``effective``, the faces' mean weighted by the transmissibility each would have without
    the fault: the one multiplier for the whole fault (a MULTFLT) that passes the same flow. ``face_records`` has one
    row per face with a multiplier, for :func:`fault_blocks`: its ``fault`` (index in ``faces``), the map edge it counts
    for (``axis`` 0 for X, 1 for Y, and the lower-index column ``i``, ``j``; a Z face, a tread inside one column, counts
    for the nearest lateral edge of its fault, as for its throw), its ``depth`` (m, mid), its ``sgr`` and ``perm``, the
    lower of the two cells' permeabilities across the face's direction (mD).
    """
    nx, ny, nz = act.shape
    rng = np.random.default_rng(seal.seed)
    out = {key: np.ones((nx, ny, nz)) for key in ("MULTX", "MULTY", "MULTZ")}
    listed = {"MULTX": np.zeros((nx, ny, nz), dtype=bool), "MULTY": np.zeros((nx, ny, nz), dtype=bool)}
    cell = 0.25 * (zc[0::2, 0::2] + zc[1::2, 0::2] + zc[0::2, 1::2] + zc[1::2, 1::2])
    tops, bots = cell[..., :-1], cell[..., 1:]
    gross = thickness if thickness is not None else float(np.median((zc[:, :, -1] - zc[:, :, 0])))
    info, rows = [], []
    for n, (fault, side) in enumerate(faces):
        records = face_records("F", side, zc, act)
        lateral = {}                                                     # what slid past each column pair
        for _, i1, _, j1, _, _, _, face in records:
            if face in "XY" and (face, i1, j1) not in lateral:
                i, j = i1 - 1, j1 - 1
                lateral[(face, i1, j1)] = _Slip(*_column_pair(zc, i, j, face), vsh[i, j],
                                                vsh[i + (face == "X"), j + (face == "Y")])
        throws = np.array([slip.throw for slip in lateral.values()] or [0.0])
        dn = float(throws.max()) / max(gross, 1e-6)
        mode, fixed, shift = "seal", None, seal.offset       # fixed: the one multiplier of an open or enhancing fault
        if seal.p_open + seal.p_enhance > 0.0:
            u = rng.random()
            if u < seal.p_open:
                mode, fixed = "open", 1.0
            elif u < seal.p_open + seal.p_enhance:
                mode, fixed = "enhancing", 10.0 ** rng.uniform(*np.log10(seal.enhance))
        if mode == "seal" and seal.scatter > 0.0:
            shift += rng.normal(0.0, seal.scatter)
        factor = 10.0 ** shift
        sin_d = max(math.sin(math.radians(fault.dip)), 1e-3)
        sgrs, mults, weights = [], [], []
        for _, i1, _, j1, _, k1, k2, face in records:
            i, j = i1 - 1, j1 - 1
            if face == "Z":
                near = min(lateral, key=lambda key: abs(key[1] - i1) + abs(key[2] - j1)) if lateral else None
            else:
                near = (face, i1, j1)
            throw = lateral[near].throw if near else 0.0
            for k in range(k1 - 1, k2):
                if face == "Z":
                    below = [kk for kk in range(k + 1, nz) if act[i, j, kk]]
                    if not below:
                        continue
                    other, mid = (i, j, below[0]), bots[i, j, k]
                    li, lj = bots[i, j, k] - tops[i, j, k], bots[other] - tops[other]
                    key, area = "MULTZ", dx * dy
                else:
                    a, b = _column_pair(zc, i, j, face)
                    ta, ba, tb, bb = a[:, :-1].mean(axis=0), a[:, 1:].mean(axis=0), b[:, :-1].mean(axis=0), b[:, 1:].mean(axis=0)
                    mid = 0.5 * (ta[k] + ba[k])
                    o = np.clip(np.minimum(bb, ba[k]) - np.maximum(tb, ta[k]), 0.0, None)
                    i2, j2 = (i + 1, j) if face == "X" else (i, j + 1)
                    other = (i2, j2, int(np.argmax(o)) if o.max() > 0.0 else k)
                    li = lj = dx if face == "X" else dy
                    key = "MULTX" if face == "X" else "MULTY"
                    area = float(o.sum()) * (dy if face == "X" else dx)
                if throw <= 0.0:
                    continue
                sgr = float(lateral[near].sgr(mid))
                d = throw / sin_d
                kf = float(fault_rock_permeability(sgr, d))
                perm = perms[0 if face == "X" else 1 if face == "Y" else 2]
                ki, kj = max(float(perm[i, j, k]), 1e-9), max(float(perm[other]), 1e-9)
                mult = 1.0 / (1.0 + d / seal.dt_ratio * (2.0 / kf - 1.0 / ki - 1.0 / kj) / (max(li, 1e-6) / ki
                                                                                         + max(lj, 1e-6) / kj))
                mult = min(max(mult * factor, 1e-12), 1.0) if fixed is None else fixed
                out[key][i, j, k] *= mult
                if key in listed:
                    listed[key][i, j, k] = True
                sgrs.append(sgr)
                mults.append(mult)
                rows.append((n, "XY".index(near[0]), near[1] - 1, near[2] - 1, mid, sgr, min(ki, kj)))
                weights.append(area / (0.5 * max(li, 1e-6) / ki + 0.5 * max(lj, 1e-6) / kj))      # T0, centre to face
        info.append(dict(name=fault.name, mode=mode, dn=dn, sgr=sgrs, mult=mults,
                         effective=float(np.dot(weights, mults) / sum(weights)) if sum(weights) > 0.0 else 1.0))
    out["faults"] = info
    out["listed"] = listed
    out["face_records"] = np.array(rows, dtype=FACE_RECORD)
    return out


EDGES = ((np.s_[:-1], np.s_[1:]), (np.s_[:, :-1], np.s_[:, 1:]))      # the cells either side of an x edge, a y edge


def _pass_levels(faces, k_top, mults, capillary):
    """The depth below which oil crosses each edge between neighbouring map columns, as arrays for the x edges (nx-1,
    ny) and the y edges (nx, ny-1). An edge nothing splits (the tops of its two columns lie on one side of every fault)
    is free, -inf. Where a fault splits the tops it is a wall, +inf, until a face between two net cells leaks: the
    lowest of their levels, depth + H, with H the oil column ``capillary`` says the face holds."""
    nx, ny = k_top.shape
    gates = [np.full((nx - 1, ny), -np.inf), np.full((nx, ny - 1), -np.inf)]
    for _, side in faces:
        s0 = np.take_along_axis(side, k_top[..., None], axis=2)[..., 0]
        for gate, (lo, hi) in zip(gates, EDGES):
            gate[s0[lo] * s0[hi] == -1] = np.inf
    rec = mults["face_records"]
    rec = rec[rec["perm"] >= capillary.net_perm]
    leak = rec["depth"] + 1e5 * seal_capacity(rec["sgr"], rec["depth"] - capillary.mudline, capillary) / (
        9.81 * capillary.delta_rho)
    for axis, gate in enumerate(gates):
        here = rec["axis"] == axis
        np.minimum.at(gate, (rec["i"][here], rec["j"][here]), leak[here])
    return gates


def _contact_exit(trap, level, spill, depth, gates):
    """How the contact ``level`` of ``trap`` is set: ``(spills, point)``. A flood enters the trap over a cell edge at
    the higher of the edge's pass level and the level of the cell outside; the edges where that is ``level`` set the
    contact. If one of them is unfaulted the block spills over its own rim (``spills``), otherwise the oil leaves across
    a fault. ``point`` is the shallowest cell outside the trap over those edges (the unfaulted ones when there are any),
    None when there are none: nothing reaches the trap."""
    hits = []
    for gate, (lo, hi) in zip(gates, EDGES):
        outside = np.where(trap[lo] & ~trap[hi], spill[hi], np.where(trap[hi] & ~trap[lo], spill[lo], np.inf))
        hits.append(np.maximum(outside, gate) == level)
    free = [hit & (gate == -np.inf) for hit, gate in zip(hits, gates)]
    spills = free[0].any() or free[1].any()
    ring = np.zeros_like(trap)
    for edges, (lo, hi) in zip(free if spills else hits, EDGES):
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
    for gate, (lo, hi) in zip(gates, EDGES):
        a, b = where[lo], where[hi]
        join = (a >= 0) & (b >= 0) & (a != b) & (gate <= np.minimum(contact[a], contact[b]))
        pairs.append(np.stack([a[join], b[join]]))
    a, b = np.concatenate(pairs, axis=1)
    _, comp = csgraph.connected_components(sparse.coo_matrix((np.ones(a.size), (a, b)), shape=(len(blocks),) * 2),
                                           directed=False)
    first = {}
    for blk, c in zip(blocks, comp):
        blk["group"] = first.setdefault(c, len(first))


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
    one contact and one group number.
    """
    nx, ny = act.shape[:2]
    k_top = np.argmax(act, axis=2)
    alive = act.any(axis=2)
    top = 0.25 * (zc[0::2, 0::2] + zc[1::2, 0::2] + zc[0::2, 1::2] + zc[1::2, 1::2])
    depth = np.where(alive, np.take_along_axis(top, k_top[..., None], axis=2)[..., 0], np.nan)
    gates = _pass_levels(faces, k_top, mults, capillary)
    spill = _spill_levels(np.where(alive, depth, np.inf), *gates)
    idx = np.arange(nx * ny).reshape(nx, ny)
    ex = (gates[0] == -np.inf) & alive[:-1] & alive[1:]
    ey = (gates[1] == -np.inf) & alive[:, :-1] & alive[:, 1:]
    rows = np.concatenate([idx[:-1][ex], idx[:, :-1][ey]])
    cols = np.concatenate([idx[1:][ex], idx[:, 1:][ey]])
    graph = sparse.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(nx * ny, nx * ny)).tocsr()
    _, labels = csgraph.connected_components(graph, directed=False)
    labels = labels.reshape(nx, ny)
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
