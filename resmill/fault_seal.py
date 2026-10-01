"""Fault seal per cell face, and fault blocks.

A fault seals or leaks by what lies on its plane. Per face of a fault (the faces ``FAULTS`` lists), the shale gouge
ratio SGR = sum(Vsh dz) / throw over the beds that slid past the face (Yielding, Freeman & Needham 1997) sets the fault
rock's permeability, log10 k_f = -4 SGR - 1/4 log10(D) (1 - SGR)^5 (mD; D the displacement in m), and its thickness
t_f = D / 66, the median ratio (Manzocchi, Walsh, Nell & Yielding 1999). The face's transmissibility multiplier is then
T = [1 + t_f (2/k_f - 1/k_i - 1/k_j) / (L_i/k_i + L_j/k_j)]^-1 between the cells on either side (permeabilities k_i and
k_j across the face, half-lengths L_i and L_j), at most 1, so every face lies somewhere between open and sealed. The
throw at a face is read off the grid, as the offset of the same layer across it, so the throws of several faults add.
The physics shapes each fault but explains none of the spread of real ones (on Norne the predicted and the
history-matched multipliers are uncorrelated), so all faces of a fault also share one log-normal factor 10^(``offset`` +
N(0, ``scatter``)), calibrated on Norne's history match (``scatter`` 0.9 and ``offset`` -0.6 on its own grid, the mean
of three fits to its 36 faults with throw). A share of faults is left open (``p_open``: every face 1) and a share raises
flow (``p_enhance``: every face one log-uniform value of ``enhance``, above 1), as Norne's history match has (5 of its
36 faults with throw at 1 or above, up to 3.9, and a fault without throw at 20). The share of faults that seal by Dn =
throw / gross reservoir thickness in Knott's (1993) North Sea counts (:func:`knott_seal_probability`) is a check on the
calibration. A fault draws in this order, so a seed reproduces a tree: one uniform for its mode when either share is
above 0, then one log-uniform value (enhancing) or one normal (seal, with a ``scatter``); with the options off nothing
is drawn.

Juxtaposition needs nothing here: the simulator connects only the cells that touch. :func:`fault_blocks` splits the
top surface into blocks bounded by sealing faces (multiplier below 0.01, Norne's sealing range) or by a throw beyond
the reservoir's thickness (nothing left to touch), and reports each block's trap: crest, spill depth and point, area
and height.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .faults import face_records
from .structure import _spill_levels

SEALING = 0.01                      # a face this tight bounds a fault block (Norne's history-matched sealing range)


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


def fault_rock_permeability(sgr, displacement):
    """Fault-rock permeability (mD) from the shale gouge ratio and the displacement (m) (Manzocchi et al. 1999)."""
    d = np.maximum(np.asarray(displacement, dtype=float), 1e-3)
    return 10.0 ** (-4.0 * np.asarray(sgr, dtype=float) - 0.25 * np.log10(d) * (1.0 - np.asarray(sgr)) ** 5)


def knott_seal_probability(dn):
    """Share of faults that seal, by throw over gross reservoir thickness: Knott (1993), the northern and southern North
    Sea counts averaged per class (over 0.9 and all for Dn >= 1; 1 in 2 and 82 % for 0.5-1; 1 in 3 to 1 in 2 and 66 %
    for 0.25-0.5; 1 in 3 and 14 % below 0.25)."""
    return 0.92 if dn >= 1.0 else 0.66 if dn >= 0.5 else 0.5 if dn >= 0.25 else 0.24                  # [J] (averages)


def _column_pair(zc, i, j, face):
    """The two columns' depths on a shared face: (2 corners, nk) for the lower-index and the upper-index column."""
    if face == "X":
        return zc[2 * i + 1, 2 * j:2 * j + 2], zc[2 * i + 2, 2 * j:2 * j + 2]
    return zc[2 * i:2 * i + 2, 2 * j + 1], zc[2 * i:2 * i + 2, 2 * j + 2]


def _window_vsh(tops, bots, vsh, lo, hi):
    """Clay times thickness, and thickness, of a column's cells inside the depth window [lo, hi]."""
    overlap = np.clip(np.minimum(bots, hi) - np.maximum(tops, lo), 0.0, None)
    return float((overlap * vsh).sum()), float(overlap.sum())


def face_multipliers(faces, zc, act, vsh, perms, dx, dy, seal, thickness=None):
    """Transmissibility multipliers per cell face from the faults' shale gouge ratio.

    ``faces`` holds ``(fault, side)`` pairs (from the geometry builder), ``zc`` the final interface stack, ``act`` the
    active cells, ``vsh`` the clay fraction per cell and ``perms`` (PERMX, PERMY, PERMZ), all (nx, ny, nz) in K-down
    order. Returns ``MULTX``, ``MULTY``, ``MULTZ`` (nx, ny, nz; a cell's + face, as the GRDECL keywords mean), the
    faces each fault listed (``listed``), and per fault its ``mode`` ("seal", "open" or "enhancing"), ``dn``, ``sgr``
    and ``mult`` per face and ``effective``, the faces' mean weighted by the transmissibility each would have without
    the fault: the one multiplier for the whole fault (a MULTFLT) that passes the same flow.
    """
    nx, ny, nz = act.shape
    rng = np.random.default_rng(seal.seed)
    out = {key: np.ones((nx, ny, nz)) for key in ("MULTX", "MULTY", "MULTZ")}
    listed = {"MULTX": np.zeros((nx, ny, nz), dtype=bool), "MULTY": np.zeros((nx, ny, nz), dtype=bool)}
    cell = 0.25 * (zc[0::2, 0::2] + zc[1::2, 0::2] + zc[0::2, 1::2] + zc[1::2, 1::2])
    tops, bots = cell[..., :-1], cell[..., 1:]
    gross = thickness if thickness is not None else float(np.median((zc[:, :, -1] - zc[:, :, 0])))
    info = []
    for fault, side in faces:
        records = face_records("F", side, zc, act)
        lateral = {}                                                     # the throw across each column pair
        for _, i1, _, j1, _, _, _, face in records:
            if face in "XY" and (face, i1, j1) not in lateral:
                a, b = _column_pair(zc, i1 - 1, j1 - 1, face)
                lateral[(face, i1, j1)] = float(np.abs(a - b).mean(axis=0).max())     # a dipping plane offsets only some layers
        throws = np.array(list(lateral.values()) or [0.0])
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
                throw = lateral[near] if near else 0.0
            else:
                throw = lateral[(face, i1, j1)]
            for k in range(k1 - 1, k2):
                if face == "Z":
                    below = [kk for kk in range(k + 1, nz) if act[i, j, kk]]
                    if not below:
                        continue
                    other, mid = (i, j, below[0]), bots[i, j, k]
                    li, lj = 0.5 * (bots[i, j, k] - tops[i, j, k]), 0.5 * (bots[other] - tops[other])
                    cols = [(tops[i, j], bots[i, j], vsh[i, j])]
                    key, area = "MULTZ", dx * dy
                else:
                    a, b = _column_pair(zc, i, j, face)
                    ta, ba, tb, bb = a[:, :-1].mean(axis=0), a[:, 1:].mean(axis=0), b[:, :-1].mean(axis=0), b[:, 1:].mean(axis=0)
                    mid = 0.5 * (ta[k] + ba[k])
                    o = np.clip(np.minimum(bb, ba[k]) - np.maximum(tb, ta[k]), 0.0, None)
                    i2, j2 = (i + 1, j) if face == "X" else (i, j + 1)
                    other = (i2, j2, int(np.argmax(o)) if o.max() > 0.0 else k)
                    li = lj = 0.5 * (dx if face == "X" else dy)
                    cols = [(ta, ba, vsh[i, j]), (tb, bb, vsh[i2, j2])]
                    key = "MULTX" if face == "X" else "MULTY"
                    area = float(o.sum()) * (dy if face == "X" else dx)
                if throw <= 0.0:
                    continue
                parts = [_window_vsh(t, b_, v, mid - 0.5 * throw, mid + 0.5 * throw) for t, b_, v in cols]
                weight = sum(p[1] for p in parts)
                sgr = sum(p[0] for p in parts) / weight if weight > 0.0 else 0.0
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
                weights.append(area / (max(li, 1e-6) / ki + max(lj, 1e-6) / kj))
        info.append(dict(name=fault.name, mode=mode, dn=dn, sgr=sgrs, mult=mults,
                         effective=float(np.dot(weights, mults) / sum(weights)) if sum(weights) > 0.0 else 1.0))
    out["faults"] = info
    out["listed"] = listed
    return out


def fault_blocks(zc, act, faces, mults, dx, dy):
    """The top surface's fault blocks and each block's trap.

    Two neighbouring columns lie in different blocks where a fault separates their tops and no connection between
    them across it, at any depth, is open (every listed face's multiplier in ``mults``, from
    :func:`face_multipliers`, below :data:`SEALING`, or no face at all: a throw beyond the reservoir leaves nothing to
    touch). Oil spills only between columns of one block and over the map's edge; a block sealed all round fills to
    its deepest point.
    Returns one dict per block holding a trap, shallowest crest first: ``crest`` (i, j), ``crest_depth``,
    ``spill_depth``, ``spill`` (i, j, or None when sealed all round), ``area`` (m2), ``height`` (m) and ``mask``.
    """
    nx, ny, nz = act.shape
    k_top = np.argmax(act, axis=2)
    alive = act.any(axis=2)
    top = 0.25 * (zc[0::2, 0::2] + zc[1::2, 0::2] + zc[0::2, 1::2] + zc[1::2, 1::2])
    depth = np.where(alive, np.take_along_axis(top, k_top[..., None], axis=2)[..., 0], np.nan)
    open_x = np.where(mults["listed"]["MULTX"], mults["MULTX"], -np.inf).max(axis=2) >= SEALING
    open_y = np.where(mults["listed"]["MULTY"], mults["MULTY"], -np.inf).max(axis=2) >= SEALING
    wall_x, wall_y = np.zeros((nx, ny), dtype=bool), np.zeros((nx, ny), dtype=bool)
    for _, side in faces:
        s0 = np.take_along_axis(side, k_top[..., None], axis=2)[..., 0]
        wall_x[:-1] |= (s0[:-1] * s0[1:] == -1) & ~open_x[:-1]
        wall_y[:, :-1] |= (s0[:, :-1] * s0[:, 1:] == -1) & ~open_y[:, :-1]
    wall_x[-1], wall_y[:, -1] = True, True
    filled = np.where(alive, depth, np.inf)
    spill = _spill_levels(filled, wall_x[:-1], wall_y[:, :-1])
    idx = np.arange(nx * ny).reshape(nx, ny)
    ex = ~wall_x[:-1] & alive[:-1] & alive[1:]
    ey = ~wall_y[:, :-1] & alive[:, :-1] & alive[:, 1:]
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
        ring = np.zeros_like(trap)
        ring[:-1] |= trap[1:] & ~wall_x[:-1]
        ring[1:] |= trap[:-1] & ~wall_x[:-1]
        ring[:, :-1] |= trap[:, 1:] & ~wall_y[:, :-1]
        ring[:, 1:] |= trap[:, :-1] & ~wall_y[:, :-1]
        ring &= ~trap & alive
        point = (np.unravel_index(int(np.argmin(np.where(ring, depth, np.inf))), depth.shape)
                 if np.isfinite(spill[crest]) and ring.any() else None)
        blocks.append(dict(crest=(int(crest[0]), int(crest[1])), crest_depth=float(depth[crest]),
                           spill_depth=float(level), spill=None if point is None else (int(point[0]), int(point[1])),
                           area=float(trap.sum()) * dx * dy, height=float(level - depth[crest]), mask=trap))
    return sorted(blocks, key=lambda blk: blk["crest_depth"])
