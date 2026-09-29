"""Levee element painter — port of ``calc_levee.for``.

Per cell: nearest-node distance ``close_distance``; ``cdd = close_distance
- CHhalfwidth``; cutbank vs pointbar via the ``dazi`` sign of (tangent to
cell) — same compass-CW convention as ``calc_levee.for:200-208``.
Asymmetry factor ``1 ± LVasym*|c|/maxc``; thinning ``(1+LVthin) -
2*s_close*LVthin`` (arc-length based); ``LV_w_scale = LVwidth/6``;
profile ``levee_top = LVheight*r*exp(-r) + chelev``,
``levee_bottom = chelev - LVdepth*(LVwidth*factor - close_distance)/LVwidth``.

Stamp ``LV`` for cells where ``levee_bottom <= z <= levee_top`` —
unconditionally (matches AL:235-241; item 1.8).
"""
import numpy as np
from numba import jit

from ._genchannel import find_near_grid


LV = 2  # Alluvsim levee facies code


@jit(nopython=True)
def _paint_levee_kernel(
    nz, mynx, myny, localx, localy, x, y, cy, cx,
    curv, chwidth, chelev_arr, zsiz, dd, distances,
    LV_depth, LV_width, LV_height, LV_asym, LV_thin,
    facies, lk_lv, ntg_counter, max_curv, s_arr, max_s,
    depth_norm, poro_mult_field, log_perm_offset_field,
    ev_poro_mult, ev_log_perm_offset, side_sign,
):
    LV_w_scale = LV_width / 6.0
    ndis = cx.size
    inv_max_curv = 1.0 / max(max_curv, 1e-9)

    for myid in range(localx.size):
        idx = mynx[myid]
        idy = myny[myid]
        idis = dd[myid]
        close_distance = distances[myid]
        cdd = close_distance - chwidth[idis]
        chelev = chelev_arr[idis]

        # Cutbank vs pointbar test, AL ``calc_levee.for:200-208``:
        # ``if (dazi<=0 .and. curv<=0) .or. (dazi>0 .and. curv>0): cutbank``.
        # Cross-product equivalent: side > 0 ↔ dazi > 0 (right of flow). With
        # compass-CW positive curv (right turn) this labels the RIGHT of flow,
        # the inner bank, as cutbank; ``side_sign = -1`` (``cutbank_outer``)
        # mirrors it to the outer bank, where the channel migrates to.
        dx2 = x[idx] - cx[idis]
        dy2 = y[idy] - cy[idis]
        if idis > 0:
            tx = cx[idis] - cx[idis - 1]
            ty = cy[idis] - cy[idis - 1]
        elif idis + 1 < ndis:
            tx = cx[idis + 1] - cx[idis]
            ty = cy[idis + 1] - cy[idis]
        else:
            tx = 1.0
            ty = 0.0
        side = side_sign * (dx2 * ty - dy2 * tx)   # >0: right of tangent (-1 mirrors)
        c = curv[idis]
        # AL same-sign rule: same-sign(side, curv) → cutbank
        if (side <= 0.0 and c <= 0.0) or (side > 0.0 and c > 0.0):
            factor_asym = 1.0 + LV_asym * abs(c) * inv_max_curv
        else:
            factor_asym = 1.0 - LV_asym * abs(c) * inv_max_curv

        # Distal thinning by arc-length ratio (item 2.12)
        if max_s > 1e-9:
            s_frac = s_arr[idis] / max_s
        else:
            s_frac = 0.0
        factor_thin = (1.0 + LV_thin) - 2.0 * s_frac * LV_thin
        factor = factor_asym * factor_thin
        if factor < 1e-3:
            factor = 1e-3

        if close_distance < 0.0:
            levee_top = chelev
            levee_bottom = chelev - LV_depth
        else:
            scale = LV_w_scale * factor
            if scale < 1e-9:
                continue
            r = cdd / scale
            levee_top = LV_height * r * np.exp(-r) + chelev
            denom = max(LV_width, 1e-9)
            levee_bottom = chelev - LV_depth * ((LV_width * factor - close_distance) / denom)

        if levee_top <= levee_bottom:
            continue

        # Stamp LV — unconditionally overwrites (AL ``calc_levee.for:235-241``;
        # item 1.8). NTG count only on FF/FFCH→reservoir transition.
        # AL z-face convention: ``z = zmn + iz*zsiz`` (item 2.14).
        # Levees have no within-deposit upward fining (overbank
        # silt/fine sand) → set depth_norm = 0.5 (neutral, ramp = 1.0).
        for iz in range(nz):
            z_face = (iz + 0.5) * zsiz
            if z_face >= levee_bottom and z_face <= levee_top:
                prev = facies[idx, idy, iz]
                if prev < 1:
                    ntg_counter[0] += 1
                facies[idx, idy, iz] = lk_lv
                depth_norm[idx, idy, iz] = np.float32(0.5)
                poro_mult_field[idx, idy, iz] = np.float32(ev_poro_mult)
                log_perm_offset_field[idx, idy, iz] = np.float32(ev_log_perm_offset)
    return 0


def paint_levee(
    cx: np.ndarray, cy: np.ndarray, curv: np.ndarray,
    chwidth_arr: np.ndarray, chelev_arr,
    LV_depth: float, LV_width: float, LV_height: float,
    LV_asym: float, LV_thin: float,
    x: np.ndarray, y: np.ndarray,
    xsiz: float, ysiz: float, zsiz: float,
    nx: int, ny: int, nz: int,
    facies: np.ndarray, ntg_counter: np.ndarray,
    maxCHhalfwidth: float,
    *, xmn: float = 0.0, ymn: float = 0.0,
    lk_lv: int = LV,
    depth_norm: np.ndarray | None = None,
    poro_mult_field: np.ndarray | None = None,
    log_perm_offset_field: np.ndarray | None = None,
    ev_poro_mult: float = 1.0, ev_log_perm_offset: float = 0.0,
    side_sign: float = 1.0,
):
    """Public entry. No-op if LV is disabled. ``side_sign=-1`` puts the wider,
    cut-bank levee on the outer bank of a bend instead of Alluvsim's inner bank."""
    if LV_width <= 0.0 or (LV_height + LV_depth) <= 0.0:
        return
    if cx is None or cx.size < 3:
        return
    if depth_norm is None:
        depth_norm = np.full((nx, ny, nz), 0.5, dtype=np.float32)
    if poro_mult_field is None:
        poro_mult_field = np.ones((nx, ny, nz), dtype=np.float32)
    if log_perm_offset_field is None:
        log_perm_offset_field = np.zeros((nx, ny, nz), dtype=np.float32)

    if np.isscalar(chelev_arr):
        chelev_arr = np.full(cx.size, float(chelev_arr), dtype=np.float64)
    elif chelev_arr.size != cx.size:
        chelev_arr = np.full(cx.size, float(chelev_arr.mean()), dtype=np.float64)

    # Bbox: maxCHhalfwidth + 1.5*LVwidth (item 2.13, AL:84,147).
    bbox_b = float(maxCHhalfwidth) + 1.5 * LV_width
    good = np.zeros((nx, ny))
    find_near_grid(cx, cy, good, xsiz, ysiz, xmn, ymn, bbox_b, nx, ny)
    mynx, myny = np.where(good == 1)
    if mynx.size == 0:
        return
    localx = x[mynx]
    localy = y[myny]
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([cx, cy]))
    distances, dd = tree.query(np.column_stack([localx, localy]), k=1)

    max_curv = float(np.abs(curv).max()) if curv is not None and curv.size else 1.0

    # Per-node arc-length s_arr (for distal thinning)
    s_arr = np.zeros(cx.size, dtype=np.float64)
    s_arr[1:] = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2)
    s_arr = np.cumsum(s_arr)
    max_s = float(s_arr[-1])

    _paint_levee_kernel(
        nz, mynx, myny, localx, localy, x, y, cy, cx,
        curv, chwidth_arr, chelev_arr, zsiz, dd, distances,
        float(LV_depth), float(LV_width), float(LV_height),
        float(LV_asym), float(LV_thin),
        facies, int(lk_lv), ntg_counter, max_curv, s_arr, max_s,
        depth_norm, poro_mult_field, log_perm_offset_field,
        float(ev_poro_mult), float(ev_log_perm_offset), float(side_sign),
    )
