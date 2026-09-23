"""Active-channel U-shape stamp (port of Alluvsim ``genchannel.for``).

Paints one streamline's Deutsch-Wang asymmetric U-shape cross-section into
``facies`` for the **active channel** (CH=4) and **lateral-accretion** (LA=3)
elements. Mud-plug abandonment is in ``_genabandoned.py``.

Per-cell: find nearest streamline node, refine via 5-step ``splint`` walk to
sub-node accuracy (matches AL ``genchannel.for:146-167``), evaluate
``chelev``, ``chwidth`` and ``thalweg`` at the refined arc-length, decide
which side of the centreline the cell sits on (cross-product on tangent),
build the asymmetric U-shape ``CHbot``, and stamp facies for cells in
``(CHbot, CHelev]``.

When ``erode_above`` is True (active CH) and the cell sits *above*
``CHelev`` with an existing reservoir code, it is reset to FF — port of
``genchannel.for:213-218``.
"""
import math

import numpy as np
from numba import jit


@jit(nopython=True)
def find_near_grid(cx, cy, good, xsiz, ysiz, xmn, ymn, b, nx, ny):
    """Mark cells within ~6b of any centerline node (search bbox).

    Uses xmn/ymn cell-origin offset (item 1.14 fix). Per-axis bbox radius:
    ``int(6*b / xsiz)`` for x, ``int(6*b / ysiz)`` for y — isotropic in
    physical units even on anisotropic grids.
    """
    ndx = max(int(6.0 * b / xsiz), 1)
    ndy = max(int(6.0 * b / ysiz), 1)
    for i in range(cx.size):
        mynx = int((cx[i] - xmn) / xsiz)
        myny = int((cy[i] - ymn) / ysiz)
        for j in range(max(0, mynx - ndx), min(nx, mynx + ndx + 1)):
            for k in range(max(0, myny - ndy), min(ny, myny + ndy + 1)):
                good[j, k] = 1
    return 0


@jit(nopython=True)
def mychannel(nz, mynx, myny, localx, localy, x, y, vx, vy, cy, cx,
              thalweg, chelev_arr, zsiz, dd, facies, poro, chwidth,
              dist_arr, dwratio, merge_overlap, facies_code, ntg_counter,
              compute_poro, erode_above, poro0,
              depth_norm, poro_mult_field, log_perm_offset_field,
              ev_poro_mult, ev_log_perm_offset):
    """Paint one streamline's U-shape into facies + per-cell auxiliary fields.

    Writes ``depth_norm[ix,iy,iz]`` = (chelev - z_face) / (chelev - chbot)
    for every CH/LA cell stamped — vertical position within this channel
    event's cross-section, used by ``_finalize_facies_table`` to apply
    the Walker-1992 upward-fining ramp PER EVENT (no per-column leakage).
    Erode-above branch resets depth_norm to neutral 0.5.

    Per-event scalars ``ev_poro_mult`` / ``ev_log_perm_offset`` are stamped
    into ``poro_mult_field`` / ``log_perm_offset_field`` at every cell this
    event paints — last-event-wins overwrite, matching facies semantics.
    """
    for myid in range(localx.size):
        idx = mynx[myid]
        idy = myny[myid]
        idis = dd[myid]
        dist = dist_arr[myid]
        # Local-node halfwidth gate (item 1.6)
        if dist > chwidth[idis]:
            continue

        t = dwratio * chwidth[idis]
        WW = chwidth[idis] * 2.0
        chelev = chelev_arr[idis]

        dx2 = x[idx] - cx[idis]
        dy2 = y[idy] - cy[idis]
        indicator = dx2 * vy[idis] - dy2 * vx[idis]
        if indicator > 0:
            wid = dist + chwidth[idis]
        else:
            wid = chwidth[idis] - dist
        # Out-of-channel skip (item 1.15)
        if wid < 0.0 or wid > WW:
            continue
        # AL clamp: a in [1e-3, 1-1e-3] (item 4.2)
        a = min(0.999, max(0.001, thalweg[idis]))

        if a < 0.5:
            by = -np.log(2.0) / np.log(a)
            chbot = chelev - 4.0 * t * (wid / WW)**by * (1.0 - (wid / WW)**by)
            wid2 = a * WW
            maxD = 4.0 * t * (wid2 / WW)**by * (1.0 - (wid2 / WW)**by)
        else:
            cy_ = -np.log(2.0) / np.log(1.0 - a)
            chbot = chelev - 4.0 * t * (1.0 - wid / WW)**cy_ * (1.0 - (1.0 - wid / WW)**cy_)
            wid2 = a * WW
            maxD = 4.0 * t * (1.0 - wid2 / WW)**cy_ * (1.0 - (1.0 - wid2 / WW)**cy_)

        # Erode residual reservoir code above the channel surface (AL:213-218)
        # only for active CH stamps. Iterate every iz; gate on z_face > chelev.
        if erode_above and facies_code >= 1:
            for iz in range(nz):
                z_face = (iz + 0.5) * zsiz  # cell centre
                if z_face > chelev and facies[idx, idy, iz] >= 1:
                    ntg_counter[0] -= 1
                    facies[idx, idy, iz] = -1  # FF
                    depth_norm[idx, idy, iz] = np.float32(0.5)
                    poro_mult_field[idx, idy, iz] = np.float32(1.0)
                    log_perm_offset_field[idx, idy, iz] = np.float32(0.0)

        # Per-event ramp denominator: thalweg max depth ``maxD`` (constant
        # across all cells stamped by this event, irrespective of lateral
        # position w). Using maxD gives a geologically correct
        # thalweg-centred 3D fall-off: depth_norm peaks at the deepest
        # point of the channel cross-section (thalweg base) and falls
        # off both upward AND laterally toward the banks. The U-shape
        # geometry constrains near-bank cells to live only in the thin
        # near-surface sliver where ``chelev - z_face`` is small, so they
        # automatically get small depth_norm without needing a separate
        # lateral coordinate. (Earlier per-cell formula
        # ``(chelev - chbot(w))`` was a bug — it normalised each lateral
        # column to its own pinched extent, so bank cells incorrectly
        # ramped to 1.0 at their local bottom.)
        ramp_denom = maxD
        if ramp_denom < 1e-9:
            ramp_denom = 1e-9

        # Stamp the U-shape body.  Per-cell continuous-z criterion
        # (AL:219 ``if z > chbot and z < chelev``).
        for iz in range(nz):
            z_face = (iz + 0.5) * zsiz
            if z_face > chbot and z_face < chelev:
                if facies[idx, idy, iz] < 1 and facies_code >= 1:
                    ntg_counter[0] += 1
                facies[idx, idy, iz] = facies_code

                # 3D radial fall-off from the thalweg-base point of THIS
                # event's U-shape — geologically the highest-energy
                # depositional locus. Two factors, multiplicatively
                # combined (matches the lobe-layer's centroid-radial
                # gradient pattern):
                #
                #   v = (chelev - z_face) / maxD                      [0..1]
                #       1 at the channel base (thalweg bottom),
                #       0 at the channel top.
                #
                #   l = 1 - lateral_distance_from_thalweg / max_side  [0..1]
                #       1 at the thalweg lateral position (wid = a·WW),
                #       0 at either bank (wid = 0 or wid = WW).
                #       Asymmetric U-shapes (a ≠ 0.5) normalise each
                #       side independently so both banks reach 0.
                #
                # depth_norm = v · l → maximum at thalweg base, smoothly
                # decaying upward AND laterally outward — exactly the
                # 3D lobe-style envelope the user asked for.
                v = (chelev - z_face) / ramp_denom
                if v < 0.0:
                    v = 0.0
                elif v > 1.0:
                    v = 1.0
                # Lateral distance from thalweg in [0, 1]
                thalweg_w = a * WW
                if wid < thalweg_w:
                    half = thalweg_w
                else:
                    half = WW - thalweg_w
                if half < 1e-9:
                    ld = 0.0
                else:
                    ld = abs(wid - thalweg_w) / half
                if ld < 0.0:
                    ld = 0.0
                elif ld > 1.0:
                    ld = 1.0
                l_factor = 1.0 - ld
                dn = np.float32(v * l_factor)
                depth_norm[idx, idy, iz] = dn
                poro_mult_field[idx, idy, iz] = np.float32(ev_poro_mult)
                log_perm_offset_field[idx, idy, iz] = np.float32(ev_log_perm_offset)

                if compute_poro:
                    if maxD > 1e-9:
                        depth_into_channel = chelev - z_face
                        # Scalar local — distinct from the per-cell ``depth_norm``
                        # field above (renamed to avoid Numba type-unification
                        # collision with the parameter).
                        dn_local = depth_into_channel / max(maxD, 1e-9)
                        new_poro = (
                            (0.9 / (1.0 + np.exp(-4.0 * (dn_local - 0.2))) + 0.1)
                            * poro0 * min((chelev - chbot) / max(maxD, 1e-9), 1.0)**2 + 0.1
                        )
                        if merge_overlap:
                            if new_poro > poro[idx, idy, iz]:
                                poro[idx, idy, iz] = new_poro
                        else:
                            poro[idx, idy, iz] = new_poro
    return 0


@jit(nopython=True)
def nearest_refined(cx, cy, x_loc, y_loc, ndiscr, two):
    """Per local cell: index of the nearest streamline node (first minimum of the
    Euclidean distance, exactly as ``sqrt((cx-x)**2+(cy-y)**2).argmin(axis=0)``
    on the full node x cell matrix) and the ``_refine_nearest`` sub-node
    distance. Bit-identical to the former matrix + Python-loop path: the
    search squares with a multiply like NumPy's ``**2``, the refinement calls
    ``math.pow(., two)`` with ``two`` = 2.0 passed at run time so it goes
    through libm ``pow`` like CPython's ``float ** 2`` (a literal exponent
    would be folded to a multiply, which differs in the last bit).
    """
    n_local = x_loc.size
    n_nodes = cx.size
    dd = np.empty(n_local, dtype=np.int64)
    dist = np.empty(n_local, dtype=np.float64)
    guess = 0
    for myid in range(n_local):
        lx = x_loc[myid]
        ly = y_loc[myid]
        # Bound from the previous cell's nearest node (cells come in raster
        # order, so it is usually close). A node with |dx| or |dy| beyond
        # bound * (1 + 1e-12) has d > bound >= the true minimum and can never
        # be the first strict minimum, so skipping it leaves the result of
        # the in-order scan below unchanged; the 1e-12 covers the rounding
        # of sqrt(dx*dx + dy*dy) against |dx|.
        gx = cx[guess] - lx
        gy = cy[guess] - ly
        bound = np.sqrt(gx * gx + gy * gy) * (1.0 + 1e-12)
        best_d = np.inf
        best_i = 0
        for i in range(n_nodes):
            dx = cx[i] - lx
            if dx > bound or -dx > bound:
                continue
            dy = cy[i] - ly
            if dy > bound or -dy > bound:
                continue
            d = np.sqrt(dx * dx + dy * dy)
            if d < best_d:
                best_d = d
                best_i = i
        idis = best_i
        guess = idis
        lo = max(0, idis - 1)
        hi = min(n_nodes - 1, idis + 1)
        best = math.pow(cx[idis] - lx, two) + math.pow(cy[idis] - ly, two)
        for sub in range(1, ndiscr):
            t = sub / float(ndiscr)
            xt = (1.0 - t) * cx[lo] + t * cx[hi]
            yt = (1.0 - t) * cy[lo] + t * cy[hi]
            d2 = math.pow(xt - lx, two) + math.pow(yt - ly, two)
            if d2 < best:
                best = d2
        dd[myid] = idis
        dist[myid] = np.sqrt(max(best, 0.0))
    return dd, dist


def _refine_nearest(cx, cy, x_loc, y_loc, dd_initial, ndiscr=5):
    """5-step ``splint``-equivalent walk to sub-node accuracy (item 2.10).

    For each cell, given the closest discrete node ``dd_initial[myid]``, walk
    ``ndiscr`` linearly-interpolated sub-positions between the neighbouring
    nodes and return the (refined node index, refined arc-length distance).

    Linear interpolation between adjacent nodes is sufficient here (the
    streamline has already been spline-resampled to ndis0 uniform spacing
    in ``cal_curv``); the spline curve passes through these nodes.
    """
    n_local = dd_initial.size
    n_nodes = cx.size
    refined_dist = np.empty(n_local, dtype=np.float64)
    for myid in range(n_local):
        idis = int(dd_initial[myid])
        lo = max(0, idis - 1)
        hi = min(n_nodes - 1, idis + 1)
        best = (cx[idis] - x_loc[myid])**2 + (cy[idis] - y_loc[myid])**2
        for sub in range(1, ndiscr):
            t = sub / float(ndiscr)
            xt = (1.0 - t) * cx[lo] + t * cx[hi]
            yt = (1.0 - t) * cy[lo] + t * cy[hi]
            d = (xt - x_loc[myid])**2 + (yt - y_loc[myid])**2
            if d < best:
                best = d
        refined_dist[myid] = float(np.sqrt(max(best, 0.0)))
    return refined_dist


def genchannel(b, xsiz, ysiz, chelev_arr, zsiz, nx, ny, nz, cx, cy, x, y,
               vx, vy, curv, LV_asym, lV_height, ps, pavul, out, totalid,
               facies, poro, poro0, thalweg, chwidth, dwratio, cutoff, NN=800,
               *, xmn=0.0, ymn=0.0,
               merge_overlap=False, facies_code=1, ntg_counter=None,
               compute_poro=True, erode_above=False,
               depth_norm=None, poro_mult_field=None,
               log_perm_offset_field=None,
               ev_poro_mult=1.0, ev_log_perm_offset=0.0):
    """Public wrapper around ``mychannel`` — Alluvsim ``genchannel.for`` entry.

    ``chelev_arr`` is the per-node CHelev array (item 2.11). For a flat
    channel it can be a constant array; for graded channels it varies.

    ``depth_norm`` / ``poro_mult_field`` / ``log_perm_offset_field`` are
    per-cell auxiliary fields (float32, shape ``(nx,ny,nz)``) that the
    kernel writes alongside ``facies``. They drive the Walker upward-
    fining ramp + per-event poro/perm variability in
    ``_finalize_facies_table``. ``ev_poro_mult`` / ``ev_log_perm_offset``
    are the freshly-drawn per-event scalars (one pair per stamp call).
    Tests / standalone callers can omit these (None → no aux output).
    """
    if ntg_counter is None:
        ntg_counter = np.zeros(1, dtype=np.int64)
    if np.isscalar(chelev_arr):
        chelev_arr = np.full(cx.size, float(chelev_arr), dtype=np.float64)
    elif chelev_arr.size != cx.size:
        chelev_arr = np.full(cx.size, float(chelev_arr.mean()), dtype=np.float64)
    # Provide neutral-default aux arrays so the kernel signature is
    # always satisfied (tests in tests/ don't pass these).
    if depth_norm is None:
        depth_norm = np.full((nx, ny, nz), 0.5, dtype=np.float32)
    if poro_mult_field is None:
        poro_mult_field = np.ones((nx, ny, nz), dtype=np.float32)
    if log_perm_offset_field is None:
        log_perm_offset_field = np.zeros((nx, ny, nz), dtype=np.float32)
    ndis = cx.size
    good = np.zeros((nx, ny))
    find_near_grid(cx, cy, good, xsiz, ysiz, xmn, ymn, b, nx, ny)
    mynx, myny = np.where(good == 1)
    if mynx.size == 0:
        return 0
    localx = x[mynx]
    localy = y[myny]
    # Nearest node + sub-node refinement (item 2.10) in one pass; replaces
    # the ndis x n_local distance matrix, its argmin and a Python loop.
    dd, dist_arr = nearest_refined(cx, cy, localx, localy, 5, 2.0)

    mychannel(nz, mynx, myny, localx, localy, x, y, vx, vy, cy, cx, thalweg,
              chelev_arr, zsiz, dd, facies, poro, chwidth, dist_arr,
              dwratio, bool(merge_overlap), int(facies_code), ntg_counter,
              bool(compute_poro), bool(erode_above), float(poro0),
              depth_norm, poro_mult_field, log_perm_offset_field,
              float(ev_poro_mult), float(ev_log_perm_offset))
    return 0
