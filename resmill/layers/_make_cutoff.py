"""Geometric neck-cutoff — port of Alluvsim ``neckcutoff.for``.

Scans every (idis, jdis) pair where ``jdis - idis >= dis_thresh`` to find
two non-adjacent nodes whose Euclidean separation is below ``ctol``. With
nodes a few metres apart, Alluvsim's rule also takes a gently curved stretch
just longer than ``ctol``, whose shortcut stays inside the channel;
``loop_ratio > 1`` also asks the channel between the two nodes to be at least
that many times longer than the gap between them, which only a real loop is
(a neck cutoff removes a loop many times its neck). When
found, deletes nodes (idis, jdis] (the oxbow loop) by shifting the tail
left and reducing ``ndis``. Restarts the scan from the top after every
cut (matches AL's ``goto 435``).

Returns the new (compacted) ``ndis`` so the caller can slice the cx/cy
arrays. If the optional ``idx_map`` integer array is provided it is
compacted alongside cx/cy, so its first ``new_n`` entries identify which
original indices survived. The dropped (oxbow) original indices are then
``set(range(orig_n)) - set(idx_map[:new_n])`` — used by the caller to
locate and stamp FFCH mud plugs at the abandoned bend.
"""
import numpy as np
from numba import jit


@jit(nopython=True)
def _make_cutoff_core(cx, cy, dlength, ctol, idx_map, loop_ratio):
    ndis = cx.size
    if ndis < 4:
        return ndis
    thresh = ctol * ctol
    # dis_thresh = ctol / (avg_arc_length / ndis) + 2  (AL:92-93)
    if ndis > 1:
        s_avg = float(np.sum(dlength)) / max(ndis, 1)
        dis_thresh = int(ctol / max(s_avg, 1e-9)) + 2
    else:
        dis_thresh = 2
    if dis_thresh < 2:
        dis_thresh = 2

    arc = np.zeros(ndis)                    # channel length from node 0 (loop_ratio > 1)
    while True:
        if loop_ratio > 1.0:
            for k in range(1, ndis):
                arc[k] = arc[k - 1] + np.sqrt((cx[k] - cx[k - 1]) ** 2 + (cy[k] - cy[k - 1]) ** 2)
        cut_found = False
        for idis in range(ndis):
            for jdis in range(idis + dis_thresh, ndis):
                xi = cx[idis]
                yi = cy[idis]
                xj = cx[jdis]
                yj = cy[jdis]
                cdist = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)
                if cdist < thresh:
                    if loop_ratio > 1.0 and arc[jdis] - arc[idis] < loop_ratio * np.sqrt(cdist):
                        continue                    # a bend, not a loop
                    # Shift-left compaction: keep [0..idis], drop (idis..jdis], keep [jdis+1..]
                    count = 1
                    for j in range(jdis + 1, ndis):
                        cx[idis + count] = cx[j]
                        cy[idis + count] = cy[j]
                        idx_map[idis + count] = idx_map[j]
                        count += 1
                    ndis = ndis - (jdis - idis)
                    cut_found = True
                    break
            if cut_found:
                break
        if not cut_found:
            break
    return ndis


def make_cutoff(cx, cy, dlength, ctol, idx_map=None, loop_ratio=1.0):
    """Modify cx/cy in place; return new ndis.

    If ``idx_map`` is provided (an int64 ndarray of size cx.size), it is
    compacted alongside cx/cy so its leading ``new_n`` entries identify
    which **original** indices survived the cutoff(s). With ``loop_ratio > 1``
    a cut also needs the channel between the two nodes to be at least
    ``loop_ratio`` times longer than the gap between them (Alluvsim: 1, no check).
    """
    if idx_map is None:
        idx_map = np.arange(cx.size, dtype=np.int64)
    return _make_cutoff_core(cx, cy, dlength, ctol, idx_map, float(loop_ratio))
