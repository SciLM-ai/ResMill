"""Fusion of channels that meet: a channel path that reaches an older path of its own level joins it.

The engine has one active channel at a time, so two paths of one level never run together: a path drawn later (a new avulsion
channel, a new tail) or moved into one by migration cuts straight through an older, abandoned one and both go on, sand on sand,
at 50-90 degrees. Coeval channels do not do that: they merge at a confluence and continue as one, and an avulsion channel that
meets an older course usually reoccupies it and follows it (Smith et al. 1998, Can. J. Earth Sci. 35: 453-466; Aslan & Blum 1999,
IAS SP 28; Hundey & Ashmore 2009, Water Resour. Res. 45: W08409: a braided river is a chain of confluence-bifurcation units, no
thread crosses another). With ``fuse_prob`` a channel that meets an older channel *of the same level* (a younger level cutting
an older one is another generation and stays a crossing) joins it with that probability, decided once for the pair of lineages
(a lineage is the migration history of one path between two births): its path is cut at the crossing, bends into the older
path's heading over a few channel widths and follows it to the edge of the model; otherwise it crosses and meets the next. This
is looked for when the path is born and again after every migration.

The older paths of the level are kept as a raster: the cell each stamped path crossed holds the path (the latest to cross it) and
the segment it crossed on, so a path finds the paths it meets by looking up the cells it runs through, and the exact crossing is
computed on the few segments of an older path next to the cell.
"""
import numpy as np
from numba import njit

MIN_ANGLE = 15.0
"""Degrees: a path crossing an older one at less is running along it, not crossing it [J]."""

BEND_WIDTHS = 2.5
"""Channel widths the junction's bend runs before and after the crossing: the note asks for the last 2-3 widths of the new path
to bend into the older heading [J]."""

REACH_WIDTHS = 3.0
"""Channel widths of its own length a path needs before a crossing counts: a path's first reach, from its birth, still runs with
the path it left or the one that entered the model beside it, so it is not a pair of distinct channels [J]."""

_CHUNK = 64
"""Samples of a path looked up at a time; the search stops at the first chunk that holds a crossing."""


@njit(cache=False)
def burn(owner, node, pid, x, y, x0, y0, dx, dy):
    """Write path ``pid`` into the raster: every cell the polyline (x, y) passes through (sampled every half cell) gets the
    path and the index of the segment it was on. ``(x0, y0)`` is the lower corner of cell (0, 0)."""
    nx, ny = owner.shape
    half = 0.5 * min(dx, dy)
    for i in range(x.size - 1):
        k = max(1, int(np.ceil(np.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) / half)))
        for s in range(k):
            t = s / k
            ix = int(np.floor((x[i] + t * (x[i + 1] - x[i]) - x0) / dx))
            iy = int(np.floor((y[i] + t * (y[i + 1] - y[i]) - y0) / dy))
            if 0 <= ix < nx and 0 <= iy < ny:
                owner[ix, iy] = pid
                node[ix, iy] = i


def _tangent(x, y, i):
    """Unit direction of the polyline at node ``i`` (central difference, one-sided at the ends)."""
    a, b = max(i - 1, 0), min(i + 1, x.size - 1)
    d = np.array([x[b] - x[a], y[b] - y[a]])
    return d / max(np.hypot(*d), 1e-12)


def bend(p0, t0, p1, t1, step):
    """The interior points of the cubic Hermite curve from ``p0`` to ``p1`` leaving along the unit vector ``t0`` and arriving
    along ``t1``, about ``step`` apart."""
    chord = np.hypot(*(p1 - p0))
    t = np.linspace(0.0, 1.0, max(2, int(np.ceil(1.2 * chord / step))) + 1)[1:-1, None]
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * p0 + (t ** 3 - 2 * t ** 2 + t) * chord * t0
            + (-2 * t ** 3 + 3 * t ** 2) * p1 + (t ** 3 - t ** 2) * chord * t1)


class Belts:
    """The channel paths one level has stamped, and where a new path crosses them.

    ``(x0, y0)`` and ``(dx, dy)``: lower corner and size of the cells of the (nx, ny) raster. Paths are kept in the frame the
    engine walks them in; the raster is in the frame they are stamped in, so ``add`` and ``first_crossing`` also take the
    path's stamping-frame coordinates (the same arrays when the model is not rotated).
    """

    def __init__(self, nx, ny, x0, y0, dx, dy):
        self._owner = np.full((nx + 2, ny + 2), -1, dtype=np.int32)      # a cell wider all round: a lookup beside the edge
        self._node = np.zeros((nx + 2, ny + 2), dtype=np.int32)          # finds no path
        self.geometry = (x0, y0, dx, dy)
        self.start = [0]                                                  # path p is xy[start[p]:start[p + 1]]
        self.xy = np.empty((1024, 2))
        self.lineage = []                                                 # the lineage each path belongs to

    @property
    def n_paths(self):
        return len(self.start) - 1

    def add(self, x, y, rx, ry, lineage=0):
        """Keep the path (x, y), stamped at (rx, ry), as the newest of the level, a path of lineage ``lineage``."""
        burn(self._owner[1:-1, 1:-1], self._node[1:-1, 1:-1], self.n_paths, np.asarray(rx, float), np.asarray(ry, float),
             *self.geometry)
        a, b = self.start[-1], self.start[-1] + len(x)
        if b > len(self.xy):
            self.xy = np.concatenate([self.xy, np.empty((max(len(self.xy), b - len(self.xy)), 2))])
        self.xy[a:b, 0], self.xy[a:b, 1] = x, y
        self.start.append(b)
        self.lineage.append(lineage)

    def path(self, pid):
        """The kept path ``pid`` as arrays (x, y)."""
        xy = self.xy[self.start[pid]:self.start[pid + 1]]
        return xy[:, 0], xy[:, 1]

    def first_crossing(self, x, y, rx, ry, start, min_angle=MIN_ANGLE, joins=None):
        """The first place from node ``start`` on where the path (x, y), stamped at (rx, ry), crosses a kept path at ``min_angle``
        degrees or more: ``(segment j, fraction u, path, segment s, fraction v)`` with the two segments the crossing is on, or
        None. ``joins(lineage)``, asked about the crossings in the order they come along the path, passes over those with the
        paths of a lineage it declines (None: the first crossing)."""
        x0, y0, dx, dy = self.geometry
        rx, ry = np.asarray(rx, float), np.asarray(ry, float)
        k = np.maximum(1, np.ceil(np.hypot(np.diff(rx), np.diff(ry)) / (0.5 * min(dx, dy))).astype(int))
        seg = np.repeat(np.arange(rx.size - 1), k)
        frac = (np.arange(seg.size) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(k, k)
        ix = np.floor((rx[seg] + frac * (rx[seg + 1] - rx[seg]) - x0) / dx).astype(int) + 1     # cells of the padded raster
        iy = np.floor((ry[seg] + frac * (ry[seg + 1] - ry[seg]) - y0) / dy).astype(int) + 1
        nx, ny = self._owner.shape
        on = (seg >= start) & (ix >= 1) & (ix < nx - 1) & (iy >= 1) & (iy < ny - 1)
        seg, ix, iy = seg[on], ix[on], iy[on]
        for c in range(0, seg.size, _CHUNK):
            j, pid, node = [], [], []
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    owner = self._owner[ix[c:c + _CHUNK] + a, iy[c:c + _CHUNK] + b]
                    held = owner >= 0
                    j.append(seg[c:c + _CHUNK][held])
                    pid.append(owner[held])
                    node.append(self._node[ix[c:c + _CHUNK] + a, iy[c:c + _CHUNK] + b][held])
            found = self._intersect(x, y, np.concatenate(j), np.concatenate(pid), np.concatenate(node),
                                    np.cos(np.radians(min_angle)), joins)
            if found is not None:
                return found
        return None

    def _intersect(self, x, y, j, pid, n, cos_min, joins=None):
        """The first crossing (by segment of (x, y), then position on it) of segments ``j`` of (x, y) with the segments
        n - 3 ... n + 3 of kept paths ``pid``, at an angle of at least arccos(``cos_min``), that ``joins`` accepts, or None."""
        if j.size == 0:
            return None
        first = np.array(self.start)
        end = len(self.xy) + 1
        key = np.unique(j.astype(np.int64) * end + first[pid] + n)               # each (segment, older node) pair once
        j, g0 = key // end, key % end
        pid = np.searchsorted(first, g0, side="right") - 1
        g = g0[:, None] + np.arange(-3, 4)
        ok = (g >= first[pid][:, None]) & (g + 1 < first[pid + 1][:, None])      # segments with both ends in their path
        j, pid, g = np.broadcast_to(j[:, None], g.shape)[ok], np.broadcast_to(pid[:, None], g.shape)[ok], g[ok]
        d1x, d1y = x[j + 1] - x[j], y[j + 1] - y[j]
        d2x, d2y = self.xy[g + 1, 0] - self.xy[g, 0], self.xy[g + 1, 1] - self.xy[g, 1]
        ex, ey = self.xy[g, 0] - x[j], self.xy[g, 1] - y[j]
        den = d1x * d2y - d1y * d2x
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = (ex * d2y - ey * d2x) / den, (ex * d1y - ey * d1x) / den
        hit = (den != 0.0) & (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0) \
            & (np.abs(d1x * d2x + d1y * d2y) < cos_min * np.hypot(d1x, d1y) * np.hypot(d2x, d2y))
        for best in np.flatnonzero(hit)[np.lexsort((u[hit], j[hit]))]:
            if joins is None or joins(self.lineage[pid[best]]):
                return int(j[best]), float(u[best]), int(pid[best]), int(g[best] - first[pid[best]]), float(v[best])
        return None


def fuse(belts, x, y, rx, ry, width, step, start, max_angle, joins=None):
    """The path (x, y) joined to the first older path it crosses from node ``start`` on and ``joins`` (see
    :meth:`Belts.first_crossing`), or None when there is none.

    The new path is kept up to ``BEND_WIDTHS`` channel widths before the crossing, then a smooth bend carries it into the older
    path ``BEND_WIDTHS`` widths beyond the crossing, and it follows the older path from there to its end. The bend leaves along
    the new path's heading, turned towards the older heading when they differ by more than ``max_angle`` degrees (a confluence
    joins at an acute angle), and arrives along the older heading, so the path never turns more than the angle between the two
    in one bend. ``width``: the channel width, ``step``: the spacing of the nodes.
    """
    crossing = belts.first_crossing(x, y, rx, ry, start, joins=joins)
    if crossing is None:
        return None
    j, u, pid, s, v = crossing
    ax, ay = belts.path(pid)
    reach = max(2, int(round(BEND_WIDTHS * width / step)))
    i0, q = max(j - reach, 0), min(s + 1 + reach, ax.size - 1)
    t0, t1 = _tangent(x, y, i0), _tangent(ax, ay, q)
    turn = np.arctan2(t0[0] * t1[1] - t0[1] * t1[0], t0 @ t1)                 # from the new path's heading to the older one's
    if abs(turn) > np.radians(max_angle):
        a = np.arctan2(t1[1], t1[0]) - np.sign(turn) * np.radians(max_angle)
        t0 = np.array([np.cos(a), np.sin(a)])
    inner = bend(np.array([x[i0], y[i0]]), t0, np.array([ax[q], ay[q]]), t1, step)
    return (np.concatenate([x[:i0 + 1], inner[:, 0], ax[q:]]), np.concatenate([y[:i0 + 1], inner[:, 1], ay[q:]]))
