"""Helpers shared by the fusion and braid tests: every channel path a build stamps, and the places where two distinct channels
of one level cross.

``PathLog`` wraps three engine methods (nothing in the engine records paths): each channel (CH) stamp is logged with its level and
its birth, the number of the path draw (pool draw or avulsion inside) it descends from, so a *lineage* is the migration history of
one path between two births, as in the research note on channel crossings (design_notes/channel_crossings_braiding.md).

``crossing_sites`` counts the crossings of two lineages' representative paths (their median stamp) that are two distinct channels,
not one reach shared by a join or a split: at least ``min_angle`` degrees apart, and ``reach`` channel widths along either path on
either side of the crossing that path is still at least one width from the whole belt (every stamp) of the other (a side past the
end of a path is not tested). Sites within one width of each other are one place.
"""
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from resmill.layers import _fluvial

CH = _fluvial.CH


class PathLog:
    """Record the channel paths of every engine built while it is installed (``with PathLog(monkeypatch) as log``)."""

    def __init__(self, monkeypatch):
        self.paths = []                 # (level, lineage, cx, cy), walk frame, one per channel stamp, in time order
        self.births = 0
        cls, log = _fluvial.fluvial, self
        stamp, pool, inside = cls._stamp_channel, cls._draw_from_pool, cls._avulse_inside

        def _stamp_channel(engine, facies_code, erode_above):
            out = stamp(engine, facies_code, erode_above)
            if facies_code == CH and engine.cx is not None and engine.cx.size >= 3:
                log.paths.append((int(engine._level), log.births, engine.cx.copy(), engine.cy.copy()))
            return out

        def _draw_from_pool(engine):
            ok = pool(engine)
            log.births += bool(ok)
            return ok

        def _avulse_inside(engine):
            ok = inside(engine)
            log.births += bool(ok)
            return ok

        monkeypatch.setattr(cls, "_stamp_channel", _stamp_channel)
        monkeypatch.setattr(cls, "_draw_from_pool", _draw_from_pool)
        monkeypatch.setattr(cls, "_avulse_inside", _avulse_inside)

    def lineages(self):
        """{(level, lineage): [(cx, cy), ...]} in time order."""
        out = defaultdict(list)
        for level, lineage, cx, cy in self.paths:
            out[(level, lineage)].append((cx, cy))
        return out


def polyline_crossings(a, b):
    """Where polyline ``a`` = (x, y) crosses polyline ``b``: arrays px, py, segment of a, segment of b, angle (degrees, 0-90)."""
    (ax, ay), (bx, by) = a, b
    d1x, d1y, d2x, d2y = np.diff(ax)[:, None], np.diff(ay)[:, None], np.diff(bx)[None, :], np.diff(by)[None, :]
    ex, ey = bx[None, :-1] - ax[:-1, None], by[None, :-1] - ay[:-1, None]
    den = d1x * d2y - d1y * d2x
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = (ex * d2y - ey * d2x) / den, (ex * d1y - ey * d1x) / den
        cos = np.abs(d1x * d2x + d1y * d2y) / np.hypot(d1x, d1y) / np.hypot(d2x, d2y)
    hit = (den != 0) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    ia, ib = np.nonzero(hit)
    return (ax[ia] + u[ia, ib] * d1x[ia, 0], ay[ia] + u[ia, ib] * d1y[ia, 0], ia, ib,
            np.degrees(np.arccos(np.clip(cos[ia, ib], 0.0, 1.0))))


def crossing_sites(log, width, step, min_angle=20.0, reach=3.0):
    """The places where two distinct channels cross: ``{"same": sites, "different": sites}`` with ``sites`` = ``(x, y, level a,
    level b, angle)`` arrays, for the pairs of one level (the crossings fusion removes) and of different levels (the ones it
    keeps).

    ``width`` is the channel width and ``step`` the spacing of a path's nodes (m); every third node is looked at.
    """
    lineages = log.lineages()
    rep = {key: tuple(c[::3] for c in events[len(events) // 2]) for key, events in lineages.items()}
    belt = {key: cKDTree(np.column_stack([np.concatenate([e[0][::3] for e in events]), np.concatenate([e[1][::3] for e in events])]))
            for key, events in lineages.items()}
    box = {key: (path[0].min(), path[0].max(), path[1].min(), path[1].max()) for key, path in rep.items()}
    m = max(2, int(round(reach * width / (3 * step))))
    keys, sites = sorted(rep), {"same": [], "different": []}
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            a, b = box[ka], box[kb]
            if a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]:
                continue
            px, py, ia, ib, ang = polyline_crossings(rep[ka], rep[kb])
            for k in np.nonzero(ang >= min_angle)[0]:
                apart = True
                for other, seg, path in ((kb, ia[k], rep[ka]), (ka, ib[k], rep[kb])):
                    for node in (seg - m, seg + 1 + m):
                        if 0 <= node < path[0].size and belt[other].query((path[0][node], path[1][node]))[0] < width:
                            apart = False
                if apart:
                    sites["same" if ka[0] == kb[0] else "different"].append((px[k], py[k], ka[0], kb[0], ang[k]))
    return {kind: _one_per_place(found, width) for kind, found in sites.items()}


def _one_per_place(found, width):
    """The sites ``found`` with every place (a disc of one ``width``) kept once, the steepest crossing standing for it."""
    if not found:
        return tuple(np.zeros(0) for _ in range(5))
    x, y, la, lb, ang = (np.array(c) for c in zip(*found))
    tree, alive, kept = cKDTree(np.column_stack([x, y])), np.ones(x.size, bool), []
    for k in np.argsort(-ang, kind="stable"):
        if alive[k]:
            kept.append(k)
            alive[tree.query_ball_point((x[k], y[k]), width)] = False
    kept = np.array(kept)
    return x[kept], y[kept], la[kept], lb[kept], ang[kept]
