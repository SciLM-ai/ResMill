"""Fault patterns tied to the fold.

:func:`fold_faults` draws the faults of one folded trap for its trap style, following the structure research's rule
set (R-1 to R-14 of ``fold_fault_relations.md``):

* a strain field E = H(d~)/k_ref + R n(x)n: the curvature of the smoothed fold (the Hessian corrected for the limbs'
  dip), scaled by its largest value in the trap, plus a regional extension (R > 0) or shortening (R < 0) along n, as
  Withjack & Scheiner's models superpose doming and regional stress; R and n follow the style's setting, +n pointing
  down the regional slope (basinward) where the fold has one;
* fold-related faults where that strain is largest (density eps1+ + eps2+ plus a floor), inside the trap buffered by
  0.3 of its half-width, striking across the extension they take up, with a scatter that narrows as the strain gets
  more anisotropic, their tips clipped to the buffered trap so the throw dies at the fold's edge;
* regionally oriented faults over the whole model at the regional share R / (1 + R) of the trap's density, striking
  across n, mostly dipping basinward, concave toward their hanging wall, their throw growing downward; zero to three
  of them are major faults, 3-20 km long, that cross the field;
* the style's own sets: crestal grabens or step faults (throw at most 0.2-0.6 of the trap's relief), break thrusts on
  the steeper limb, tear faults (strike-slip faults as steep low-throw faults), a bounding fault through the spill
  point with minor faults in its hanging wall; inherited faults in some cases; relays where the grid resolves them;
  the block styles (:mod:`resmill.block_styles`) draw the faults inside the trap of their model with the population
  alone, a rollover's with the keystone graben of the Gulf anticline, 0.5 of the time;
* throw 0.03 L^0.92 sin(dip) 10^N(0, 0.27) on Norne's lengths, dips by kind, curvature, wander.

Each fault's tip ellipse is placed from the reservoir's depth at the fault (the datum ``top`` plus the fold there). The
density counts faults whose throw in the reservoir reaches 5 m, the threshold of the S6 densities, and only those are
drawn: a population fault keeps its own displacement-length scatter and is drawn longer until it reaches 5 m. Values tagged [J] are judgement values (the
research's, approved by the owner on 2026-09-30, or added in the visual checks and labelled so); the rest are sourced
there. Returns :class:`resmill.faults.Fault` objects (``kind`` labels the set) for ``to_grdecl``.
"""
import math

import numpy as np
from scipy import ndimage

from .faults import TIP_ASPECT, Fault, ww_profile
from .structure import _spill_levels

STYLES = ("four_way", "turtle", "faulted_anticline", "fold_belt", "fault_bounded", "low_relief", "tilted_blocks", "rollover")
MIN_THROW = 5.0                     # m in the reservoir: the faults S6's densities count
ORDER = ("inherited", "bounding", "major", "regional", "longitudinal", "oblique", "transverse", "minor", "graben",
         "step", "thrust", "tear")  # genetic order [J]


def _strike(angle):
    """Package strike (degrees clockwise from +x) of a trace at math ``angle`` (radians, counterclockwise)."""
    return float(-math.degrees(angle) % 360.0)


def _length(rng):
    """Norne's fault lengths (m): log-normal, median 800 m, clipped to 150-4,000 m."""
    return float(np.clip(10.0 ** rng.normal(2.9, 0.35), 150.0, 4000.0))


def _regional_length(rng):
    """Major regional faults cut across the fold and beyond (Conroe; Burgan's 3-6 km): log-uniform 3-20 km [J]."""
    return float(10.0 ** rng.uniform(math.log10(3000.0), math.log10(20000.0)))


def _throw(rng, length, dip, under=1.0):
    """Throw from displacement-length scaling, 0.03 L^0.92 sin(dip) 10^N(0, 0.27) (Lathrop; Norne's scatter)."""
    return float(0.03 * length ** 0.92 * math.sin(math.radians(dip)) * 10.0 ** rng.normal(0.0, 0.27) * under)


def _in_reservoir(zc, top, thickness, half):
    """Share of a fault's central throw left in a reservoir from ``top`` down ``thickness`` when its tip ellipse,
    ``half`` high, is centred at depth ``zc``."""
    gap = max(abs(zc - (top + 0.5 * thickness)) - 0.5 * thickness, 0.0)
    return float(ww_profile(gap / half))


def _frame(fold, x_len, y_len, dx):
    """The fold on a grid of ``dx`` cells: depth, trap mask, crest (the trap's top), relief, long-axis angle,
    half-length and half-width. The trap is the local high with the largest closure, so a regional tilt that makes a
    corner the map's shallowest point does not hide it."""
    nx, ny = max(int(round(x_len / dx)), 3), max(int(round(y_len / dx)), 3)
    xs, ys = (np.arange(nx) + 0.5) * x_len / nx, (np.arange(ny) + 0.5) * y_len / ny
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    depth = np.asarray(fold(X, Y), dtype=float) * np.ones_like(X)
    if not np.isfinite(depth).all():
        raise ValueError("the fold has non-finite depths")
    spill = _spill_levels(depth)
    highs = (depth == ndimage.minimum_filter(depth, size=5)) & (spill > depth)
    m = np.zeros(depth.shape, dtype=bool)
    for i, j in zip(*np.nonzero(highs)):
        if m[i, j]:
            continue                                         # a high inside the best trap shares its trap
        labels, _ = ndimage.label(depth < spill[i, j])
        trap = labels == labels[i, j]
        if trap.sum() > m.sum():
            m = trap
    if m.sum() < 4:
        return None
    crest = np.unravel_index(int(np.argmin(np.where(m, depth, np.inf))), depth.shape)
    mx, my = X[m].mean(), Y[m].mean()
    vals, vecs = np.linalg.eigh(np.cov(np.vstack([X[m] - mx, Y[m] - my])))
    return dict(X=X, Y=Y, depth=depth, mask=m, crest=crest, dx=x_len / nx, dy=y_len / ny,
                relief=float(depth[m].max() - depth[crest]), axis=float(math.atan2(vecs[1, 1], vecs[0, 1])),
                a=2.0 * math.sqrt(vals[1]), b=2.0 * math.sqrt(max(vals[0], 1e-12)))


def _strain(fr, R, n_angle, smooth):
    """Principal values and the first direction (radians) of E = H(d~)/k_ref + R n(x)n, the Hessian divided by
    (1 + |grad d|^2)^1.5 so that it is the curvature on steeper limbs too."""
    ds = ndimage.gaussian_filter(fr["depth"], smooth * fr["b"] / fr["dx"], mode="nearest")
    gx, gy = np.gradient(ds, fr["dx"], fr["dy"])
    hxx, hxy = np.gradient(gx, fr["dx"], fr["dy"])
    hyy = np.gradient(gy, fr["dx"], fr["dy"])[1]
    scale = (1.0 + gx ** 2 + gy ** 2) ** 1.5
    hxx, hxy, hyy = hxx / scale, hxy / scale, hyy / scale
    k1 = 0.5 * (hxx + hyy) + np.sqrt((0.5 * (hxx - hyy)) ** 2 + hxy ** 2)
    kref = max(float(k1[fr["mask"]].max()), 1e-12)
    nx_, ny_ = math.cos(n_angle), math.sin(n_angle)
    exx, eyy, exy = hxx / kref + R * nx_ * nx_, hyy / kref + R * ny_ * ny_, hxy / kref + R * nx_ * ny_
    mean, rad = 0.5 * (exx + eyy), np.sqrt((0.5 * (exx - eyy)) ** 2 + exy ** 2)
    return mean + rad, mean - rad, 0.5 * np.arctan2(2.0 * exy, exx - eyy)


def _regional_dip(fr):
    """The regional slope: unit vector down it and its gradient (m/m), from a plane fitted around the trap."""
    out = ~ndimage.binary_dilation(fr["mask"], iterations=3)
    if out.sum() < 10:
        out = np.ones_like(fr["mask"])
    A = np.c_[fr["X"][out], fr["Y"][out], np.ones(int(out.sum()))]
    (gx, gy, _), *_ = np.linalg.lstsq(A, fr["depth"][out], rcond=None)
    g = math.hypot(gx, gy)
    return (gx / g, gy / g, g) if g > 0.0 else (1.0, 0.0, 0.0)


def _inside(fr, region, x, y):
    """Whether points (x, y) lie inside the model and in ``region`` (a grid mask)."""
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    i, j = np.floor(x / fr["dx"]).astype(int), np.floor(y / fr["dy"]).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < region.shape[0]) & (j < region.shape[1])
    out = np.zeros(x.shape, dtype=bool)
    out[ok] = region[i[ok], j[ok]]
    return out


def _clip(fr, allowed, cx, cy, angle, length):
    """The part of a straight trace through (cx, cy) at ``angle`` that stays inside ``allowed`` around its centre:
    its new centre and length, or None when less than two cells remain."""
    s = np.arange(-0.5 * length, 0.5 * length + 1e-9, 0.5 * fr["dx"])
    inside = _inside(fr, allowed, cx + s * math.cos(angle), cy + s * math.sin(angle))
    k0 = int(np.argmin(np.abs(s)))
    if not inside[k0]:
        return None
    lo, hi = k0, k0
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    while hi < len(s) - 1 and inside[hi + 1]:
        hi += 1
    new = s[hi] - s[lo]
    if new < 2.0 * fr["dx"]:
        return None
    mid = 0.5 * (s[hi] + s[lo])
    return cx + mid * math.cos(angle), cy + mid * math.sin(angle), float(new)


def _reach(fr, start, angle):
    """Distance (m) from ``start`` along ``angle`` to the trap's edge."""
    s = np.arange(0.0, 4.0 * fr["a"] + fr["dx"], 0.5 * fr["dx"])
    inside = _inside(fr, fr["mask"], start[0] + s * math.cos(angle), start[1] + s * math.sin(angle))
    return float(s[np.argmin(inside)] if not inside.all() else s[-1])


def _kind(angle, axis):
    """The set a fold-related fault belongs to, by its trace's angle to the fold axis."""
    d = abs((math.degrees(angle - axis) + 90.0) % 180.0 - 90.0)
    return "longitudinal" if d < 30.0 else "transverse" if d > 60.0 else "oblique"


def fold_faults(style, fold, x_len, y_len, dx, density, top, thickness, seed, regional=None, over_salt=False,
                max_faults=300, basinward=0.8):
    """The faults of one folded trap (a list of :class:`resmill.faults.Fault`, in genetic order).

    ``style`` is one of :data:`STYLES`; ``fold`` the trap's structure without roughness (depth shift, m, positive
    down; a rough surface has many local highs and is slow to search); ``x_len``, ``y_len`` the model and ``dx`` its
    cell size (m); ``density`` the faults per km2 of the trap (the closure, as S6 counts them) whose throw in the
    reservoir reaches 5 m, drawn by the caller from the style's S6 range (regionally oriented faults keep their share
    of it over the whole model); ``top`` the reservoir top's datum depth, to which ``fold`` adds its shift (as a
    layer's ``top_depth``), and ``thickness`` its thickness (m); ``regional`` an optional ``(R, n_azimuth)`` replacing
    the style's draw (n's azimuth in degrees clockwise from +x; for a fault-bounded trap it then also replaces the
    bounding fault's direction for the regional set); ``over_salt`` marks a dome over deep salt or a salt-cored fold.
    A density of 0 still draws the style's own sets; ``max_faults`` caps the total, dropping the regional faults
    farthest from the trap first. ``basinward`` is the share of the regionally oriented faults whose hanging wall lies on
    the +n side, basinward (0.8 [J]; a rollover's crestal faults are 40-70 % antithetic, Evamy et al. 1978, Okari).
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")
    rng = np.random.default_rng(seed)
    fr = _frame(fold, x_len, y_len, dx)
    if fr is None:
        return []
    axis, a, b, cell = fr["axis"], fr["a"], fr["b"], fr["dx"]
    away = ndimage.distance_transform_edt(~fr["mask"], sampling=(fr["dx"], fr["dy"]))   # distance from the trap (m)
    allowed = away <= 0.3 * b                                                                     # [J] (E6)
    area = float(fr["mask"].sum()) * fr["dx"] * fr["dy"] / 1e6           # the closure, as S6 counts its faults
    crest = (float(fr["X"][fr["crest"]]), float(fr["Y"][fr["crest"]]))
    ux, uy, wx, wy = math.cos(axis), math.sin(axis), -math.sin(axis), math.cos(axis)

    def top_at(x, y):
        return top + float(np.asarray(fold(np.array([x]), np.array([y])), dtype=float).ravel()[0])

    def below():                                                         # growth faults, thrusts: throw grows downward
        u = rng.uniform(0.25, 0.5)                                                                # [J] (E9, E10)
        return lambda half, t_loc: t_loc + thickness + u * half

    if regional is not None:
        R, n_angle = float(regional[0]), -math.radians(float(regional[1]))
    elif style in ("four_way", "turtle"):
        R, n_angle = 0.0, axis
    elif style in ("faulted_anticline", "low_relief", "tilted_blocks", "rollover"):   # Gulf type: n at any angle to the axis
        R, n_angle = rng.uniform(0.5, 2.0), axis + rng.choice((-1.0, 1.0)) * math.radians(rng.uniform(0.0, 90.0))
        down_x, down_y, slope = _regional_dip(fr)
        if (math.cos(n_angle) * down_x + math.sin(n_angle) * down_y < 0.0) if slope > 1e-3 else rng.uniform() < 0.5:
            n_angle += math.pi                                           # +n basinward, down the regional slope
    elif style == "fold_belt":                                           # shortening across the fold axis (T3)
        R, n_angle = -rng.uniform(0.25, 0.6), axis + 0.5 * math.pi
    else:                                                                # fault-bounded: set by the bounding fault
        R, n_angle = rng.uniform(0.5, 2.0), None
    smooth = rng.uniform(0.1, 0.25)                                                               # [J] (A1)
    under_range = (math.log10(0.1), math.log10(0.6)) if style == "low_relief" else (0.0, 0.0)     # [J] (Burgan)

    faults = []

    def add(cx, cy, angle, length, kind, dip=None, hw=None, r_over_l=None, reverse=False, throw=None, dl=None,
            cap=None, z=None, clip=True, relay=True, factor=None, min_res=0.0):
        """Append one fault, or a relay pair; return its throw in the reservoir, None when it could not be placed,
        or "small" when that throw falls short of ``min_res`` (then nothing is appended)."""
        if len(faults) >= max_faults:
            return None
        if clip:
            got = _clip(fr, allowed, cx, cy, angle, length)
            if got is None:
                return None
            cx, cy, length = got
        dip = rng.uniform(45.0, 70.0) if dip is None else dip                                    # E6, E22
        hw = int(rng.choice((-1, 1))) if hw is None else hw
        if r_over_l is None:
            r_over_l = rng.choice((-1.0, 1.0)) * rng.uniform(10.0, 30.0)                         # minor, >= 10 L [J]
        sin_d = math.sin(math.radians(dip))
        if throw is None:
            if dl is not None:
                throw = dl * length
            elif factor is not None:                                     # the fault's own displacement-length scatter
                throw = 0.03 * length ** 0.92 * sin_d * factor
            else:
                throw = _throw(rng, length, dip, 10.0 ** rng.uniform(*under_range))
        throw = min(throw, cap) if cap is not None else throw
        if not throw > 0.0:
            return None
        t_loc = top_at(cx, cy)
        half = 0.5 * length / TIP_ASPECT * sin_d
        zc = t_loc + 0.5 * thickness + rng.normal(0.0, 0.6 * half) if z is None else z(half, t_loc)   # A16
        res = throw * _in_reservoir(zc, t_loc, thickness, half)
        if res < min_res:
            return "small"
        common = dict(dip=dip, hanging_wall=hw, hw_share=rng.uniform(0.5, 0.8),                  # [J] (A15)
                      drag=(rng.uniform(0.3, 0.5), rng.uniform(0.15, 0.3)), reverse=reverse, z_center=zc,
                      bends=rng.uniform(0.015, 0.03), kind=kind)                                 # [J]
        q, ratio = rng.uniform(0.15, 0.3), rng.uniform(3.0, 5.0)        # overlap 15-30 % of a segment, ~4 x sep
        seg = length / (2.0 - q)                                         # two segments spanning the trace exactly
        sep, seg_throw = q * seg / ratio, throw * (seg / length) ** 0.92
        t, n = np.array([math.cos(angle), math.sin(angle)]), np.array([-math.sin(angle), math.cos(angle)])
        centres = [np.array([cx, cy]) + sgn * (0.5 * length - 0.5 * seg) * t + sgn * 0.5 * sep * n for sgn in (-1, 1)]
        seg_res = min(seg_throw * _in_reservoir(zc, top_at(*c), thickness, 0.5 * seg / TIP_ASPECT * sin_d)
                      for c in centres)                                  # each segment at its own centre
        tips = [c + sgn * 0.5 * seg * t for c in centres for sgn in (-1, 1)]
        if (relay and length > 600.0 and sep >= 2.0 * cell and rng.uniform() < 0.25 and seg_res >= MIN_THROW  # R-11
                and len(faults) + 2 <= max_faults
                and (not clip or _inside(fr, allowed, *np.array(tips).T).all())):
            for c in centres:
                faults.append(Fault(center=(float(c[0]), float(c[1])), strike=_strike(angle), length=seg,
                                    throw=seg_throw, radius=math.copysign(abs(r_over_l) * seg, r_over_l),
                                    seed=int(rng.integers(2 ** 31)), **common))
            return seg_res
        faults.append(Fault(center=(float(cx), float(cy)), strike=_strike(angle), length=length, throw=throw,
                            radius=math.copysign(abs(r_over_l) * length, r_over_l), seed=int(rng.integers(2 ** 31)),
                            **common))
        return res

    counted = 0

    def template(*args, **kw):
        nonlocal counted
        res = add(*args, **kw)
        counted += res is not None and res >= MIN_THROW
        return res

    def drawn(x, y, angle, kind, length_law, z=None, **kw):
        """A population fault of at least 5 m in the reservoir: everything about it drawn once (its displacement-length
        scatter, a low-relief under-displacement, dip, sides, curvature, vertical position), then only its length drawn
        again until that throw is reached (S6 counts these)."""
        factor = 10.0 ** rng.normal(0.0, 0.27) * 10.0 ** rng.uniform(*under_range)
        kw.setdefault("dip", rng.uniform(45.0, 70.0))
        kw.setdefault("hw", int(rng.choice((-1, 1))))
        kw.setdefault("r_over_l", rng.choice((-1.0, 1.0)) * rng.uniform(10.0, 30.0))
        if z is None:
            xi = rng.normal()                                                                     # A16
            z = lambda half, t_loc: t_loc + 0.5 * thickness + xi * 0.6 * half
        for _ in range(20):
            length = length_law()
            res = add(*(x(length), y(length)) if callable(x) else (x, y), angle, length, kind, factor=factor,
                      min_res=MIN_THROW, z=z, **kw)
            if res != "small":
                return res
        return None

    # The style's own sets.
    if style == "fault_bounded":                         # a bounding fault through the spill point (the saddle)
        ring = ndimage.binary_dilation(fr["mask"]) & ~fr["mask"]
        d = np.where(ring, fr["depth"], np.inf)
        ties = np.flatnonzero((d <= d.min() + 1e-6 * max(1.0, abs(d.min()))).ravel())  # exact ties: a flat regional
        spill = np.unravel_index(int(rng.choice(ties)), d.shape)
        cx, cy = float(fr["X"][spill]), float(fr["Y"][spill])
        out = math.atan2(cy - crest[1], cx - crest[0])                  # up-dip: from the crest out through the spill
        angle = out + 0.5 * math.pi
        if regional is None:
            n_angle = out + math.pi                                      # extension across it, basinward = down-dip
        hw = -1 if rng.uniform() < 0.6 else 1      # trap in the footwall 60 % [J] (E17, E20); +1: trap in hanging wall
        along = (fr["X"][fr["mask"]] - cx) * math.cos(angle) + (fr["Y"][fr["mask"]] - cy) * math.sin(angle)
        template(cx, cy, angle, rng.uniform(0.8, 1.5) * float(np.ptp(along) + cell), "bounding", hw=hw,  # [J]
                 r_over_l=hw * rng.uniform(2.0, 10.0), clip=False, relay=False, z=below())
        side = hw * ((fr["X"] - cx) * math.cos(out + math.pi) + (fr["Y"] - cy) * math.sin(out + math.pi))
        across = np.abs((fr["X"] - cx) * math.cos(angle) + (fr["Y"] - cy) * math.sin(angle))
        band = np.flatnonzero(((side > 0.5 * cell) & (side < max(0.5 * b, 3 * cell))
                               & (across < 0.5 * float(np.ptp(along)))).ravel())
        hw_dir = np.array([math.cos(out + math.pi), math.sin(out + math.pi)]) * hw
        for _ in range(int(rng.integers(1, 5)) if band.size else 0):    # minor faults in its hanging wall [J]
            i, j = np.unravel_index(int(rng.choice(band)), allowed.shape)
            synthetic = rng.uniform() < 0.7                                                       # E19, E20 [J]
            m_angle, length = angle + math.radians(rng.normal(0.0, 10.0)), _length(rng)
            mc = np.array([fr["X"][i, j], fr["Y"][i, j]])
            mt = np.array([math.cos(m_angle), math.sin(m_angle)])
            for _ in range(4):                                           # both tips stay in that hanging wall
                if min(np.dot(mc + s * 0.5 * length * mt - np.array([cx, cy]), hw_dir) for s in (-1, 1)) > 0.0:
                    template(mc[0], mc[1], m_angle, length, "minor", hw=hw if synthetic else -hw, clip=False)
                    break
                length *= 0.5
    graben = {"four_way": (0.6, 0.25) if over_salt else (0.2, 0.1), "turtle": (0.8, 0.0),       # E6, E7, E28 [J]
              "faulted_anticline": (0.6, 0.0), "fold_belt": (0.5, 0.0),
              "rollover": (0.5, 0.0)}.get(style, (0.0, 0.0))                                      # T23 [J]
    u = rng.uniform()
    transverse_graben = style == "fold_belt" and over_salt and 0.5 <= u < 0.7                     # salt-cored, E13
    share = rng.uniform(0.2, 0.6)                                       # R-6: graben throw / relief 0.2-0.6 [J]
    if fr["relief"] <= 0.0:
        u = 1.0                                                          # a flat top: no crestal sets
    if u < graben[0] or transverse_graben:                                                        # R-7
        g_axis = axis + 0.5 * math.pi if transverse_graben else axis
        width = math.exp(rng.uniform(math.log(0.05), math.log(0.4))) * 2.0 * (a if transverse_graben else b)
        length = rng.uniform(0.4, 0.95) * 2.0 * (b if transverse_graben else a)
        first = None
        for side in (-1.0, 1.0):
            ox, oy = -math.sin(g_axis), math.cos(g_axis)
            hw = -1 if side > 0 else 1                                   # hanging walls toward the graben's middle
            if template(crest[0] + side * 0.5 * width * ox, crest[1] + side * 0.5 * width * oy, g_axis, length,
                        "graben", hw=hw, relay=False, throw=share * fr["relief"] if first is None
                        else first * rng.uniform(0.4, 1.0)) is not None and first is None:
                first = faults[-1].throw
    elif u < graben[0] + graben[1]:                                       # step faults downthrown one way (E6)
        hw, steps = int(rng.choice((-1, 1))), int(rng.integers(2, 4))
        for k in range(steps):
            off = (k - 1) * rng.uniform(0.1, 0.25) * b                                            # [J]
            template(crest[0] + off * wx, crest[1] + off * wy, axis, rng.uniform(0.4, 0.95) * 2.0 * a, "step", hw=hw,
                     throw=share * fr["relief"] / steps, relay=False)
    if style == "fold_belt":
        steep = _steeper_limb(fr, crest, wx, wy)
        for side in ((steep,) if _limb_ratio(fr, crest, wx, wy) >= 1.3 else (-1.0, 1.0)):       # E10, E25
            out = math.atan2(side * wy, side * wx)
            off = rng.uniform(0.4, 0.8) * _reach(fr, crest, out)                                  # within the limb [J]
            template(crest[0] + off * math.cos(out), crest[1] + off * math.sin(out), axis,
                     rng.uniform(0.5, 0.9) * 2.0 * a, "thrust", dip=rng.uniform(45.0, 70.0), hw=-1 if side > 0 else 1,
                     r_over_l=(-1 if side > 0 else 1) * rng.uniform(2.0, 10.0), reverse=True, relay=False, z=below())
        for _ in range(int(rng.integers(0, 3))):                          # tear faults where the plunge changes
            along = rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 0.9) * a                           # [J]
            template(crest[0] + along * ux, crest[1] + along * uy, axis + 0.5 * math.pi,
                     rng.uniform(0.6, 1.2) * 2.0 * b, "tear", dip=rng.uniform(80.0, 90.0),
                     dl=rng.uniform(0.002, 0.01), relay=False)                                    # [J]

    # The populations (R-4, R-13): regional faults at their share of the density over the whole model, and the rest of
    # the trap's count beyond its own sets and the regional faults expected in it.
    eps1, eps2, theta1 = _strain(fr, R, axis if n_angle is None else n_angle, smooth)
    p = R / (1.0 + R) if R > 0.0 else 0.0                                                         # [J] (E2)
    n_regional = int(round(density * p * x_len * y_len / 1e6))
    rest = max(int(round(density * area)) - counted - int(round(density * p * area)), 0)
    n_inherited = int(round(rng.uniform(0.1, 0.3) * rest)) if rng.uniform() < 0.4 else 0         # R-12 [J]
    inherited = [axis + math.radians(rng.uniform(0.0, 180.0)) for _ in range(int(rng.integers(1, 3)))]
    n_fold = rest - n_inherited
    n_major = min(int(rng.integers(0, 4)), n_regional)                   # a few long faults cross the field [J]
    reg_sigma = math.radians(rng.uniform(10.0, 16.0))
    cells = np.flatnonzero(allowed.ravel())
    _length_law, _regional_length_law = (lambda: _length(rng)), (lambda: _regional_length(rng))
    lam = np.clip(eps1, 0.0, None) + np.clip(eps2, 0.0, None)
    lam = np.where(allowed, lam + rng.uniform(0.2, 0.5) * (lam[allowed].max() or 1.0), 0.0).ravel()  # [J] (A4)
    cdf = np.cumsum(lam) / lam.sum()

    def regional_fault(major, at=None):
        angle = n_angle + 0.5 * math.pi + rng.normal(0.0, reg_sigma)
        basin = math.cos(n_angle - (angle + 0.5 * math.pi))             # the hanging-wall side against +n
        hw = int(math.copysign(1.0, basin)) * (1 if rng.uniform() < basinward else -1)            # basinward 0.8 [J]
        if major:                                                        # through a point of the buffered trap
            i, j = np.unravel_index(int(rng.choice(cells)), allowed.shape)
            frac = rng.uniform(-0.4, 0.4)
            x = lambda length: fr["X"][i, j] + frac * length * math.cos(angle)
            y = lambda length: fr["Y"][i, j] + frac * length * math.sin(angle)
        else:
            x, y = at
        long = major or style == "low_relief"                           # Burgan's regional faults are 3-6 km long
        return drawn(x, y, angle, "major" if major else "regional", _regional_length_law if long else _length_law,
                     hw=hw, r_over_l=hw * rng.uniform(2.0, 10.0), clip=False, z=below())

    def inherited_fault():
        i, j = np.unravel_index(int(rng.choice(cells)), allowed.shape)
        angle = inherited[int(rng.integers(len(inherited)))] + math.radians(rng.normal(0.0, 5.0))
        return drawn(fr["X"][i, j], fr["Y"][i, j], angle, "inherited", _length_law, clip=False)

    def fold_fault():
        k = min(int(np.searchsorted(cdf, rng.uniform())), lam.size - 1)
        i, j = np.unravel_index(k, allowed.shape)
        e1, e2 = max(eps1[i, j], 0.0), max(eps2[i, j], 0.0)
        across = rng.uniform() < (e2 / (e1 + e2) if e1 + e2 > 0.0 else 0.0)
        anis = (eps1[i, j] - eps2[i, j]) / (abs(eps1[i, j]) + abs(eps2[i, j]) + 1e-12)
        sigma = math.radians(45.0 - 35.0 * min(anis / 0.5, 1.0))
        angle = theta1[i, j] + (0.5 * math.pi if across else 0.0) + 0.5 * math.pi + rng.normal(0.0, sigma)
        return drawn(fr["X"][i, j] + rng.uniform(-0.5, 0.5) * fr["dx"],
                     fr["Y"][i, j] + rng.uniform(-0.5, 0.5) * fr["dy"], angle, _kind(angle, axis), _length_law)

    cap = max_faults
    for target, draw in ((n_major, lambda: regional_fault(True)), (n_inherited, inherited_fault), (n_fold, fold_fault),
                         (n_regional - n_major, lambda: regional_fault(False, (rng.uniform(0.0, x_len),
                                                                               rng.uniform(0.0, y_len))))):
        got = tries = 0
        while got < target and tries < 3 * target + 5 and len(faults) < max_faults:
            tries += 1
            got += draw() is not None
        max_faults = len(faults) + 10 ** 9 if draw is fold_fault else max_faults   # regional faults: trimmed below
    if len(faults) > cap:                                                # the cap drops the regional faults far away
        def reach(f):
            c = np.array(f.center)
            return float(away[min(max(int(c[0] // fr["dx"]), 0), away.shape[0] - 1),
                              min(max(int(c[1] // fr["dy"]), 0), away.shape[1] - 1)]) - 0.5 * f.length
        far = sorted((f for f in faults if f.kind == "regional"), key=reach, reverse=True)[:len(faults) - cap]
        faults = [f for f in faults if all(f is not g for g in far)]
    faults.sort(key=lambda f: ORDER.index(f.kind))
    for n, f in enumerate(faults):
        f.name = f"F{n + 1:03d}"
    return faults


def _limb_ratio(fr, crest, wx, wy):
    """Steeper over gentler mean dip of the trap's two limbs (across the axis)."""
    s = _limb_dips(fr, crest, wx, wy)
    return max(s) / max(min(s), 1e-12)


def _steeper_limb(fr, crest, wx, wy):
    """+1 when the limb on the +across side is the steeper, else -1."""
    s = _limb_dips(fr, crest, wx, wy)
    return 1.0 if s[1] >= s[0] else -1.0


def _limb_dips(fr, crest, wx, wy):
    gx, gy = np.gradient(fr["depth"], fr["dx"], fr["dy"])
    side = (fr["X"] - crest[0]) * wx + (fr["Y"] - crest[1]) * wy
    slope = np.hypot(gx, gy)
    m = fr["mask"]
    return [float(slope[m & (side < 0)].mean()) if (m & (side < 0)).any() else 0.0,
            float(slope[m & (side > 0)].mean()) if (m & (side > 0)).any() else 0.0]
