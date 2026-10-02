import warnings

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq

from .base import Layer
from .channel import FACIES_PROPS, _correlated_noise
from ._fluvial import _gauss_clip

# ``facies_props`` entries a lobe uses: -1 mud rock, 2 the faded fringe, 3 the sand
# (its rock is poro_ave, perm_ave, poro_std and perm_std).
_FACIES_KEYS = {-1: {"poro", "log10_perm", "poro_sd", "log10_perm_sd", "kvkh"},
                2: {"ntg_floor", "ntg_crest", "kvkh"}, 3: {"kvkh"}}
_SAND_SHARE_MIN = 0.5    # a faded cell with at least this sand fraction is labelled sand (3)
_MUD_SHARE_MAX = 0.1     # below this sand fraction it is mud (-1), in between thin-bedded fringe (2)


def _compensation_weights(surface, dz, dh_ave, scale):
    """Probability that each column is the next stamp's centre, ``(ny, nx)``.

    ``exp(-deficit / (scale * dh_ave))``, the deficit being the column's height above the
    lowest column in metres (``surface`` is in cells of ``dz``) and ``dh_ave`` the stamps'
    peak thickness: a column one ``scale`` of stamp thicknesses higher is e times less
    likely. A vanishing scale always picks the lowest column, a large one is random.
    """
    weight = np.exp(-(surface - surface.min()) * dz / (scale * dh_ave))
    return weight / weight.sum()


def _faded_sand_fraction(structure, ntg, floor, crest=1.0):
    """Sub-cell sand fraction of every cell, an exponential of the cell's rank in ``structure``.

    ``floor + (crest - floor) * exp(-(1 - u) / lam)`` with ``u`` the rank of the cell's
    ``structure`` value (0 the poorest, 1 the richest, uniform): ``crest`` in the best
    cells (the lobe axis, the base of a bed), falling to ``floor`` in the poorest (the
    fringe, the top of a bed). ``lam``, the e-folding in ranks, is found so that the mean
    over the cells is ``ntg``, which must lie between ``floor`` and ``crest``.
    """
    if not floor < ntg < crest:
        raise ValueError(f"the mean sand fraction ntg = {ntg} must lie between ntg_floor = {floor} "
                         f"and ntg_crest = {crest}")
    # mean over uniform u: floor + (crest - floor) * lam * (1 - exp(-1 / lam))
    lam = brentq(lambda lam: floor + (crest - floor) * lam * (1.0 - np.exp(-1.0 / lam)) - ntg, 1e-6, 1e6)
    flat = np.asarray(structure).ravel()
    rank = np.empty(flat.size)
    rank[np.argsort(flat, kind="stable")] = (np.arange(flat.size) + 0.5) / flat.size
    return (floor + (crest - floor) * np.exp(-(1.0 - rank) / lam)).reshape(np.shape(structure))


def _stamp_mud(allsurface, nz, dz, mud_cap, erosion, floor):
    """Metres of mud in every cell of a stack of stamps, ``(nz, ny, nx)``, and the share of contacts amalgamated.

    ``allsurface`` holds the surface after every stamp (cells of ``dz``, the first the empty floor), so a stamp's
    deposit at a column is the rise of the surface there. Each deposit of thickness ``t`` is capped with mud on its
    top, ``min((1 - floor) t, mud_cap)`` thick: ``floor`` is the sand fraction of a thin deposit (the lobe fringe), so
    the sand fraction of a lobe falls from ``1 - mud_cap / t`` in its thick part to ``floor`` at its margin. The next
    stamp to cover the column cuts ``erosion x`` its own thickness off the top of that cap (the mud it cuts through
    is sand of the younger stamp, sand on sand: amalgamation); caps nothing covers again stay. The share returned is
    the columns where the cap was cut through of those where a younger stamp met one. Mud above the top of the layer
    is not counted.
    """
    shape = allsurface[0].shape
    mud = np.zeros((nz,) + shape)
    cap_len, cap_top = np.zeros(shape), np.zeros(shape)
    met = cut = 0

    def lay(rows, cols, bottom, length):
        """Add the mud of intervals [bottom, bottom + length] (m) of the columns (rows, cols) to the cells they cross."""
        first = np.floor(bottom / dz).astype(int)
        for step in range(int(np.ceil(length.max() / dz)) + 1 if length.size else 0):
            k = first + step
            overlap = np.clip(np.minimum(bottom + length, (k + 1) * dz) - np.maximum(bottom, k * dz), 0.0, None)
            ok = (k >= 0) & (k < nz) & (overlap > 0.0)
            mud[k[ok], rows[ok], cols[ok]] += overlap[ok]

    for before, after in zip(allsurface[:-1], allsurface[1:]):
        thickness = (after - before) * dz
        rows, cols = np.nonzero(thickness > 0.0)
        if rows.size == 0:
            continue
        t, held = thickness[rows, cols], cap_len[rows, cols]
        capped = held > 0.0
        left = np.maximum(held - erosion * t, 0.0)
        met += int(capped.sum())
        cut += int((capped & (left == 0.0)).sum())
        lay(rows[capped], cols[capped], (cap_top[rows, cols] - held)[capped], left[capped])
        cap_len[rows, cols] = np.minimum((1.0 - floor) * t, mud_cap)
        cap_top[rows, cols] = after[rows, cols] * dz
    rows, cols = np.nonzero(cap_len > 0.0)
    lay(rows, cols, cap_top[rows, cols] - cap_len[rows, cols], cap_len[rows, cols])
    return mud, (cut / met if met else 0.0)


def _effective_perm(sand, k_sand, k_mud):
    """Permeability of a cell of sand fraction ``sand`` whose sand and mud are mixed at random: the 2-D effective medium
    (Bruggeman), ``(b + sqrt(b^2 + 4 k_sand k_mud)) / 2`` with ``b = (2 sand - 1)(k_sand - k_mud)``. The sand percolates
    from half of the cell: below it the cell is tight (``sqrt(k_sand k_mud)`` at exactly half), above it
    ``(2 sand - 1) k_sand`` and more, so a cell is net (above 1 mD) where its sand is the majority. The arithmetic mean, right for
    layers that run through the whole cell, would leave every cell with a sixth of sand above 80 mD."""
    b = (2.0 * sand - 1.0) * (k_sand - k_mud)
    return 0.5 * (b + np.sqrt(b * b + 4.0 * k_sand * k_mud))


class LobeLayer(Layer):
    """Layer with turbidite lobe deposition geology."""

    def create_geology(self, poro_ave, perm_ave, poro_std, perm_std, ntg,
                       dh_ave=4.0, dh_std=0.5, r_ave=450.0, r_std=20.0,
                       asp=1.5, azimuth=0.0, azimuth_std=10.0,
                       m=100, upthinning=True, bouma_factor=0, compensation_scale=None,
                       facies_props=None, interlobe_erosion=None):
        """Generate turbidite lobe geology.

        Parameters
        ----------
        poro_ave, perm_ave : float
            Mean porosity and log10(permeability in mD).
        poro_std, perm_std : float
            Standard deviation of porosity and log10(permeability).
        ntg : float
            Net-to-gross ratio (0 to 1).
        dh_ave, dh_std : float
            Per-event lobe thickness, **in meters**, drawn as
            ``N(dh_ave, dh_std)`` and clipped to a small positive
            minimum.
        r_ave, r_std : float
            Per-event lobe radius (semi-minor in physical metres), drawn
            as ``N(r_ave, r_std)`` and clipped to positive. The lobe
            is a true ellipse in physical (m) space — cell offsets are
            scaled by ``self.dx`` / ``self.dy`` independently before
            the rotation, so non-isotropic cells (``dx != dy``) and
            any azimuth are handled correctly.
        asp : float
            Lobe aspect ratio (semi-major / semi-minor, dimensionless).
        azimuth : float
            Mean lobe-elongation orientation (degrees), measured
            **clockwise from +x** to match the convention used by
            ``ChannelLayer`` / ``DeltaLayer``: ``azimuth=0``
            elongates lobes along +x, ``azimuth=90`` along -y. Range
            can be 0-360 or any equivalent — the engine wraps via
            cos/sin.
        azimuth_std : float
            Per-event Gaussian std (degrees) on the lobe orientation,
            applied independently to every stamp. Default 10°. A clean
            cone of well-aligned lobes uses ~5°; turbidite-realistic
            scatter is ~15-25°.
        m : float
            Compensation exponent: a column's weight as the next stamp's centre is
            ``(height above the lowest column in cells + 0.001) ** -m``, so every ``m``
            above about 2 puts the centre on the lowest column and changes nothing
            (see ``compensation_scale``).
        upthinning : bool
            Apply vertical thinning upward.
        bouma_factor : float
            Bouma sequence discretization factor (0 = none).
        compensation_scale : float or None
            ``None`` keeps ``m``. A number replaces its weight by
            ``exp(-deficit / (compensation_scale * dh_ave))``, the deficit being the
            column's height above the lowest one in metres: a column one scale of stamp
            thicknesses higher is e times less likely to be the next stamp's centre. The
            strength is then a continuous setting from nearly random stacking (a scale of
            a few thicknesses, compensation index kappa about 0.6) to a stamp always on
            the lowest column (0.05 or less; kappa 0.8-0.95 by the stamp size).
        facies_props : dict or None
            ``None``: sand is the top ``ntg`` share of the lobe structure with the
            ``poro``/``perm`` fields standardized over all cells (so the sand's average
            moves with ``ntg``) and the mud is zero. A dict, keyed by facies code, gives
            the calibrated rock: the sand keeps exactly ``poro_ave``, ``perm_ave`` and
            the spreads ``poro_std``, ``perm_std``; the mud (``-1``) has its own ``poro``,
            ``log10_perm`` and optional ``poro_sd`` (relative) and ``log10_perm_sd``
            (decades), correlated noise; ``kvkh`` entries (``-1`` mud, ``3`` sand) give the
            vertical to horizontal permeability ratio per cell (``self.kvkh_mat``, PERMZ =
            kvkh x PERMX; none given: ``kzkx``). ``{2: {"ntg_floor": f, "ntg_crest": c,
            "kvkh": k}}`` also fades the sand: each cell's sub-cell sand fraction is an
            exponential of the rank of its lobe structure, ``c`` (default 1) in the axis
            and the base of a bed and ``f`` in the fringe and the top of a bed, and ``ntg``
            is then the mean sand fraction; porosity and permeability are the arithmetic
            mix of the sand and the mud, kv/kh runs log-linearly from ``k`` at ``f`` to
            the sand's at 1, and the facies are 3 (sand fraction of 0.5 or more) and 2
            (thin-bedded fringe), with no mud lattice. ``self.sand_fraction`` holds it.
        interlobe_erosion : float or None
            Needs ``facies_props``. ``None``: the sand is as above. A number replaces the sand's
            definition by a stack of stamps whose deposits carry interlobe mud: each stamp
            (a lobe) is capped with mud ``min((1 - f) t, M)`` thick, ``t`` its thickness at the
            column, ``f`` the sand fraction of its margin (``facies_props[2]["ntg_floor"]``,
            0 if absent) and ``M`` the cap thickness; the sand fraction of a lobe then falls from
            ``1 - M / t`` where it is thick to ``f`` at its margin (axis 85-100 %, off-axis
            50-85 %, fringe 20-50 % in the Tanqua, Spychala et al. 2017) with the mud as a
            continuous sheet over it. The next stamp over a column cuts
            ``interlobe_erosion x`` its own thickness off that cap (its scour: sand on sand,
            amalgamation, where it is thick, the axis, and none where it is thin, the margin),
            so the mud wraps each lobe at a low net-to-gross and is gone from the axes at a
            high one. ``M`` is found so that the mean sand fraction of the layer is ``ntg``
            (net-to-gross by thickness, as outcrop and core give it, whatever the grid); a cell's porosity is
            the arithmetic mix of the sand and the mud, its permeability the 2-D effective
            medium of the two (``_effective_perm``: tight below half sand, so the cells under
            1 mD are the non-net share), kv/kh runs log-linearly from the mud's through the
            fringe's (``facies_props[2]["kvkh"]``, at half sand: the most heterolithic) to the
            sand's. It also keeps the stamps' porosity decay at 1 (``clip_decay``). Facies are
            3 (net), 2 (sand fraction 0.1 to 0.5) and -1; ``self.sand_fraction`` holds the
            fraction and ``self.interlobe`` the cap thickness ``mud_thickness_m``, the share of
            contacts ``amalgamated``, the mean ``sand_fraction`` (``ntg``) and ``net_cells``,
            the share of cells whose sand is the majority (those above 1 mD).
        """
        self.poro_ave = poro_ave
        self.perm_ave = perm_ave
        self.poro_std = poro_std
        self.perm_std = perm_std
        self.ntg = ntg

        if interlobe_erosion is not None and facies_props is None:
            raise ValueError("interlobe_erosion needs facies_props: the mud and the sand have rock of their own")
        allfacies, allporo, self.allsurface = self._lobemodeling(
            dh_ave=dh_ave, dh_std=dh_std, r_ave=r_ave, r_std=r_std,
            asp=asp, azimuth=azimuth, azimuth_std=azimuth_std, m=m,
            upthinning=upthinning, bouma_factor=bouma_factor,
            compensation_scale=compensation_scale, clip_decay=interlobe_erosion is not None,
        )
        # Swap axes from (nz, ny, nx) to (nx, ny, nz)
        lobe_poro = np.swapaxes(allporo[-1], 0, -1)

        nx, ny, nz = self.nx, self.ny, self.nz
        sand_filt = [1.5, 2.5, 1.5]
        facies_filt = [2.5, 5, 2.5]
        sand_nug = 0.05
        lambda_perturb = 0.1

        # Facies: lobe structure + Gaussian perturbation, threshold at NTG
        facies_perturb = gaussian_filter(
            np.random.normal(0, 1, (3 * nx, 3 * ny, 3 * nz)),
            facies_filt, mode='wrap',
        )
        facies_perturb = facies_perturb[nx:2*nx, ny:2*ny, nz:2*nz]
        structure = lobe_poro + lambda_perturb * facies_perturb
        self.active = (structure > np.percentile(structure.flatten(), (1 - ntg) * 100))

        # Porosity field: lobe radial-decay structure + small Gaussian
        # noise mapped to the requested ``poro_ave ± poro_std`` envelope.
        poro_field = np.random.normal(0, 1, (3 * nx, 3 * ny, 3 * nz))
        poro_nug = np.random.normal(0, 1, (3 * nx, 3 * ny, 3 * nz))
        poro_field = gaussian_filter(poro_field, sand_filt, mode='wrap') + sand_nug * poro_nug
        poro_field = poro_field[nx:2*nx, ny:2*ny, nz:2*nz]
        poro_field = lobe_poro + lambda_perturb * poro_field
        # Rescale poro field to N(poro_ave, poro_std) per-cell with the
        # lobe geometry as the "structure" — no bivariate copula
        # required, so we don't lose correlation through rank-quantile
        # filtering.
        flat = poro_field.flatten()
        flat = poro_ave + poro_std * (flat - flat.mean()) / max(flat.std(), 1e-9)
        flat = np.clip(flat, poro_ave - 5 * poro_std, poro_ave + 5 * poro_std)
        self.poro_mat = flat.reshape(nx, ny, nz)

        # Permeability: deterministic Kozeny-Carman-style linear map —
        # log10(perm) tracks poro with slope = perm_std / poro_std so
        # the per-cell log-perm distribution exactly matches the
        # requested perm_std. Tiny residual noise (~0.05 log10) gives
        # cells a slight decorrelated jitter for visual texture without
        # diluting the trend.
        slope = perm_std / max(poro_std, 1e-6)
        log_perm = perm_ave + slope * (self.poro_mat - poro_ave)
        jitter = np.random.normal(0, 0.05, log_perm.shape)
        log_perm = np.clip(log_perm + jitter, perm_ave - 5 * perm_std, perm_ave + 5 * perm_std)

        if facies_props is None:
            self.active = self.active.astype(np.int8)
            self.poro_mat = self.poro_mat * self.active
            self.perm_mat = (10.0 ** log_perm) * self.active
        elif interlobe_erosion is None:
            self._calibrated_rock(structure, jitter, facies_props)
        else:
            sand, self.interlobe = self._interlobe_sand(ntg, interlobe_erosion, float(facies_props.get(2, {}).get("ntg_floor", 0.0)))
            self._calibrated_rock(structure, jitter, facies_props, sand=sand)

        # ``lobe_id`` keeps the per-lobe stacking index (1..N) for users
        # who want to colour by lobe generation. ``facies`` is the
        # uniform Alluvsim-style code: -1 (FF, floodplain / shale) where
        # inactive, 3 (LA = lateral-accretion / bar) where the cell
        # belongs to a lobe — closest analogue to Alluvsim's bar
        # facies for a turbidite-lobe deposit.
        if allfacies and len(allfacies) > 0:
            self.lobe_id = np.swapaxes(allfacies[-1].copy(), 0, -1).astype(np.int16)
        else:
            self.lobe_id = np.zeros((nx, ny, nz), dtype=np.int16)
        if facies_props is None:
            self.facies = np.where(self.active == 1, 3, -1).astype(np.int8)

    def _calibrated_rock(self, structure, jitter, facies_props, sand=None):
        """Opt-in rock by facies (see ``create_geology``'s ``facies_props``): sets ``active``,
        ``facies``, ``poro_mat``, ``perm_mat``, ``sand_fraction`` and, with a ``kvkh``, ``kvkh_mat``.

        The sand fraction ``s`` is the binary sand mask, or with ``ntg_floor`` the faded
        fraction. The sand-bed rock keeps the lobe structure's porosity (best at the axis and
        the base of a bed) standardized over the sand, weighted by ``s``, to ``poro_ave`` and
        ``poro_std``, and its log permeability follows with the slope ``perm_std / poro_std``,
        a small jitter, and an average of exactly ``perm_ave``. A cell is the arithmetic
        mix ``s x sand + (1 - s) x mud`` of porosity and permeability. ``sand``, the sand fraction of
        every cell from the interlobe mud (``interlobe_erosion``), replaces both and mixes the
        permeability as the 2-D effective medium (``_effective_perm``) instead.
        """
        for code, entry in facies_props.items():
            if int(code) not in _FACIES_KEYS or set(entry) - _FACIES_KEYS[int(code)]:
                uses = "; ".join(f"{c}: {', '.join(sorted(k))}" for c, k in sorted(_FACIES_KEYS.items()))
                raise ValueError(f"facies_props[{code}] = {dict(entry)}: a lobe uses {uses}")
        props = {code: {**FACIES_PROPS.get(code, {}), **facies_props.get(code, {})} for code in _FACIES_KEYS}
        fringe = facies_props.get(2, {})
        if sand is not None:
            s = sand
        elif "ntg_floor" in fringe:
            s = _faded_sand_fraction(structure, self.ntg, float(fringe["ntg_floor"]),
                                     float(fringe.get("ntg_crest", 1.0)))
        else:
            s = self.active.astype(float)
        weight = s / s.sum()
        mean = (weight * self.poro_mat).sum()
        z = (self.poro_mat - mean) / max(np.sqrt((weight * (self.poro_mat - mean) ** 2).sum()), 1e-12)
        poro_sd, perm_sd = self.poro_std, self.perm_std
        poro_sand = np.clip(self.poro_ave + poro_sd * z, self.poro_ave - 5 * poro_sd, self.poro_ave + 5 * poro_sd)
        poro_sand = poro_sand + self.poro_ave - (weight * poro_sand).sum()      # the clip moves the mean a little
        log_sand = self.perm_ave + perm_sd / max(poro_sd, 1e-6) * (poro_sand - self.poro_ave) + jitter
        log_sand = np.clip(log_sand, self.perm_ave - 5 * perm_sd, self.perm_ave + 5 * perm_sd)
        log_sand = log_sand + self.perm_ave - (weight * log_sand).sum()
        mud, shape = props[-1], s.shape
        poro_mud = np.full(shape, float(mud["poro"]))
        log_mud = np.full(shape, float(mud["log10_perm"]))
        if mud.get("poro_sd"):
            poro_mud = poro_mud * np.exp(float(mud["poro_sd"]) * _correlated_noise(shape, 3.0))
        if mud.get("log10_perm_sd"):
            log_mud = log_mud + float(mud["log10_perm_sd"]) * _correlated_noise(shape, 3.0)
        self.poro_mat = s * poro_sand + (1.0 - s) * poro_mud
        self.perm_mat = (s * 10.0 ** log_sand + (1.0 - s) * 10.0 ** log_mud if sand is None
                         else _effective_perm(s, 10.0 ** log_sand, 10.0 ** log_mud))
        self.sand_fraction = s.astype(np.float32)
        self.active = (s >= _SAND_SHARE_MIN).astype(np.int8)
        self.facies = np.where(s >= _SAND_SHARE_MIN, 3, np.where(s >= _MUD_SHARE_MAX, 2, -1)).astype(np.int8)
        if any("kvkh" in entry for entry in props.values()):
            knots = {0.0: props[-1].get("kvkh", self.kzkx), 1.0: props[3].get("kvkh", self.kzkx)}
            if sand is not None:
                if "kvkh" in fringe:       # the most heterolithic cell is the half-sand one
                    knots[0.5] = fringe["kvkh"]
            elif "ntg_floor" in fringe:
                knots[float(fringe["ntg_floor"])] = fringe.get("kvkh", self.kzkx)
            at = sorted(knots)
            self.kvkh_mat = np.exp(np.interp(s, at, np.log([float(knots[x]) for x in at]))).astype(np.float32)

    def _interlobe_sand(self, ntg, erosion, floor):
        """Sand fraction of every cell ``(nx, ny, nz)`` of the stamps' stack with interlobe mud, and its outcome.

        The mud cap ``M`` (see ``_stamp_mud``) is found by bisection so that the mean sand fraction of the layer is
        ``ntg``: it falls as ``M`` grows, from 1 at ``M = 0``. A layer whose sand exceeds ``ntg`` even with the
        thickest cap the stack can have (a fringe holds ``floor`` of sand, erosion cuts caps) is built with that cap,
        with a warning. Returns the fractions and ``dict(mud_thickness_m, amalgamated, sand_fraction, net_cells)``,
        the last the share of cells of sand fraction one half or more.
        """
        nz, dz, surfaces = self.nz, self.dz, self.allsurface

        def stack(mud_cap):
            mud, amalgamated = _stamp_mud(surfaces, nz, dz, mud_cap, erosion, floor)
            return mud, amalgamated, 1.0 - float(mud.mean()) / dz

        if not 0.0 < ntg < 1.0:
            raise ValueError(f"the net-to-gross must lie between 0 and 1, not {ntg}")
        lo, hi = 0.0, (1.0 - floor) * dz * max(float(np.max(b - a)) for a, b in zip(surfaces[:-1], surfaces[1:]))
        least = stack(hi)[2]
        if least > ntg:
            warnings.warn(f"the net-to-gross {ntg} is below the least this stack of stamps can hold, {least:.3f} (a mud cap "
                          f"as thick as the thickest stamp, a fringe sand fraction of {floor}, erosion {erosion}): "
                          f"it will be {least:.3f}", stacklevel=3)
            lo = hi
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if stack(mid)[2] > ntg else (lo, mid)
        mud, amalgamated, sand = stack(hi)
        return (np.swapaxes(np.clip(1.0 - mud / dz, 0.0, 1.0), 0, -1),
                dict(mud_thickness_m=hi, amalgamated=amalgamated, sand_fraction=sand,
                     net_cells=float((mud <= 0.5 * dz).mean())))

    def _lobemodeling(self, dh_ave=4.0, dh_std=0.5, r_ave=450.0, r_std=20.0,
                      asp=1.5, azimuth=0.0, azimuth_std=10.0, m=100,
                      upthinning=True, bouma_factor=0, compensation_scale=None, clip_decay=False):
        facies = np.zeros((self.nz, self.ny, self.nx))
        poro = facies.copy() - 0.1
        allsurface = []
        surface = 0.000001 * np.ones((self.ny, self.nx))
        surface0 = surface.copy()
        lat_size = self.nx * self.ny
        loc_idx = np.arange(lat_size)
        # Major axis lies along (cos(az), -sin(az)) — the same direction
        # the fluvial engine uses for channel flow at the same azimuth,
        # so a lobe with asp=2 at azimuth=N elongates in the same map
        # direction a channel at azimuth=N would flow in. Equivalent to
        # rotating cell offsets clockwise by ``azimuth`` before the
        # ellipse check.
        theta_base = np.deg2rad(azimuth)
        azimuth_std_rad = np.deg2rad(azimuth_std)
        allfacies = []
        allporo = []
        allsurface.append(surface.copy())

        i = 0
        iiii = 10000
        while i < iiii - 1:
            surface0 = surface.copy()
            zz = surface
            if compensation_scale is None:
                prob = (1 / (surface - zz.min() + 0.001)**m) / np.sum(1 / (surface - zz.min() + 0.001))
                prob = prob / np.sum(prob)
            else:
                prob = _compensation_weights(surface, self.dz, dh_ave, compensation_scale)
            prob_flat = prob.flatten()
            loc = np.random.choice(loc_idx, p=prob_flat)
            y = loc // self.nx
            x = loc - self.nx * y

            theta = theta_base + np.random.normal(0, azimuth_std_rad)
            dh = _gauss_clip(dh_ave, dh_std, lo=1e-6)
            r = _gauss_clip(r_ave, r_std, lo=1e-6)

            self._update_surface(x, y, r, asp, theta, dh, surface)
            if i != 0:
                surface2 = surface.copy()
                surface = surface0 + (surface - surface0) * (1 - (surface0 / surface0.max())**1.2)
                dsurface = surface - surface0
                surface = surface0 + dsurface * (np.sum(surface2 - surface0) / np.sum(dsurface + 0.000000001))

            dz = surface - surface0
            ychange, xchange = np.where(dz > 0)
            allsurface.append(surface.copy())
            self._assign_prop(xchange, ychange, x, y, theta, surface0, surface,
                              facies, r, poro, dz, asp, i + 1, upthinning, bouma_factor, clip_decay)
            i += 1
            # Terminate when the LOWEST column has filled to the grid
            # top — i.e. compensational stacking has filled the entire
            # basin. Stopping on ``surface.max()`` instead would fire
            # the moment any single tall lobe pokes through the top,
            # leaving the late stamps to dominate the upper part of the
            # column they cover (a contiguous sand "ceiling" artifact).
            if i == iiii or surface.min() >= self.nz:
                allfacies.append(facies.copy())
                allporo.append(poro.copy())
                break
        return allfacies, allporo, allsurface

    def _update_surface(self, x, y, r, asp, theta, dh, surface):
        # Geometry is done in physical (m) space: cell offsets are
        # scaled by ``self.dx`` / ``self.dy`` independently before the
        # rotation, so the lobe stays a true ellipse for any azimuth
        # and any cell aspect (dx != dy). ``surface`` is in cell-z
        # units, so the per-cell deposit ``dz0`` (m) is divided by
        # ``self.dz`` before accumulation.
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        # Tight rotated-ellipse half-extent (Steiner) along physical axes
        half_x_m = np.sqrt((r * asp * cos_t) ** 2 + (r * sin_t) ** 2)
        half_y_m = np.sqrt((r * asp * sin_t) ** 2 + (r * cos_t) ** 2)
        bx = int(np.ceil(half_x_m / self.dx)) + 1
        by = int(np.ceil(half_y_m / self.dy)) + 1
        r2 = r * r
        inv_dz = 1.0 / self.dz
        for ii in range(max(0, y - by), min(y + by + 1, self.ny)):
            dy_m = (ii - y) * self.dy
            for jj in range(max(0, x - bx), min(x + bx + 1, self.nx)):
                dx_m = (jj - x) * self.dx
                ax_m = dx_m * cos_t - dy_m * sin_t   # along major axis
                cr_m = dx_m * sin_t + dy_m * cos_t   # cross axis
                r1_sq = (ax_m / asp) ** 2 + cr_m ** 2
                if r1_sq <= r2:
                    dz0_m = dh * (1.0 - r1_sq / r2)
                    surface[ii, jj] += dz0_m * inv_dz

    def _assign_prop(self, xchange, ychange, x, y, theta, surface0, surface,
                     facies, r, poro, dz, asp, i, upthinning, bouma_factor, clip_decay=False):
        # ``r`` is in meters; cell offsets are scaled per-axis to
        # physical coords before the rotation so the radial ratio
        # ``r1 / r`` (used for the porosity decay) is correct in
        # physical space at any azimuth and any ``dx != dy``.
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        for n in range(xchange.size):
            ii = ychange[n]
            jj = xchange[n]
            dx_m = (jj - x) * self.dx
            dy_m = (ii - y) * self.dy
            ax_m = dx_m * cos_t - dy_m * sin_t
            cr_m = dx_m * sin_t + dy_m * cos_t
            r1 = np.sqrt((ax_m / asp) ** 2 + cr_m ** 2)
            bot = int(np.rint(surface0[ii, jj]))
            top = int(min(np.rint(surface[ii, jj]), self.nz))
            # Use continuous surface values for smooth porosity gradients
            surf_bot = surface0[ii, jj]
            surf_top = min(surface[ii, jj], float(self.nz))
            thickness = surf_top - surf_bot
            poromin = 0.05
            poromax = 0.3 * ((1 - surf_bot / self.nz) / 2 + 0.5) + 0.05 if upthinning else 0.35
            if top > bot and thickness > 1e-10:
                facies[bot:top, ii, jj][facies[bot:top, ii, jj] == 0] = i
                for kk in np.arange(bot, top):
                    upthinning_factor = ((1 - kk / self.nz) / 2 + 0.5) if upthinning else 1
                    vert_decay = max((surf_top - kk) / thickness, 0.0)
                    if clip_decay:      # a thin stamp's lower cell face lies below its base: the decay would pass 1
                        vert_decay = min(vert_decay, 1.0)
                    poro[kk, ii, jj] = 0.3 * vert_decay * (1 - (r1 / r)) * upthinning_factor + 0.05
                    poronorm = (poro[kk, ii, jj] - poromin) / (poromax - poromin)
                    bouma_seq_lims = [0, 0.1, 0.2, 0.3, 0.4, 1]
                    for bouma_idx in range(len(bouma_seq_lims) - 1):
                        if bouma_seq_lims[bouma_idx] <= poronorm < bouma_seq_lims[bouma_idx + 1]:
                            bouma_seq_mid = (bouma_seq_lims[bouma_idx] + bouma_seq_lims[bouma_idx + 1]) / 2
                            poronorm = (1 - bouma_factor) * (poronorm - bouma_seq_mid) + bouma_seq_mid
                            break
                    poro[kk, ii, jj] = poromin + poronorm * (poromax - poromin)
