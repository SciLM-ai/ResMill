import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

from .base import Layer
from .channel import FACIES_PROPS, _correlated_noise
from .drapes import thin_barrier
from ._fluvial import _gauss_clip

# ``facies_props`` entries a lobe uses: -1 mud rock, 2 the heterolithic fringe (its rock, and ``ntg_floor``, the sand share of a
# lobe's margin, which ``interlobe_erosion`` needs), 3 the sand (its rock is poro_ave, perm_ave, poro_std and perm_std).
_FACIES_KEYS = {-1: {"poro", "log10_perm", "poro_sd", "log10_perm_sd", "kvkh"},
                2: {"ntg_floor", "poro", "log10_perm", "poro_sd", "log10_perm_sd", "kvkh"}, 3: {"kvkh"}}
_SAND_SHARE_MIN = 0.5    # a cell with at least this sand share is sand (3)
_MUD_SHARE_MAX = 0.2     # below this sand share it is mud (-1), the Tanqua's distal fringe (under 20 % sandstone); in between the heterolithic fringe (2)
_MIN_CAP_SHARE = 1e-3    # mud thinner than this share of a cell is dust of the overlap arithmetic, not a thin cap
_NET_LOG10_PERM = 0.0    # a facies is net rock above 1 mD (ResSimMill's net cut-off)


def _compensation_weights(surface, dz, dh_ave, scale):
    """Probability that each column is the next stamp's centre, ``(ny, nx)``.

    ``exp(-deficit / (scale * dh_ave))``, the deficit being the column's height above the
    lowest column in metres (``surface`` is in cells of ``dz``) and ``dh_ave`` the stamps'
    peak thickness: a column one ``scale`` of stamp thicknesses higher is e times less
    likely. A vanishing scale always picks the lowest column, a large one is random.
    """
    weight = np.exp(-(surface - surface.min()) * dz / (scale * dh_ave))
    return weight / weight.sum()


def _stamp_mud(allsurface, nz, dz, mud_cap, erosion, floor):
    """Metres of mud in every cell of a stack of stamps, ``(nz, ny, nx)``, where it sits in the cell, and the share of caps cut through.

    ``allsurface`` holds the surface after every stamp (cells of ``dz``, the first the empty floor), so a stamp's
    deposit at a column is the rise of the surface there. Each deposit of thickness ``t`` is capped with mud on its
    top, ``min((1 - floor) t, mud_cap)`` thick: ``floor`` is the sand fraction of a thin deposit (the lobe fringe), so
    the sand fraction of a lobe falls from ``1 - mud_cap / t`` in its thick part to ``floor`` at its margin. A younger
    stamp scours ``erosion x`` its own thickness below its base, and the mud in that reach is gone, replaced by its sand
    (sand on sand: amalgamation), whichever older deposit it belongs to, a thin one between included; so a cap
    keeps its part below the deepest reach of any later stamp, and none of it where that reach passes its base. The
    share returned is the caps of at least a tenth of a cell met by a later stamp that were cut through. Mud above the top of the
    layer is not counted. Stamps are taken last to first, the deepest reach so far kept per column.

    The second array is the first moment of the mud about each cell's base: the sum over its pieces of their thickness times the
    height of their middle in cell heights, so ``moment / mud`` is where the mud sits in the cell, 0 at its base and 1 at its top.
    """
    shape = allsurface[0].shape
    mud = np.zeros((nz,) + shape)
    moment = np.zeros_like(mud)
    reach = np.full(shape, np.inf)
    met = cut = 0
    for before, after in zip(allsurface[-2::-1], allsurface[:0:-1]):
        thickness = (after - before) * dz
        rows, cols = np.nonzero(thickness > 0.0)
        if rows.size == 0:
            continue
        t, top, deepest = thickness[rows, cols], after[rows, cols] * dz, reach[rows, cols]
        cap = np.minimum((1.0 - floor) * t, mud_cap)
        length = np.maximum(np.minimum(top, deepest) - (top - cap), 0.0)
        counted = (cap > 0.1 * dz) & np.isfinite(deepest)
        met += int(counted.sum())
        cut += int((counted & (length == 0.0)).sum())
        bottom = top - cap
        first = np.floor(bottom / dz).astype(int)
        for step in range(int(np.ceil(length.max() / dz)) + 1):
            k = first + step
            low, high = np.maximum(bottom, k * dz), np.minimum(bottom + length, (k + 1) * dz)
            overlap = np.clip(high - low, 0.0, None)
            ok = (k >= 0) & (k < nz) & (overlap > 0.0)
            mud[k[ok], rows[ok], cols[ok]] += overlap[ok]
            moment[k[ok], rows[ok], cols[ok]] += overlap[ok] * (0.5 * (low + high)[ok] / dz - k[ok])
        reach[rows, cols] = np.minimum(deepest, before[rows, cols] * dz - erosion * t)
    return mud, moment, (cut / met if met else 0.0)


def _cap_cells(mud, moment, dz):
    """What the mud of a stack makes of every cell: its sand share, its facies and the thin caps, all ``(nx, ny, nz)`` with k up.

    A cell is one facies, by what fills it: sand (3) from half sand, the heterolithic fringe (2) from a fifth to half (the
    Tanqua's classes, Spychala et al. 2017), mud (-1) below, with the share ``s = 1 - mud / dz``; no cell is a mix. The mud in
    a sand cell is under half of it, too thin to be a cell: it is a thin cap, put on the face of the cell nearest its middle
    (the upper face if the mud's middle is in the upper half of the cell, else the lower). ``thin[..., k]`` is the metres of such
    mud on the lower face of cell ``k`` (the face between ``k - 1`` and ``k``); the layer's base and top carry none.
    """
    s = np.clip(1.0 - mud / dz, 0.0, 1.0)
    facies = np.where(s >= _SAND_SHARE_MIN, 3, np.where(s >= _MUD_SHARE_MAX, 2, -1)).astype(np.int8)
    inside = (facies == 3) & (mud > _MIN_CAP_SHARE * dz)
    height = np.divide(moment, mud, out=np.zeros_like(mud), where=mud > 0.0)
    thin = np.zeros_like(mud)
    thin[..., 1:] += np.where(inside & (height >= 0.5), mud, 0.0)[..., :-1]      # the upper face of cell k is the lower face of k + 1
    thin[..., 1:] += np.where(inside & (height < 0.5), mud, 0.0)[..., 1:]
    return s, facies, thin


def _facies_rock(entry, shape):
    """Porosity and log10 permeability (mD) of a facies with a rock of its own, ``shape`` arrays: the entry's ``poro`` and
    ``log10_perm``, with ``poro_sd`` (relative) and ``log10_perm_sd`` (decades) as correlated noise when given."""
    poro = np.full(shape, float(entry["poro"]))
    log_perm = np.full(shape, float(entry["log10_perm"]))
    if entry.get("poro_sd"):
        poro = poro * np.exp(float(entry["poro_sd"]) * _correlated_noise(shape, 3.0))
    if entry.get("log10_perm_sd"):
        log_perm = log_perm + float(entry["log10_perm_sd"]) * _correlated_noise(shape, 3.0)
    return poro, log_perm


def _check_facies_props(facies_props, interlobe):
    """Refuse a ``facies_props`` a lobe does not read: an unknown facies or key, and with ``interlobe_erosion`` a fringe (2)
    without the rock of its own that every cell of it takes (``poro`` and ``log10_perm``)."""
    for code, entry in facies_props.items():
        if int(code) not in _FACIES_KEYS or set(entry) - _FACIES_KEYS[int(code)]:
            uses = "; ".join(f"{c}: {', '.join(sorted(k))}" for c, k in sorted(_FACIES_KEYS.items()))
            raise ValueError(f"facies_props[{code}] = {dict(entry)}: a lobe uses {uses}")
    if interlobe and not {"poro", "log10_perm"} <= set(facies_props.get(2, {})):
        raise ValueError("interlobe_erosion needs the fringe's rock: facies_props[2] with 'poro' and 'log10_perm'")


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
            (decades), correlated noise; ``kvkh`` entries (``-1`` mud, ``2`` fringe, ``3`` sand) give the
            vertical to horizontal permeability ratio of every cell of the facies (``self.kvkh_mat``, PERMZ =
            kvkh x PERMX; none given: ``kzkx``). Every cell is one facies and has that facies' rock, nothing is
            mixed. ``2``, the heterolithic fringe, is used with ``interlobe_erosion`` only: its
            ``ntg_floor`` is the sand fraction of a lobe's margin, and it has a rock of its own
            (``poro``, ``log10_perm``, ``poro_sd``, ``log10_perm_sd``, as the mud).
        interlobe_erosion : float or None
            Needs ``facies_props``. ``None``: the sand is as above. A number replaces the sand's
            definition by a stack of stamps whose deposits carry interlobe mud: each stamp
            (a lobe) is capped with mud ``min((1 - f) t, M)`` thick, ``t`` its thickness at the
            column, ``f`` the sand fraction of its margin (``facies_props[2]["ntg_floor"]``,
            0 if absent) and ``M`` the cap thickness; the sand fraction of a lobe then falls from
            ``1 - M / t`` where it is thick to ``f`` at its margin (axis 85-100 %, off-axis
            50-85 %, fringe 20-50 % in the Tanqua, Spychala et al. 2017) with the mud as a
            continuous sheet over it. A younger stamp scours ``interlobe_erosion x`` its own
            thickness below its base and the mud in that reach is gone, replaced by its sand
            (sand on sand, amalgamation, where it is thick, the axis, and none where it is thin,
            the margin), whichever older cap it reaches, so the mud wraps each lobe at a low
            net-to-gross and is gone from the axes at a high one. ``M`` is found so that the share of
            net cells is ``ntg``. Each cell is one facies, by the share of it that is sand: sand (3,
            half or more), the heterolithic fringe (2, a fifth to a half, with the fringe's rock) or mud
            (-1, under a fifth), as channel, levee and floodplain cells are, with the rock and the
            kv/kh of its facies and no mixing rule. A cap fills cells where it is thick: the mud of
            a cap half a cell thick or more makes mud cells (or fringe cells at its edge). The mud in a sand
            cell, under half of it, is a thin cap and no cell: it is a vertical transmissibility barrier on the Z
            face nearest its middle, the thin-barrier factor of the mud drapes
            (:func:`resmill.layers.drapes.thin_barrier`, with the cap's thickness, the mud's permeability and the
            harmonic mean of the two cells' PERMZ), written as MULTZ: ``self.mult_z``, or None if there is no
            thin cap. Net cells (``ntg`` counts them, they are the cells above 1 mD) are the sand and, if
            its ``log10_perm`` is above 0, the fringe. ``self.sand_fraction`` holds each cell's sand share
            and ``self.interlobe`` the cap thickness ``mud_thickness_m``, the share of
            contacts ``amalgamated``, ``net_cells`` (the realized share), ``sand_fraction`` (the mean sand
            share, net-to-gross by thickness), ``aim_missed`` and ``barrier_faces`` (the share of the layer's
            interior faces with a multiplier under a half). It also keeps the stamps' porosity decay at 1 (``clip_decay``).
        """
        self.poro_ave = poro_ave
        self.perm_ave = perm_ave
        self.poro_std = poro_std
        self.perm_std = perm_std
        self.ntg = ntg

        if interlobe_erosion is not None and facies_props is None:
            raise ValueError("interlobe_erosion needs facies_props: the mud and the sand have rock of their own")
        if facies_props is not None:
            _check_facies_props(facies_props, interlobe_erosion is not None)
        self.mult_z = None
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
            fringe = facies_props[2]
            mud, moment, self.interlobe = self._interlobe_cells(
                ntg, interlobe_erosion, float(fringe.get("ntg_floor", 0.0)), float(fringe["log10_perm"]) > _NET_LOG10_PERM)
            self._calibrated_rock(structure, jitter, facies_props, stack=(mud, moment))

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

    def _calibrated_rock(self, structure, jitter, facies_props, stack=None):
        """Opt-in rock by facies (see ``create_geology``'s ``facies_props``): sets ``active``,
        ``facies``, ``poro_mat``, ``perm_mat``, ``sand_fraction``, ``mult_z`` and, with a ``kvkh``, ``kvkh_mat``.

        Every cell is one facies, by its sand share ``s``: the binary sand mask (sand and mud), or with ``stack``,
        the ``(mud, moment)`` of the interlobe stamps (:func:`_cap_cells`: sand, fringe and mud, and the thin caps). The
        sand-bed rock keeps the lobe structure's porosity (best at the axis and the base of a bed) standardized over the sand
        cells to ``poro_ave`` and ``poro_std``, and its log permeability follows with the slope ``perm_std / poro_std``,
        a small jitter, and an average of exactly ``perm_ave``. The fringe and the mud have the rock their entries give;
        no cell mixes the rock of two facies.
        """
        props = {code: {**FACIES_PROPS.get(code, {}), **facies_props.get(code, {})} for code in _FACIES_KEYS}
        if stack is None:
            s = self.active.astype(float)
            facies = np.where(s >= _SAND_SHARE_MIN, 3, -1).astype(np.int8)
        else:
            s, facies, thin = _cap_cells(*stack, self.dz)
        sand = facies == 3
        weight = sand / sand.sum()
        mean = (weight * self.poro_mat).sum()
        z = (self.poro_mat - mean) / max(np.sqrt((weight * (self.poro_mat - mean) ** 2).sum()), 1e-12)
        poro_sd, perm_sd = self.poro_std, self.perm_std
        poro_sand = np.clip(self.poro_ave + poro_sd * z, self.poro_ave - 5 * poro_sd, self.poro_ave + 5 * poro_sd)
        poro_sand = poro_sand + self.poro_ave - (weight * poro_sand).sum()      # the clip moves the mean a little
        log_sand = self.perm_ave + perm_sd / max(poro_sd, 1e-6) * (poro_sand - self.poro_ave) + jitter
        log_sand = np.clip(log_sand, self.perm_ave - 5 * perm_sd, self.perm_ave + 5 * perm_sd)
        log_sand = log_sand + self.perm_ave - (weight * log_sand).sum()
        shape = s.shape
        poro_mud, log_mud = _facies_rock(props[-1], shape)
        poro, log_perm = np.where(sand, poro_sand, poro_mud), np.where(sand, log_sand, log_mud)
        if stack is not None:
            fringe = facies == 2
            poro_fringe, log_fringe = _facies_rock(props[2], shape)
            poro, log_perm = np.where(fringe, poro_fringe, poro), np.where(fringe, log_fringe, log_perm)
        self.poro_mat = poro
        self.perm_mat = 10.0 ** log_perm
        self.sand_fraction = s.astype(np.float32)
        self.active = sand.astype(np.int8)
        self.facies = facies
        if any("kvkh" in entry for entry in props.values()):
            kvkh = np.where(sand, props[3].get("kvkh", self.kzkx), props[-1].get("kvkh", self.kzkx))
            if stack is not None:
                kvkh = np.where(facies == 2, props[2].get("kvkh", self.kzkx), kvkh)
            self.kvkh_mat = kvkh.astype(np.float32)
        if stack is not None and (thin > 0.0).any():
            self.mult_z = self._thin_cap_multipliers(thin, 10.0 ** float(props[-1]["log10_perm"]))
            self.interlobe["barrier_faces"] = float((self.mult_z[..., 1:] < 0.5).mean())

    def _thin_cap_multipliers(self, thin, mud_perm):
        """MULTZ of the thin caps (``thin``: metres on the lower face of each cell, k up), ``(nx, ny, nz)`` float32, 1 elsewhere.

        The thin-barrier factor of the mud drapes (:func:`resmill.layers.drapes.thin_barrier`) for the metres of mud on a face,
        ``mud_perm`` (mD) and the harmonic mean of the PERMZ of the cell above and the cell below, in cells of ``dz``.
        """
        kz = self.perm_mat * (self.kzkx if self.kvkh_mat is None else self.kvkh_mat)
        k_lo, k_hi = kz[..., :-1], kz[..., 1:]
        ks = 2.0 * k_lo * k_hi / (k_lo + k_hi)
        mult = np.ones(kz.shape, dtype=np.float32)
        faces = thin[..., 1:] > 0.0
        mult[..., 1:][faces] = thin_barrier(thin[..., 1:][faces], self.dz, ks[faces], mud_perm)
        return mult

    def _interlobe_cells(self, ntg, erosion, floor, fringe_net):
        """The mud of the stamps' stack as ``(mud, moment, outcome)``, ``mud`` and ``moment`` ``(nx, ny, nz)`` with k up.

        The mud cap ``M`` (see ``_stamp_mud``) is found by bisection so that the share of net cells is ``ntg``: a cell is
        net from a sand share of one half, or from a fifth when the fringe is net rock (``fringe_net``, its permeability above
        1 mD). The share falls as ``M`` grows, from 1 at ``M = 0``. A layer whose net share exceeds ``ntg`` even with the
        thickest cap the stack can have (a fringe holds ``floor`` of sand, erosion cuts caps) is built with that cap, with a
        warning. ``outcome`` is ``dict(mud_thickness_m, amalgamated, net_cells, sand_fraction, aim_missed)``, the realized
        share of net cells, the mean sand share (net-to-gross by thickness) and whether the aim was out of reach.
        """
        nz, dz, surfaces = self.nz, self.dz, self.allsurface
        net_share = _MUD_SHARE_MAX if fringe_net else _SAND_SHARE_MIN

        def stack(mud_cap):
            mud, moment, amalgamated = _stamp_mud(surfaces, nz, dz, mud_cap, erosion, floor)
            return mud, moment, amalgamated, float((1.0 - mud / dz >= net_share).mean())

        if not 0.0 < ntg < 1.0:
            raise ValueError(f"the net-to-gross must lie between 0 and 1, not {ntg}")
        lo, hi = 0.0, (1.0 - floor) * dz * max(float(np.max(b - a)) for a, b in zip(surfaces[:-1], surfaces[1:]))
        least = stack(hi)[3]
        missed = least > ntg
        if missed:
            warnings.warn(f"the net-to-gross {ntg} is below the least this stack of stamps can hold, {least:.3f} (a mud cap "
                          f"as thick as the thickest stamp, a fringe sand fraction of {floor}, erosion {erosion}): "
                          f"it will be {least:.3f}", stacklevel=3)
            lo = hi
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if stack(mid)[3] > ntg else (lo, mid)
        mud, moment, amalgamated, net = stack(hi)
        return (np.swapaxes(mud, 0, -1), np.swapaxes(moment, 0, -1),
                dict(mud_thickness_m=hi, amalgamated=amalgamated, net_cells=net, sand_fraction=1.0 - float(mud.mean()) / dz,
                     aim_missed=bool(missed)))

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
