"""Fluvial channel simulation engine — port of Alluvsim's ``streamsim.for``.

The driver is :py:class:`fluvial`, whose ``simulation()`` method ports the
per-event loop in ``streamsim.for:737-1004``. Per aggradation level we
draw an initial streamline, then loop events until the level NTG target is
met (or ``ntime`` is exhausted). Each event is one of:

* **Avulsion-outside** — abandon current streamline, draw fresh one from
  the candidate pool (``streamsim.for:757-774``).
* **Avulsion-inside** — abandon current streamline, splice a new tail from
  a curvature-weighted node (``avulsioninside.for``).
* **Migration** — stamp LA on OLD path, redraw geometry, run one bank-retreat
  step (``calcusb.for``: explicit 30-node Sun-1996 integral with
  ``exp(-2 Cf ds / h0)`` decay), neck-cut, then stamp CH on NEW path.

Conventions inside the engine match Alluvsim:

* Angles in **compass degrees** (0=+y, 90=+x, increasing CW). The AR(2)
  walk uses math-degree angles internally (``ang = 450 - chazi``) so
  ``x += step*cosd(ang); y += step*sind(ang)`` reproduces the channel
  azimuth without conversion gymnastics.
* Curvature is ``dazi/ds`` in deg/m, **compass-CW positive** (right turn
  = positive). The thalweg, splay-side, and levee-cutbank tests are all
  written against this sign convention so they match ``calc_levee.for``,
  ``streamsim.for:907-911`` and ``calc_lobe.for`` directly.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from ._genchannel import genchannel
from ._genabandoned import paint_abandoned
from ._calc_levee import paint_levee
from ._calc_lobe_splay import paint_lobe, paint_splay
from ._make_cutoff import make_cutoff

# Upper clip applied to every drawn sinuosity in ``_sample_streamline``;
# used to size the ``ndis_cap`` walk safety net from the grid diagonal.
_MAX_SINUOSITY = 1.9


# ---------------------------------------------------------------------------
# Alluvsim facies codes
# ---------------------------------------------------------------------------
FF, FFCH, CS, LV, LA, CH = -1, 0, 1, 2, 3, 4


def _gauss_clip(mean: float, stdev: float, lo: float = 0.0, hi: float | None = None) -> float:
    """One Gaussian draw, clipped — matches Alluvsim's per-event ``max(0,...)`` idiom."""
    val = float(np.random.normal(mean, stdev)) if stdev > 0 else float(mean)
    val = max(val, lo)
    if hi is not None:
        val = min(val, hi)
    return val


def _movwinsmooth(arr: np.ndarray, nwin: int) -> np.ndarray:
    """Triangular moving-window smoother (port of Alluvsim ``movwinsmooth.for``).

    Weights ``w[i] = (nwin - |i| + 1) / (nwin + 1)`` for ``|i| <= nwin``,
    normalised per evaluation point so edge handling matches Alluvsim's
    ``count``-based renormalisation.
    """
    n = arr.size
    if n == 0 or nwin <= 0:
        return arr.copy()
    out = np.empty_like(arr, dtype=np.float64)
    weights = np.array(
        [(nwin - abs(i) + 1) / (nwin + 1.0) for i in range(-nwin, nwin + 1)],
        dtype=np.float64,
    )
    for i in range(n):
        lo = max(0, i - nwin)
        hi = min(n, i + nwin + 1)
        wlo = lo - (i - nwin)
        whi = wlo + (hi - lo)
        w = weights[wlo:whi]
        out[i] = float((arr[lo:hi] * w).sum() / w.sum())
    return out


class fluvial:
    """Single-streamline fluvial simulator with Alluvsim-faithful event loop."""

    def __init__(
        self,
        # ---- grid (required) -------------------------------------------
        nx: int, ny: int, nz: int,
        xsiz: float, ysiz: float, zsiz: float,
        xmn: float = 0.0, ymn: float = 0.0,
        # ---- aggradation schedule --------------------------------------
        nlevel: int = 3,
        level_z: list[float] | None = None,
        NTGtarget: float = 0.10,
        ntime: int = 120,
        # ---- avulsion probabilities ------------------------------------
        probAvulOutside: float = 0.10,
        probAvulInside: float = 0.05,
        # ---- channel geometry (per-event Gaussian draws) ---------------
        mCHdepth: float = 2.5, stdevCHdepth: float = 0.3, stdevCHdepth2: float = 0.2,
        mCHwdratio: float = 10.0, stdevCHwdratio: float = 1.0,
        mCHsinu: float = 1.6, stdevCHsinu: float = 0.15,
        mCHazi: float = 90.0, stdevCHazi: float = 1.0,
        mCHsource: float | None = None, stdevCHsource: float = 80.0,
        # ---- migration --------------------------------------------------
        mdistMigrate: float = 35.0, stdevdistMigrate: float = 10.0,
        # ---- levee (LV) — Alluvsim makepar-style defaults --------------
        mLVdepth: float = 1.0, stdevLVdepth: float = 0.2,
        mLVwidth: float = 40.0, stdevLVwidth: float = 5.0,
        mLVheight: float = 0.5, stdevLVheight: float = 0.1,
        mLVasym: float = 0.3, stdevLVasym: float = 0.1,
        mLVthin: float = 0.3, stdevLVthin: float = 0.1,
        # ---- crevasse splay (CS) — makepar defaults --------------------
        mCSnum: float = 2.0, stdevCSnum: float = 0.5,
        mCSnumlobe: float = 3.0, stdevCSnumlobe: float = 1.0,
        mCSsource: float = 50.0, stdevCSsource: float = 20.0,
        mCSLOLL: float = 200.0, stdevCSLOLL: float = 50.0,
        mCSLOWW: float = 30.0, stdevCSLOWW: float = 10.0,
        mCSLOl: float = 100.0, stdevCSLOl: float = 20.0,
        mCSLOw: float = 20.0, stdevCSLOw: float = 10.0,
        mCSLO_hwratio: float = 0.03, stdevCSLO_hwratio: float = 0.01,
        mCSLO_dwratio: float = 0.02, stdevCSLO_dwratio: float = 0.005,
        # ---- abandoned-channel mud plug (FFCH) -------------------------
        mFFCHprop: float = 0.0, stdevFFCHprop: float = 0.0,
        # Mud-plug fraction painted on the *neck-cutoff oxbow loop* every
        # time ``make_cutoff`` excises a tight bend from the active
        # centerline. 0.0 = pure Alluvsim behavior (the abandoned bend
        # keeps its prior LA/CH stamps and is later overwritten as the
        # new shorter centerline migrates). >0 = the cells of the dropped
        # oxbow are converted to FFCH (mud plug) over the upper
        # ``mNeckFFCHprop`` of their column at that node's CHelev — i.e.
        # the oxbow lake fills with mud, exactly as a real abandoned
        # bend does after a neck cutoff.
        mNeckFFCHprop: float = 0.0,
        # ---- hydraulic — Alluvsim makepar central values --------------
        Cf: float = 0.0078, A: float = 2.0, I: float = 0.001, Q: float = 5.0,
        # ---- pool / discretisation -------------------------------------
        CHndraw: int = 50, ndiscr: int = 5, nCHcor: int = 10,
        # ---- back-compat (kept for DeltaLayer-style internal callers) --
        azimuth: float = 0.0,
        # ---- delta extensions (used by DeltaLayer; harmless defaults) --
        # Fraction of proximal streamline nodes protected from
        # avulsion-inside splicing. 0.0 = no protection (Alluvsim
        # default). >0 keeps the upstream trunk geometrically stable
        # so it meanders as one channel before the fan starts to
        # bifurcate downstream — exactly the architecture of a real
        # delta-plain.
        min_avul_node_frac: float = 0.0,
        # Per-level x-offset of the streamline entry (for delta-style
        # progradation). ``None`` = entry pinned at xmin every level
        # (Alluvsim default). When supplied, the pool is rebuilt at
        # each level with ``x0 = xmin + offset[ilevel]`` so the apex
        # advances toward +flow as generations stack vertically — the
        # classic clinoform-progradation architecture.
        mCHentry_x_offset_per_level: list[float] | None = None,
        # Standard deviation (in compass degrees) of the random perturbation
        # added to ``local_azi`` when an avulsion-inside event launches a
        # fresh AR(2) tail. 0.0 = Alluvsim default (tail launches along
        # exact parent direction at the splice node); >0 spreads the new
        # branches around the parent direction immediately at launch,
        # giving direct control over the angular fan-out between sibling
        # distributaries (used by DeltaLayer's ``branch_spread_deg``).
        stdev_branch_azi: float = 0.0,
        # ---- several entry points (channel presets) --------------------
        # ``n_sources`` > 1: that many sources are drawn along the upstream
        # edge at simulation start, at least ``source_spacing_min`` of the
        # edge apart and away from its ends, and every streamline enters at
        # one of them (``mCHsource`` +- ``stdevCHsource`` becomes source +-
        # ``stdevCHsource``). With the default 1 nothing changes.
        n_sources: int = 1,
        source_spacing_min: float = 0.15,
        # ---- distributary tree (DeltaLayer ``bifurcate`` mode) --------
        # When ``True`` a level is not an event loop but one branching
        # network grown from the level's trunk: ``n_bifurcations`` splits,
        # each handing ``U(split_frac_lo, split_frac_hi)`` of the parent's
        # discharge to a new branch launched at ``branch_angle_mean`` +-
        # ``branch_angle_sd`` degrees from the local heading. Width scales
        # as ``Q ** width_exp`` and depth as ``Q ** depth_exp``
        # (Leopold-Maddock), a branch runs at most
        # ``branch_length_scale * diagonal * sqrt(Q)`` and then ends on the
        # plain (tip recorded for a mouth bar), branches below ``q_min``
        # never split, and with ``merge_branches`` a walk that runs into
        # another branch joins it. Channels never use this.
        bifurcate: bool = False,
        n_trees: int = 1,
        n_bifurcations: int = 8,
        split_frac_lo: float = 0.25,
        split_frac_hi: float = 0.5,
        branch_angle_mean: float = 55.0,
        branch_angle_sd: float = 15.0,
        q_min: float = 0.05,
        branch_length_scale: float = 0.6,
        max_splits_per_branch: int = 2,
        # where along a segment a split lands: 1 = uniform beyond the
        # protected reach, >1 biased toward the segment's upstream end
        # (Wax Lake style: the primaries leave within a short reach)
        split_pos_exp: float = 1.0,
        # a split is a Y: the parent's downstream reach is re-walked,
        # deflected away from the new branch in proportion to the share the
        # branch takes (only when nothing else hangs off that reach yet)
        y_split: bool = True,
        # how far a branch's mean heading eases back from its launch angle
        # toward the regional slope: 0 keeps the launch angle, 1 runs with
        # the regional flow after the first bend
        branch_relax: float = 0.5,
        # width at a branch's end relative to its start (discharge lost to
        # the plain and to unresolved splits along the way)
        branch_taper: float = 0.7,
        # the delta front. A branch ends where its own lobe ends, at
        # ``front_radius * (1 + front_bulge * sqrt(Q)) * exp(N(0, front_jitter))``
        # walk-frame x spans from the apex: bigger channels prograde further,
        # so the shoreline is scalloped (large ``front_bulge``: bird's foot).
        # ``front_radius`` > 1 puts the front outside the grid and channels
        # run through to the edges. At every terminus the jet drops a mouth
        # bar and the channel splits around it into two short terminal
        # channels with bars of their own (the mouth-bar complex).
        front_radius: float = 1.5,
        front_bulge: float = 0.3,
        front_jitter: float = 0.1,
        merge_branches: bool = True,
        width_exp: float = 0.5,
        depth_exp: float = 0.4,
        # When ``True``, ``ntime`` is interpreted as the per-level event
        # cap and the global event counter is reset at the top of every
        # level — so each of the ``nlevel`` levels gets its own full
        # ``ntime`` event budget. Default ``False`` matches Alluvsim
        # ``streamsim.for`` semantics (``ntime`` is the *total* event cap
        # across all levels), which makes sense when the cumulative
        # ``NTGtarget`` is the dominant exit criterion. For multi-level
        # presets where each level is a stand-alone architectural unit
        # (e.g. ``MEANDER_OXBOW`` stacking eight independent meander
        # belts) this flag should be ``True`` so upper levels actually
        # run instead of being starved by the lower levels.
        ntime_per_level: bool = False,
        # ---- misc -------------------------------------------------------
        seed: int | None = None,
    ):
        # Grid
        self.nx, self.ny, self.nz = nx, ny, nz
        self.xsiz, self.ysiz, self.zsiz = xsiz, ysiz, zsiz
        self.xmn = xmn
        self.ymn = ymn
        self.xmin = xmn - 0.5 * xsiz
        self.ymin = ymn - 0.5 * ysiz
        self.xmax = self.xmin + xsiz * nx
        self.ymax = self.ymin + ysiz * ny
        self.x = np.linspace(xmn, self.xmax - xmn, nx)
        self.y = np.linspace(ymn, self.ymax - ymn, ny)

        # Aggradation
        self.nlevel = int(nlevel)
        if level_z is None:
            # Default chelev spread: bottom level at z = mCHdepth so the
            # bottom channel U fits fully inside the grid, top level at
            # z = nz·zsiz so the topmost channel reaches the grid top.
            # With ``nlevel ≥ ceil(nz·zsiz / mCHdepth)`` the column fills
            # contiguously with no FF gap between adjacent levels.
            z_top = float(nz * zsiz)
            z_bot = max(zsiz, mCHdepth)
            if self.nlevel == 1:
                level_z = [z_top]
            else:
                level_z = list(np.linspace(z_bot, z_top, self.nlevel))
        self.level_z = [float(z) for z in level_z]
        if len(self.level_z) != self.nlevel:
            raise ValueError(
                f"len(level_z)={len(self.level_z)} != nlevel={self.nlevel}")
        self.NTGtarget = float(NTGtarget)
        self.ntime = int(ntime)

        # Avulsion
        self.probAvulOutside = float(probAvulOutside)
        self.probAvulInside = float(probAvulInside)

        # Per-event geometry draws
        self.mCHdepth, self.stdevCHdepth = float(mCHdepth), float(stdevCHdepth)
        self.stdevCHdepth2 = float(stdevCHdepth2)
        self.mCHwdratio, self.stdevCHwdratio = float(mCHwdratio), float(stdevCHwdratio)
        self.mCHsinu, self.stdevCHsinu = float(mCHsinu), float(stdevCHsinu)
        self.mCHazi, self.stdevCHazi = float(mCHazi), float(stdevCHazi)
        self.mCHsource = float(mCHsource) if mCHsource is not None else 0.5 * (self.ymin + self.ymax)
        self.stdevCHsource = float(stdevCHsource)
        self.mdistMigrate, self.stdevdistMigrate = float(mdistMigrate), float(stdevdistMigrate)

        self.mLVdepth, self.stdevLVdepth = float(mLVdepth), float(stdevLVdepth)
        self.mLVwidth, self.stdevLVwidth = float(mLVwidth), float(stdevLVwidth)
        self.mLVheight, self.stdevLVheight = float(mLVheight), float(stdevLVheight)
        self.mLVasym, self.stdevLVasym = float(mLVasym), float(stdevLVasym)
        self.mLVthin, self.stdevLVthin = float(mLVthin), float(stdevLVthin)

        self.mCSnum, self.stdevCSnum = float(mCSnum), float(stdevCSnum)
        self.mCSnumlobe, self.stdevCSnumlobe = float(mCSnumlobe), float(stdevCSnumlobe)
        self.mCSsource, self.stdevCSsource = float(mCSsource), float(stdevCSsource)
        self.mCSLOLL, self.stdevCSLOLL = float(mCSLOLL), float(stdevCSLOLL)
        self.mCSLOWW, self.stdevCSLOWW = float(mCSLOWW), float(stdevCSLOWW)
        self.mCSLOl, self.stdevCSLOl = float(mCSLOl), float(stdevCSLOl)
        self.mCSLOw, self.stdevCSLOw = float(mCSLOw), float(stdevCSLOw)
        self.mCSLO_hwratio, self.stdevCSLO_hwratio = float(mCSLO_hwratio), float(stdevCSLO_hwratio)
        self.mCSLO_dwratio, self.stdevCSLO_dwratio = float(mCSLO_dwratio), float(stdevCSLO_dwratio)

        self.mFFCHprop, self.stdevFFCHprop = float(mFFCHprop), float(stdevFFCHprop)
        self.mNeckFFCHprop = float(mNeckFFCHprop)
        self.ntime_per_level = bool(ntime_per_level)

        # Hydraulic
        g = 9.8
        self.g = g
        self.Cf, self.A, self.I, self.Q = float(Cf), float(A), float(I), float(Q)
        self.us0 = ((g * self.Q * self.I) /
                    (self.mCHdepth * self.mCHwdratio * self.Cf))**(1.0 / 3.0)
        self.h0 = self.Q / (self.mCHdepth * self.mCHwdratio * self.us0)

        # Pool / discretisation
        self.CHndraw = int(CHndraw)
        self.ndiscr = int(ndiscr)
        self.nCHcor = int(nCHcor)

        # Delta extensions (no-ops when at default values).
        self.min_avul_node_frac = float(np.clip(min_avul_node_frac, 0.0, 0.95))
        self.stdev_branch_azi = float(max(stdev_branch_azi, 0.0))
        self.n_sources = int(max(n_sources, 1))
        self.source_spacing_min = float(np.clip(source_spacing_min, 0.0, 0.5))
        self._sources: list[float] | None = None
        self._last_src: int | None = None      # source of the streamline just sampled
        self._draw_counter = 0                 # rotates the pool draws through the sources
        self.bifurcate = bool(bifurcate)
        self.n_trees = int(max(n_trees, 1))
        self.n_bifurcations = int(max(n_bifurcations, 0))
        self.split_frac_lo = float(np.clip(split_frac_lo, 0.05, 0.5))
        self.split_frac_hi = float(np.clip(split_frac_hi, self.split_frac_lo, 0.5))
        self.branch_angle_mean = float(branch_angle_mean)
        self.branch_angle_sd = float(max(branch_angle_sd, 0.0))
        self.q_min = float(max(q_min, 1e-3))
        self.branch_length_scale = float(max(branch_length_scale, 0.05))
        self.max_splits_per_branch = int(max(max_splits_per_branch, 1))
        self.split_pos_exp = float(max(split_pos_exp, 0.2))
        self.y_split = bool(y_split)
        self.branch_relax = float(np.clip(branch_relax, 0.0, 1.0))
        self.branch_taper = float(np.clip(branch_taper, 0.1, 1.0))
        self.front_radius = float(max(front_radius, 0.05))
        self.front_jitter = float(max(front_jitter, 0.0))
        self.front_bulge = float(max(front_bulge, 0.0))
        self.merge_branches = bool(merge_branches)
        self.width_exp = float(width_exp)
        self.depth_exp = float(depth_exp)
        self.tree_branches: list[dict] = []
        if mCHentry_x_offset_per_level is None:
            self.mCHentry_x_offset_per_level = None
        else:
            offs = list(mCHentry_x_offset_per_level)
            if len(offs) != self.nlevel:
                raise ValueError(
                    f"mCHentry_x_offset_per_level length {len(offs)} "
                    f"!= nlevel {self.nlevel}")
            self.mCHentry_x_offset_per_level = [float(v) for v in offs]
        # Active entry-x offset; updated per level when progradation is on.
        self._entry_x_offset = 0.0
        # Endpoints of every streamline that ended up active at the
        # close of a level — used by DeltaLayer to paint optional
        # mouth-bar lobes at the prograding front.
        self.distal_tips: list[tuple[float, float, float]] = []

        # Azimuth (back-compat with delta-style rotated stamping)
        self.azimuth_rad = float(np.deg2rad(azimuth))
        self._cos_az = float(np.cos(self.azimuth_rad))
        self._sin_az = float(np.sin(self.azimuth_rad))
        self._pivot_x = 0.5 * (self.xmin + self.xmax)
        self._pivot_y = 0.5 * (self.ymin + self.ymax)

        # Streamline discretisation step. Match Alluvsim ``streamsim.for:593``:
        # ``step = (xsiz + ysiz) / 2``. ndis0 multiplier matches AL's *2.
        self.step = (self.xsiz + self.ysiz) / 2
        self.step0 = self.step
        # Every streamline is resampled to ``ndis0`` nodes, so ``ndis0`` sets
        # the node density and with it every per-node rule (cutoff necks,
        # width and depth draws along the channel; MEANDER_OXBOW net-to-gross
        # falls from 0.45 to 0.26 when the density is nearly tripled).
        # Alluvsim sizes it from the mean horizontal span, which on a square
        # grid is the span the walk crosses. Keep that density on any grid by
        # using the walk-frame x span, the span every streamline traverses.
        self.ndis0 = int((self.xmax - self.xmin) / self.step) * 2
        # The AR(2) walk itself is capped separately. The cap is a SAFETY
        # NET, not a stopping criterion: a channel stops when it leaves the
        # grid. Size it from the diagonal times the maximum sinuosity, times
        # 2, so even the most sinuous walk across the longest line of the
        # grid never runs out of nodes mid-domain. Overshooting is nearly
        # free because the walk exits as soon as it leaves the grid;
        # undershooting silently truncates (it did on elongated grids when
        # the cap was ``ndis0``: the far end never filled).
        diag = float(np.hypot(self.xmax - self.xmin, self.ymax - self.ymin))
        self.ndis_cap = int(diag / self.step * _MAX_SINUOSITY) * 2
        # "Walk died immediately, redraw it" threshold (Alluvsim: ndis0/10).
        self.ndis_min = max(2, self.ndis0 // 10)

        # Counters mutable from numba kernels
        self.ntg_counter = np.zeros(1, dtype=np.int64)
        self.ffch_counter = np.zeros(1, dtype=np.int64)

        # Output arrays
        self.facies = np.full((nx, ny, nz), FF, dtype=np.int8)
        self.poro = np.zeros((nx, ny, nz), dtype=np.float32)
        self.poro0 = 0.3

        # Per-cell auxiliary fields written alongside ``facies`` by every
        # event stamp. Used downstream by ``ChannelLayer._finalize_facies_table``
        # to apply the geologically-correct per-event Walker upward-fining
        # ramp + per-event poro/perm multipliers (see channel.py for the
        # combination formula).
        #
        # ``depth_norm[ix,iy,iz]`` ∈ [0, 1] for CH/LA cells: 0 at the top
        # of the channel cross-section, 1 at the base. 0.5 (neutral, ramp
        # = 1.0) for non-channel facies and inactive cells.
        # ``poro_mult_field`` defaults to 1.0; ``log_perm_offset_field`` to 0.0.
        self.depth_norm = np.full((nx, ny, nz), 0.5, dtype=np.float32)
        self.poro_mult_field = np.ones((nx, ny, nz), dtype=np.float32)
        self.log_perm_offset_field = np.zeros((nx, ny, nz), dtype=np.float32)

        # Per-event std for the K-C-coupled multiplier draws. Small —
        # the dominant scale of variability is now the per-realization
        # mult Sobol-sampled at the sample level (see ChannelLayer /
        # DeltaLayer ``poro_realization_mult`` / ``perm_realization_mult``).
        # Per-event mult is the wiggle WITHIN a realization, e.g. one
        # channel slightly cleaner than another in the same reservoir.
        self.poro_mult_std = 0.04
        self.log_perm_offset_std = 0.12

        # Cache for the current channel event's K-C-coupled poro/perm pair.
        # ``_stamp_channel`` redraws and refreshes this; ``_stamp_levee``
        # and ``_stamp_splays`` consume the same pair so a channel +
        # its levees/splays share one consistent depositional regime
        # (geologically: levees and splays form contemporaneously with
        # the channel they border, fed by the same flood pulse).
        self._event_poro_mult = 1.0
        self._event_log_perm_offset = 0.0

        if seed is not None:
            np.random.seed(int(seed))

        # Streamline state
        self.cx = None
        self.cy = None
        self.curv = None
        self.vx = None
        self.vy = None
        self.azi = None  # per-node azimuth in compass degrees (matches chamat[:,9])
        self.thalweg = None
        self.chelev = self.level_z[0]
        self.chelev_arr = None  # per-node elevation array (chamat[:,11])

        # Per-event channel geometry
        self.CHdepth = self.mCHdepth
        self.CHwdratio = self.mCHwdratio
        self.gr_dwratio = 2.0 / max(self.CHwdratio, 1e-6)  # depth / half-width
        self.CHhalfwidth = 0.5 * self.CHdepth * self.CHwdratio
        self.maxCHhalfwidth = self.CHhalfwidth

        # Per-streamline width perturbation (onedrf)
        self._chwidth_arr = None  # cached per-node halfwidth (chamat[:,5])
        self._chwidth_state_n = -1

        # Bookkeeping (back-compat)
        self.LV_asym = 0.9
        self.lV_height = 5 * zsiz
        self.totalid = 0
        self.out = 0
        self.ps = 1
        self.pavul = 0

        # First-streamline-of-sim flag (Alluvsim ``CHcurrent.gt.1`` guard).
        self._is_first_streamline = True

        # Streamline candidate pool (Alluvsim ``buildCHtable``)
        self._pool = None  # list of dicts {cx, cy, ndis, chazi, chsinu, chdepth, chwidth_arr, weight}

    # ----------------------------------------------------------------- AR(2) walk

    @staticmethod
    def _resc(s1, s2, t1, t2, x):
        """Linear rescale ``x in [s1,s2] → [t1,t2]`` (Alluvsim ``resc``)."""
        if s2 == s1:
            return t1
        return t1 + (t2 - t1) * (x - s1) / (s2 - s1)

    def _ar2_walk(self, x0: float, y0: float, chazi: float, chsinu: float,
                  ndis_max: int, max_attempts: int = 1000, azi0: float | None = None):
        """Disturbed-periodic (Pyrcz/Sun) AR(2) walk in **compass degrees**.

        Faithful port of Alluvsim ``buildCHtable.for:70-128``:
          ``k=0.3, h=0.8``
          ``b1 = 2 exp(-kh) cosd(k cosd(asind(h)))``
          ``b2 = -exp(-2kh)``
          ``m = (450 - chazi)*16.33/360``  (math-degree restoring force)
          ``s = resc(1.0, 2.0, 1.0, 13.0, chsinu)``  (deg/step)
          ``ang1 = ang2 = 450 - chazi``
          ``ang = b1*ang1 + b2*ang2 + (xp*s + m)``
          ``x += step*cosd(ang); y += step*sind(ang)``

        Restarts noise if the walk leaves the grid in the first ``ndis_min``
        nodes (matches AL's ``goto 512`` regen-on-short-failure).

        Returns ``(cx, cy)`` arrays of length ``>= ndis_min`` on success, or
        (None, None) after ``max_attempts`` failed regenerations.
        """
        k = 0.3
        h = 0.8
        phi = np.degrees(np.arcsin(h))
        b1 = 2.0 * np.exp(-k * h) * np.cos(np.radians(k * np.cos(np.radians(phi))))
        b2 = -1.0 * np.exp(-2.0 * k * h)
        m = (450.0 - chazi) * (16.33 / 360.0)
        s = self._resc(1.0, 2.0, 1.0, 13.0, chsinu)
        # ``azi0``: launch heading (compass) when it differs from the mean
        # heading ``chazi``; the restoring force then bends the walk from
        # the launch direction toward its mean over the first few tens of
        # steps (a distributary leaving its parent at an angle and easing
        # back toward the regional slope).
        ang_init = 450.0 - (chazi if azi0 is None else azi0)

        for _ in range(max_attempts):
            noise = np.random.normal(0.0, 1.0, ndis_max) * s + m
            cx = np.empty(ndis_max + 1, dtype=np.float64)
            cy = np.empty(ndis_max + 1, dtype=np.float64)
            cx[0] = x0
            cy[0] = y0
            ang1 = ang_init
            ang2 = ang_init
            short = False
            n_ok = 1
            for i in range(ndis_max):
                ang = b1 * ang1 + b2 * ang2 + noise[i]
                x = cx[i] + self.step * np.cos(np.radians(ang))
                y = cy[i] + self.step * np.sin(np.radians(ang))
                # Test where the channel actually lands, not where it is
                # walked: with azimuth != 0 the two differ by a rotation, and
                # testing the unrotated point bounds the walk by a ROTATED
                # copy of the grid. That leaves the grid's own corners
                # permanently empty and makes net-to-gross depend on azimuth
                # (0.564 at 0 and 90 deg, 0.534 at 45 -- a square overlaps
                # its own rotation least at 45).
                xr, yr = self._rot_xy(x, y)
                if xr > self.xmax or xr < self.xmin or yr > self.ymax or yr < self.ymin:
                    if i < self.ndis_min:
                        short = True
                        break
                    n_ok = i  # nodes 0..i-1 are in-grid; AL: ``ndis = i-1`` then exit
                    break
                cx[i + 1] = x
                cy[i + 1] = y
                ang2 = ang1
                ang1 = ang
                n_ok = i + 2
            if short:
                continue
            return cx[:n_ok].copy(), cy[:n_ok].copy()
        return None, None

    def _snap_entry(self, x0, y0):
        """Move a streamline entry onto the grid boundary along the mean flow.

        Entries are drawn on the upstream edge of the UNROTATED walk frame,
        and stamping rotates the streamline by ``azimuth`` about the grid
        centre. For any azimuth that is not a multiple of 90 degrees that
        edge lands partly inside the grid and partly outside it: a channel,
        or a whole delta fan, starts in the open floodplain with nothing
        feeding it, and entries outside the grid are wasted. Slide the entry
        along the mean flow direction to where its flow line crosses the grid
        boundary: backwards if the entry is inside, forwards if it is outside.
        Returns the walk-frame entry, or None when the flow line misses the
        grid.
        """
        ang = np.radians(450.0 - self.mCHazi)
        dx, dy = float(np.cos(ang)), float(np.sin(ang))
        qx, qy = self._rot_xy(x0, y0)
        ex = dx * self._cos_az + dy * self._sin_az
        ey = -dx * self._sin_az + dy * self._cos_az
        s_in, s_out = -np.inf, np.inf
        for q, e, lo, hi in ((qx, ex, self.xmin, self.xmax), (qy, ey, self.ymin, self.ymax)):
            if abs(e) < 1e-12:
                if q < lo or q > hi:
                    return None
                continue
            t1, t2 = (lo - q) / e, (hi - q) / e
            s_in, s_out = max(s_in, min(t1, t2)), min(s_out, max(t1, t2))
        if s_in > s_out:
            return None
        return x0 + s_in * dx, y0 + s_in * dy

    def _draw_sources(self):
        """Draw ``n_sources`` entry positions along the upstream edge.

        Positions are uniform over the middle 80% of everything the grid
        spans perpendicular to the flow, at least ``source_spacing_min`` of
        that span apart (rejection, up to 200 tries per source).
        """
        half = 0.5 * ((self.xmax - self.xmin) * abs(self._sin_az)
                      + (self.ymax - self.ymin) * abs(self._cos_az))
        lo, hi = self._pivot_y - 0.8 * half, self._pivot_y + 0.8 * half
        gap = self.source_spacing_min * 2.0 * half
        src = []
        for _ in range(self.n_sources):
            for _try in range(200):
                y = float(np.random.uniform(lo, hi))
                if all(abs(y - s) >= gap for s in src):
                    src.append(y)
                    break
        self._sources = sorted(src)

    def _sample_streamline(self):
        """Sample (x0, y0, chazi, chsinu) per Alluvsim ``buildCHtable.for:73-88``.

        The entry is drawn on the upstream edge of the walk frame, moved onto
        the grid boundary along the mean flow (``_snap_entry``), then advanced
        along the flow by ``entry_x_offset`` when delta-style progradation is
        configured (0 by default, Alluvsim semantics).
        """
        chazi = float(np.random.normal(self.mCHazi, max(self.stdevCHazi, 1e-9)))
        chsinu = float(np.random.normal(self.mCHsinu, max(self.stdevCHsinu, 1e-9)))
        chsinu = float(np.clip(chsinu, 1.1, 1.9))
        ang = np.radians(450.0 - self.mCHazi)
        dx, dy = float(np.cos(ang)), float(np.sin(ang))
        for _ in range(1000):
            if self.stdevCHsource > 0.0:
                centre = self.mCHsource
                self._last_src = None
                if self._sources:
                    self._last_src = int(np.random.randint(len(self._sources)))
                    centre = self._sources[self._last_src]
                y0 = float(np.random.normal(centre, self.stdevCHsource))
                y0 = float(np.clip(y0, self.ymin, self.ymax))
            else:
                # Uniform over everything the grid spans perpendicular to the
                # flow, so a rotated grid is entered along its whole width and
                # not only along the rotated image of one edge.
                half = 0.5 * ((self.xmax - self.xmin) * abs(self._sin_az)
                              + (self.ymax - self.ymin) * abs(self._cos_az))
                y0 = float(np.random.uniform(self._pivot_y - half, self._pivot_y + half))
            snapped = self._snap_entry(self.xmin, y0)
            if snapped is not None:
                return (snapped[0] + self._entry_x_offset * dx,
                        snapped[1] + self._entry_x_offset * dy, chazi, chsinu)
        raise RuntimeError('no streamline entry reaches the grid')

    def generate_streamline(self, x0=None, y0=None, chazi=None, chsinu=None) -> int:
        """Build a fresh streamline by AR(2) walk → spline-resample to ndis0.

        Returns 1 on success, 0 on failure (walk too short or spline error).
        """
        if x0 is None:
            x0_, y0_, chazi_, chsinu_ = self._sample_streamline()
            if x0 is None: x0 = x0_
            if y0 is None: y0 = y0_
            if chazi is None: chazi = chazi_
            if chsinu is None: chsinu = chsinu_
        cx0, cy0 = self._ar2_walk(x0, y0, chazi, chsinu, self.ndis_cap)
        if cx0 is None or cx0.size < 20:
            return 0

        # Natural cubic spline + resample to ndis0 uniformly along arc length
        length = np.zeros(cx0.size)
        length[1:] = np.sqrt(np.diff(cx0) ** 2 + np.diff(cy0) ** 2)
        length = np.cumsum(length)
        try:
            sx = CubicSpline(length, cx0, bc_type='natural')
            sy = CubicSpline(length, cy0, bc_type='natural')
        except Exception:
            return 0

        L = np.linspace(0.0, length[-1], self.ndis0)
        self.cx = sx(L)
        self.cy = sy(L)
        self.ndis = self.cx.size
        # Per-node CHelev — uniform at chelev (Alluvsim chamat[:,11] is set
        # constant per streamline by ``avulsioninside.for:149`` /
        # ``lookupstream.for``).
        self.chelev_arr = np.full(self.ndis, self.chelev, dtype=np.float64)
        # Reset per-streamline width cache so onedrf is regenerated
        self._chwidth_arr = None
        self._chwidth_state_n = -1
        # Store sampled azimuth/sinuosity for later avulsion-inside
        self._chazi = float(chazi)
        self._chsinu = float(chsinu)
        return 1

    # ----------------------------------------------------------------- streamline pool (buildCHtable)

    def _build_streamline_pool(self):
        """Pre-build ``CHndraw`` candidate streamlines (Alluvsim ``buildCHtable.for``).

        Each candidate has its own (chazi, chsinu, y0) draw, AR(2) walk, and
        per-node onedrf width. Sampling at avulsion-outside is uniform over
        the pool (no horimat trend; Alluvsim's ``chadrawwt`` collapses to
        uniform when ``horifl`` is unset, which is the default for our use
        case — see tracker §3.3).
        """
        pool = []
        for _ in range(self.CHndraw):
            x0, y0, chazi, chsinu = self._sample_streamline()
            cx, cy = self._ar2_walk(x0, y0, chazi, chsinu, self.ndis_cap)
            if cx is None or cx.size < 20:
                continue
            length = np.zeros(cx.size)
            length[1:] = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
            length = np.cumsum(length)
            try:
                sx = CubicSpline(length, cx, bc_type='natural')
                sy = CubicSpline(length, cy, bc_type='natural')
            except Exception:
                continue
            L = np.linspace(0.0, length[-1], self.ndis0)
            cx_r = sx(L)
            cy_r = sy(L)
            # Per-node halfwidth via onedrf (used in ``lookupstream``)
            chdepth = max(0.0, float(np.random.normal(self.mCHdepth, self.stdevCHdepth)))
            chwdratio = max(0.0, float(np.random.normal(self.mCHwdratio, self.stdevCHwdratio)))
            half_arr = self._onedrf(cx_r.size, chdepth, self.stdevCHdepth2) * chwdratio * 0.5
            half_arr = np.clip(half_arr, 0.01, None)
            pool.append({
                'cx': cx_r, 'cy': cy_r,
                'chazi': chazi, 'chsinu': chsinu,
                'chdepth': chdepth, 'chwdratio': chwdratio,
                'chwidth_arr': half_arr,
                'src': self._last_src,
            })
        self._pool = pool

    def _draw_from_pool(self) -> int:
        """Sample one streamline from the pre-built pool (uniform weights).

        Mirrors ``probdraw + lookupstream`` in ``streamsim.for:727-728``.
        With several sources the draws rotate through them (the k-th channel
        drawn enters at source k mod N), so every source is represented as
        soon as a level has drawn N channels. Returns 1 on success.
        """
        if not self._pool:
            self._build_streamline_pool()
            if not self._pool:
                return 0
        cands = []
        if self._sources:
            src = self._draw_counter % len(self._sources)
            self._draw_counter += 1
            cands = [i for i, c in enumerate(self._pool) if c.get('src') == src]
        idx = int(cands[int(np.random.randint(0, len(cands)))]) if cands else int(np.random.randint(0, len(self._pool)))
        c = self._pool[idx]
        self.cx = c['cx'].copy()
        self.cy = c['cy'].copy()
        self.ndis = self.cx.size
        self.CHdepth = float(c['chdepth'])
        self.CHwdratio = float(c['chwdratio'])
        self.gr_dwratio = 2.0 / max(self.CHwdratio, 1e-6)
        self.CHhalfwidth = 0.5 * self.CHdepth * self.CHwdratio
        self._chazi = float(c['chazi'])
        self._chsinu = float(c['chsinu'])
        self._chwidth_arr = c['chwidth_arr'].copy()
        self._chwidth_state_n = self.cx.size
        self.chelev_arr = np.full(self.ndis, self.chelev, dtype=np.float64)
        return 1

    # ----------------------------------------------------------------- per-node onedrf width

    def _onedrf(self, n: int, tmean: float, tstdev: float) -> np.ndarray:
        """1D Gaussian RF with triangular covariance (port of ``onedrf.for``).

        Returns N(tmean, tstdev) array of length n with correlation range
        ``self.nCHcor`` cells.
        """
        L = max(int(self.nCHcor), 1)
        white = np.random.normal(0.0, 1.0, n + 2 * L)
        # Triangular kernel: w[0]=1/L, w[i]=(L-i)/L^2 for |i|<=L
        kernel = np.array([(L - abs(i)) / float(L * L) for i in range(-L, L + 1)],
                          dtype=np.float64)
        smoothed = np.convolve(white, kernel, mode='same')[L:L + n]
        var = float(smoothed.var())
        std = np.sqrt(max(var, 0.001))
        return (smoothed - smoothed.mean()) * (max(tstdev, 0.0) / std) + tmean

    def _refresh_chwidth(self):
        """Regenerate the per-node halfwidth via onedrf if needed.

        Mirrors Alluvsim ``buildCHtable.for:140-159`` /
        ``avulsioninside.for:212-225``: onedrf produces per-node depth
        (mean=CHdepth, stdev=stdevCHdepth2), then halfwidth = depth*CHwdratio*0.5.
        """
        if (self._chwidth_arr is not None
                and self._chwidth_state_n == self.cx.size
                and self._chwidth_arr.size == self.cx.size):
            return
        depth_arr = self._onedrf(self.cx.size, self.CHdepth, self.stdevCHdepth2)
        half_arr = depth_arr * self.CHwdratio * 0.5
        self._chwidth_arr = np.clip(half_arr, 0.01, None)
        self._chwidth_state_n = self.cx.size
        self.maxCHhalfwidth = float(self._chwidth_arr.max())

    # ----------------------------------------------------------------- curvature pipeline

    def cal_curv(self):
        """Build splines and compute per-node (azi, curv, dcsids, thalweg).

        Direct port of Alluvsim ``curvature2.for``:

        1. Per-segment azimuth via ``azimuth(x1,x2,y1,y2)`` (compass deg).
        2. ``movwinsmooth(spline_i, nwin=10)`` — smooth azimuth.
        3. Curvature ``c = dazi/ds`` (with 360° wrap fix).
        4. ``movwinsmooth(spline_c, nwin=10)`` — smooth curvature.
        5. ``d = dc/ds`` (finite difference) → ``movwinsmooth(spline_d, nwin=10)``.
        6. Single global ``maxcurve = max|c|`` → thalweg = 0.5 ± 0.25|c|/maxcurve.
        7. Resample (cx, cy, w, t, c, d, i, z, ds) to ``ndis0`` uniform spacing.
        """
        if self.cx is None or self.cx.size < 3:
            return
        n0 = self.cx.size
        dl = np.zeros(n0)
        dl[1:] = np.sqrt(np.diff(self.cx) ** 2 + np.diff(self.cy) ** 2)
        s_seg = np.cumsum(dl)
        # Per-segment compass azimuth
        azi = np.zeros(n0)
        for i in range(1, n0):
            di = self.cx[i] - self.cx[i - 1]
            dj = self.cy[i] - self.cy[i - 1]
            if di == 0.0:
                azi[i] = 0.0 if dj > 0 else 180.0
            elif di > 0.0 and dj >= 0.0:
                azi[i] = 90.0 - np.degrees(np.arctan(dj / di))
            elif di < 0.0:
                azi[i] = 270.0 - np.degrees(np.arctan(dj / di))
            else:  # di > 0, dj < 0
                azi[i] = 90.0 - np.degrees(np.arctan(dj / di))
        azi[0] = azi[1]
        azi = _movwinsmooth(azi, 10)

        # Curvature with 360° wrap correction
        c = np.zeros(n0)
        for i in range(1, n0):
            ds = s_seg[i] - s_seg[i - 1]
            a1 = azi[i - 1]
            a2 = azi[i]
            d1 = a2 - a1
            d2 = a2 - (a1 + 360.0)
            dazi = d1 if abs(d1) < abs(d2) else d2
            c[i] = dazi / max(ds, 1e-9)
        c[0] = c[1]
        c = _movwinsmooth(c, 10)

        # dCsi/ds
        d = np.zeros(n0)
        for i in range(1, n0):
            ds = s_seg[i] - s_seg[i - 1]
            d[i] = (c[i] - c[i - 1]) / max(ds, 1e-9)
        d[0] = d[1]
        d = _movwinsmooth(d, 10)

        # Thalweg with single global max (matches AL)
        maxcurve = float(np.abs(c).max()) + 1e-9
        thalweg = np.where(
            c < 0.0,
            0.5 - 0.25 * np.abs(c) / maxcurve,
            0.5 + 0.25 * np.abs(c) / maxcurve,
        )

        # Per-node halfwidth
        self._refresh_chwidth()
        w = self._chwidth_arr
        if w is None or w.size != n0:
            w = self.CHhalfwidth * np.ones(n0)
        z = self.chelev_arr if (self.chelev_arr is not None
                                 and self.chelev_arr.size == n0) else np.full(n0, self.chelev)

        # Spline-resample to ndis0 uniform
        try:
            sx = CubicSpline(s_seg, self.cx, bc_type='natural')
            sy = CubicSpline(s_seg, self.cy, bc_type='natural')
            sw = CubicSpline(s_seg, w, bc_type='natural')
            st = CubicSpline(s_seg, thalweg, bc_type='natural')
            sc = CubicSpline(s_seg, c, bc_type='natural')
            sd = CubicSpline(s_seg, d, bc_type='natural')
            si = CubicSpline(s_seg, azi, bc_type='natural')
            sz = CubicSpline(s_seg, z, bc_type='natural')
        except Exception:
            return

        L = np.linspace(0.0, s_seg[-1], self.ndis0)
        self.length = L
        self.cx = sx(L)
        self.cy = sy(L)
        self.chwidth = np.clip(sw(L), 0.01, None)
        self.thalweg = np.clip(st(L), 1e-3, 1.0 - 1e-3)
        self.curv = sc(L)
        self.dcsids = sd(L)
        self.azi = si(L)
        self.chelev_arr = sz(L)
        # Per-node ds for the migration integral
        ds = np.zeros(self.ndis0)
        ds[1:] = np.diff(L)
        ds[0] = ds[1] if ds.size > 1 else self.step
        self.dlength = ds
        self.ndis = self.cx.size
        # Per-node halfwidth becomes the resampled view; cache for stamps
        self._chwidth_arr = self.chwidth
        self._chwidth_state_n = self.ndis
        self.maxCHhalfwidth = float(self.chwidth.max())
        # Tangent vectors (for downstream cross-product side tests)
        self.vx = np.cos(np.radians(450.0 - self.azi))
        self.vy = np.sin(np.radians(450.0 - self.azi))

    # ----------------------------------------------------------------- migration

    def _migrate_one_step(self, distMigrate: float) -> int:
        """One bank-retreat step (port of ``calcusb.for`` + ``migrate.for``).

        ``calcusb.for`` form (Sun 1996 eq. 15):
          ``part1 = -us0*Csi``
          ``part2 = mCHhalfwidth*Cf/us0``
          ``part3 = us0^4 / (g*h0^2)``
          ``part4 = (scour_factor+2)*us0^2/h0``
          ``inte = sum_{j=idis}^{idis-30} exp(-2 Cf ds_cum / h0) * Csi_j``
          ``usbmat[idis] = part1 + part2*(part3+part4)*inte``

        Then ``migrate.for``: ``ang = chamat[idis,9] + 90``,
        ``x' = x + dist*sind(ang); y' = y + dist*cosd(ang)``. Equivalent to
        offsetting each node by ``dist`` perpendicular to its tangent.
        """
        if self.cx is None or self.cx.size < 20:
            return 0
        self.cal_curv()
        n = self.ndis
        ds = self.dlength
        c = self.curv
        # Pre-sum ds backward for the 30-node integral with exp decay
        usb = np.zeros(n)
        mCHhalfwidth_eff = self.CHhalfwidth
        part2 = mCHhalfwidth_eff * self.Cf / self.us0
        part3 = self.us0 ** 4 / (self.g * self.h0 ** 2)
        part4 = (self.A + 2.0) * self.us0 ** 2 / self.h0
        for idis in range(1, n):
            start = max(0, idis - 30)
            ds_cum = 0.0
            inte = 0.0
            for j in range(idis, start - 1, -1):
                ds_cum += ds[j]
                inte += np.exp(-2.0 * self.Cf * ds_cum / self.h0) * c[j]
            usb[idis] = -self.us0 * c[idis] + part2 * (part3 + part4) * inte
        # Rescale to peak distMigrate
        max_abs = float(np.max(np.abs(usb)))
        if max_abs <= 1e-9:
            return 0
        usb *= distMigrate / max_abs
        # Perpendicular offset. AL ``migrate.for:93-97`` literally writes
        # ``ang = chamat[idis,9] + 90`` (compass-right of motion direction)
        # but with c>0 = compass-CW = right-turn, the cutbank is on the
        # LEFT of motion (outer bend), so AL's literal +90 sends the channel
        # toward the inner bank and the streamline straightens instead of
        # amplifying bends. We use ``azi - 90`` (compass-LEFT of motion) so
        # positive usb at a c>0 apex moves the channel toward the cutbank
        # and bends grow per Sun 1996. Verified against Pyrcz 2003 Fig 4.
        ang_perp = self.azi - 90.0
        nx_perp = np.sin(np.radians(ang_perp))
        ny_perp = np.cos(np.radians(ang_perp))
        # Pin proximal endpoint (idx 0); migrate idx 1..n-1
        self.cx[1:] = self.cx[1:] + usb[1:] * nx_perp[1:]
        self.cy[1:] = self.cy[1:] + usb[1:] * ny_perp[1:]

        # Geometric neckcutoff (Alluvsim ``neckcutoff.for``).
        #
        # Save the full pre-cutoff centerline (and aligned per-node arrays)
        # before make_cutoff compacts cx/cy in place — when
        # ``mNeckFFCHprop > 0`` we use them to paint the dropped oxbow
        # loop as an FFCH mud plug ("neck cutoff → oxbow lake → mud
        # plug" sequence visible in cross-section).
        thresh = self.maxCHhalfwidth * 3.0
        n_pre = self.cx.size
        cx_pre = self.cx.copy() if self.mNeckFFCHprop > 0.0 else None
        cy_pre = self.cy.copy() if self.mNeckFFCHprop > 0.0 else None
        vx_pre = self.vx.copy() if (self.mNeckFFCHprop > 0.0 and self.vx is not None) else None
        vy_pre = self.vy.copy() if (self.mNeckFFCHprop > 0.0 and self.vy is not None) else None
        thalweg_pre = (self.thalweg.copy()
                        if (self.mNeckFFCHprop > 0.0 and self.thalweg is not None
                            and self.thalweg.size == n_pre) else None)
        chwidth_pre = (self._chwidth_arr.copy()
                        if (self.mNeckFFCHprop > 0.0 and self._chwidth_arr is not None
                            and self._chwidth_arr.size == n_pre) else None)
        chelev_pre = (self.chelev_arr.copy()
                       if (self.mNeckFFCHprop > 0.0 and self.chelev_arr is not None
                           and self.chelev_arr.size == n_pre) else None)
        idx_map = np.arange(n_pre, dtype=np.int64)
        new_n = make_cutoff(self.cx, self.cy, self.dlength, thresh, idx_map)
        if (self.mNeckFFCHprop > 0.0 and new_n < n_pre
                and cx_pre is not None and vx_pre is not None
                and thalweg_pre is not None and chwidth_pre is not None
                and chelev_pre is not None):
            self._stamp_neck_oxbows(
                idx_map[:new_n], n_pre,
                cx_pre, cy_pre, vx_pre, vy_pre,
                thalweg_pre, chwidth_pre, chelev_pre,
                self.mNeckFFCHprop,
            )
        if new_n < 20:
            return 0
        self.cx = self.cx[:new_n].copy()
        self.cy = self.cy[:new_n].copy()
        # Always recompute curvature after migration (item 1.11)
        self.ndis = self.cx.size
        # Width array stays per-node — re-trim to match new node count
        if self._chwidth_arr is not None and self._chwidth_arr.size > self.ndis:
            self._chwidth_arr = self._chwidth_arr[:self.ndis].copy()
            self._chwidth_state_n = self.ndis
        if self.chelev_arr is not None and self.chelev_arr.size > self.ndis:
            self.chelev_arr = self.chelev_arr[:self.ndis].copy()
        self.cal_curv()
        return 1

    # ----------------------------------------------------------------- avulsion

    def _avulse_inside(self):
        """Port of ``avulsioninside.for`` — curvature-weighted node, fresh AR(2) tail."""
        if self.cx is None or self.cx.size < 20:
            return False
        self.cal_curv()
        n = self.ndis
        # Pick avulsion node weighted by |curv| over [1, n-2] (AL excludes only 1 endpoint each side)
        weights = np.abs(self.curv).copy()
        if n > 2:
            weights[0] = 0.0
            weights[-1] = 0.0
        # Trunk protection (DeltaLayer): zero weights on the proximal
        # ``min_avul_node_frac`` of nodes so the trunk meanders as a single
        # channel before any avulsion-inside splice can fire — the new
        # tail is always grafted onto the distal portion.
        if self.min_avul_node_frac > 0.0 and n > 2:
            n_protect = max(1, int(self.min_avul_node_frac * n))
            n_protect = min(n_protect, n - 2)
            weights[:n_protect] = 0.0
        if weights.sum() <= 1e-12:
            lo = max(1, int(self.min_avul_node_frac * n))
            ianode = int(np.random.randint(lo, n - 1)) if lo < n - 1 else n - 2
        else:
            p = weights / weights.sum()
            ianode = int(np.random.choice(n, p=p))
        # Anchor the new tail to the regional trunk azimuth ``mCHazi``,
        # NOT to the local segment direction at the avulsion node.
        # Geologically: avulsion picks a new path driven by the regional
        # gradient, not by where the old channel happened to be pointing
        # locally. Using ``self.azi[ianode]`` (Alluvsim default) caused
        # accumulating drift across many avulsions — late branches
        # would launch sideways or even backward to flow, especially
        # in delta runs where ``probAvulInside`` is high (~0.7) and
        # 1000+ avulsions stack across generations. Anchoring to
        # ``mCHazi`` keeps every new branch aligned with the regional
        # gradient; ``branch_spread_deg`` (DeltaLayer) still adds
        # controlled angular spread on top.
        local_azi = float(self.mCHazi)
        if self.stdev_branch_azi > 0.0:
            local_azi += float(np.random.normal(0.0, self.stdev_branch_azi))
        chsinu = float(getattr(self, '_chsinu', self.mCHsinu))

        # AR(2) tail from (cx[ianode], cy[ianode]) toward local_azi
        cx_tail, cy_tail = self._ar2_walk(
            float(self.cx[ianode]), float(self.cy[ianode]),
            chazi=local_azi, chsinu=chsinu,
            ndis_max=self.ndis_cap,
        )
        if cx_tail is None or cx_tail.size < 5:
            return False
        # Splice (drop the duplicate first node of the tail)
        cx_new = np.concatenate([self.cx[:ianode + 1], cx_tail[1:]])
        cy_new = np.concatenate([self.cy[:ianode + 1], cy_tail[1:]])
        if cx_new.size < 20:
            return False
        # ``cororigin``-style proximal smoothing of the splice (item 3.9)
        nsiz = min(int(self.nCHcor), max(cx_new.size // 8, 3))
        i0 = max(ianode - nsiz, 1)
        i1 = min(ianode + nsiz + 1, cx_new.size)
        if i1 - i0 >= 3:
            half_w = max(nsiz // 2, 1)
            sm_x = cx_new[i0:i1].copy()
            sm_y = cy_new[i0:i1].copy()
            for j in range(i0, i1):
                lo = max(j - half_w, 0)
                hi = min(j + half_w + 1, cx_new.size)
                sm_x[j - i0] = cx_new[lo:hi].mean()
                sm_y[j - i0] = cy_new[lo:hi].mean()
            cx_new[i0:i1] = sm_x
            cy_new[i0:i1] = sm_y

        # Resample to ndis0 (NO 4× upsample — item 2.9)
        length = np.zeros(cx_new.size)
        length[1:] = np.sqrt(np.diff(cx_new) ** 2 + np.diff(cy_new) ** 2)
        length = np.cumsum(length)
        if length[-1] < 1e-3:
            return False
        try:
            sx = CubicSpline(length, cx_new, bc_type='natural')
            sy = CubicSpline(length, cy_new, bc_type='natural')
        except Exception:
            return False
        L = np.linspace(0.0, length[-1], self.ndis0)
        self.cx = sx(L)
        self.cy = sy(L)
        self.ndis = self.cx.size
        self.chelev_arr = np.full(self.ndis, self.chelev, dtype=np.float64)

        # Width splice with delta_width offset (item 2.27)
        self._refresh_chwidth_after_splice(ianode)
        return True

    def _refresh_chwidth_after_splice(self, ianode_old: int):
        """Regenerate per-node width via onedrf, splice continuously at ianode.

        Port of ``avulsioninside.for:212-225`` — keeps width continuous at
        the avulsion node (``delta_width = chamat[ianode,5,old] - new[ianode]``).
        """
        # We have replaced the streamline; regenerate full width array.
        depth_arr = self._onedrf(self.ndis, self.CHdepth, self.stdevCHdepth2)
        half_arr = np.clip(depth_arr * self.CHwdratio * 0.5, 0.01, None)
        # Find the splice node in the resampled coord (~ ianode/n_old fraction)
        if self._chwidth_arr is not None and self._chwidth_arr.size > 0:
            # Approximate: use the closest fractional index
            ia = min(max(ianode_old, 0), self._chwidth_arr.size - 1)
            old_w = float(self._chwidth_arr[ia])
            ia_new = min(max(int(round(ia / max(self._chwidth_arr.size - 1, 1)
                                          * (self.ndis - 1))), 0), self.ndis - 1)
            delta_w = old_w - float(half_arr[ia_new])
            half_arr[ia_new + 1:] = np.clip(half_arr[ia_new + 1:] + delta_w, 0.01, None)
            half_arr[:ia_new + 1] = self._chwidth_arr[:ia_new + 1] if ia_new + 1 <= self._chwidth_arr.size \
                                    else half_arr[:ia_new + 1]
        self._chwidth_arr = half_arr
        self._chwidth_state_n = self.ndis
        self.maxCHhalfwidth = float(half_arr.max())

    # ----------------------------------------------------------------- coordinate rotation (back-compat)

    def _rot_xy(self, x, y):
        """Map one walk coordinate into the frame the channel is stamped in.

        Streamlines are walked axis-aligned and rotated by ``azimuth`` about
        the grid centre at stamping time (see ``_rotated_stream``), so a point
        is inside the model only if its ROTATED image is inside the grid.
        """
        if self.azimuth_rad == 0.0:
            return x, y
        dx = x - self._pivot_x
        dy = y - self._pivot_y
        return (self._pivot_x + dx * self._cos_az + dy * self._sin_az,
                self._pivot_y - dx * self._sin_az + dy * self._cos_az)

    def _rotated_stream(self):
        if self.azimuth_rad == 0.0:
            return self.cx, self.cy, self.vx, self.vy
        c, s = self._cos_az, self._sin_az
        px, py = self._pivot_x, self._pivot_y
        dx = self.cx - px
        dy = self.cy - py
        cx_r = px + dx * c + dy * s
        cy_r = py - dx * s + dy * c
        vx_r = self.vx * c + self.vy * s
        vy_r = -self.vx * s + self.vy * c
        return cx_r, cy_r, vx_r, vy_r

    # ----------------------------------------------------------------- per-event geometry redraw

    def _redraw_event_geometry(self):
        """Per-event Gaussian draws for CHdepth, CHwdratio.

        Mirrors ``streamsim.for:831-840``. Width-array regenerated via onedrf.
        """
        self.CHdepth = _gauss_clip(self.mCHdepth, self.stdevCHdepth, lo=1e-3)
        self.CHwdratio = _gauss_clip(self.mCHwdratio, self.stdevCHwdratio, lo=1e-3)
        self.CHhalfwidth = 0.5 * self.CHdepth * self.CHwdratio
        self.gr_dwratio = 2.0 / max(self.CHwdratio, 1e-6)
        self._chwidth_arr = None
        self._chwidth_state_n = -1

    # ----------------------------------------------------------------- stamps

    def _draw_event_mults(self) -> tuple[float, float]:
        """Kozeny-Carman-coupled per-event ``(poro_mult, log_perm_offset)``.

        Replicates the coupling that ``LobeLayer`` uses (see
        ``resmill/layers/lobe.py``): higher poro → higher perm in the
        same event, with a small independent scatter so the relationship
        isn't perfectly 1:1. Slope = ``log_perm_offset_std / poro_mult_std``
        keeps the marginal stds at the requested values while binding
        the two together within a single event.

        Used in three places:

        * ``_stamp_channel`` — draws and caches the pair for the current
          channel event (reused by levee/splay stamps that follow).
        * ``_stamp_abandoned`` — draws fresh (later, separate event).
        * ``_stamp_neck_oxbows`` — draws fresh per dropped loop.

        Clipping caps the multipliers at ±2σ.
        """
        pm = float(np.random.normal(1.0, self.poro_mult_std))
        # ±2σ clip on poro_mult
        pm_lo = max(0.05, 1.0 - 2.0 * self.poro_mult_std)
        pm_hi = 1.0 + 2.0 * self.poro_mult_std
        if pm < pm_lo:
            pm = pm_lo
        elif pm > pm_hi:
            pm = pm_hi
        # Kozeny-Carman: log_perm tracks (poro_mult - 1) with the same
        # std ratio lobes use, plus a small independent scatter (lobes
        # add N(0, 0.05) on log10 perm; we match).
        slope = self.log_perm_offset_std / max(self.poro_mult_std, 1e-6)
        scatter = float(np.random.normal(0.0, 0.05))
        po = slope * (pm - 1.0) + scatter
        # ±2σ clip on the result so extreme combinations stay physical.
        po_lo = -2.0 * self.log_perm_offset_std
        po_hi = +2.0 * self.log_perm_offset_std
        if po < po_lo:
            po = po_lo
        elif po > po_hi:
            po = po_hi
        return pm, po

    def _stamp_channel(self, facies_code: int, erode_above: bool):
        if self.cx is None or self.cx.size < 3:
            return
        if (self.vx is None or self.vx.size != self.cx.size
                or self.curv is None or self.curv.size != self.cx.size
                or self.thalweg is None or self.thalweg.size != self.cx.size):
            self.cal_curv()
        self._refresh_chwidth()
        cx_r, cy_r, vx_r, vy_r = self._rotated_stream()
        chwidth_arr = self._chwidth_arr
        chelev_arr = self.chelev_arr if (self.chelev_arr is not None
                                          and self.chelev_arr.size == self.cx.size) \
                     else np.full(self.cx.size, self.chelev, dtype=np.float64)
        # Draw the K-C-coupled (poro_mult, log_perm_offset) pair for
        # this channel event and cache it on the engine so the
        # subsequent _stamp_levee / _stamp_splays calls inherit the
        # same depositional regime.
        ev_pm, ev_po = self._draw_event_mults()
        self._event_poro_mult = ev_pm
        self._event_log_perm_offset = ev_po
        genchannel(
            float(self.maxCHhalfwidth), self.xsiz, self.ysiz, chelev_arr, self.zsiz,
            self.nx, self.ny, self.nz, cx_r, cy_r, self.x, self.y,
            vx_r, vy_r, self.curv, self.LV_asym, self.lV_height,
            self.ps, self.pavul, self.out, self.totalid, self.facies,
            self.poro, self.poro0, self.thalweg, chwidth_arr, self.gr_dwratio,
            [1_000_000_000], 800,
            xmn=self.xmn, ymn=self.ymn,
            merge_overlap=False,
            facies_code=int(facies_code),
            ntg_counter=self.ntg_counter,
            compute_poro=False,
            erode_above=bool(erode_above),
            depth_norm=self.depth_norm,
            poro_mult_field=self.poro_mult_field,
            log_perm_offset_field=self.log_perm_offset_field,
            ev_poro_mult=ev_pm, ev_log_perm_offset=ev_po,
        )

    def _stamp_splays(self, n_splay: int, n_lobe_per_splay: int):
        """Place ``n_splay`` crevasse-splay clusters along the streamline.

        Each splay is a curvature-weighted CSnode; for each lobe the splay
        is a ``gensplay``-style random walker (``onedrf`` perturbed
        azimuth, std=40°) of length ``CSLOLL ± 10%``, painting the lobe
        envelope on the cutbank side.
        """
        if n_splay <= 0 or n_lobe_per_splay <= 0:
            return
        if self.cx is None or self.cx.size < 5:
            return
        if (self.vx is None or self.vx.size != self.cx.size
                or self.curv is None or self.curv.size != self.cx.size
                or self.azi is None):
            self.cal_curv()
        self._refresh_chwidth()
        cx_r, cy_r, vx_r, vy_r = self._rotated_stream()
        weights = np.abs(self.curv)
        if weights.sum() <= 1e-12:
            weights = np.ones_like(weights)
        p = weights / weights.sum()

        for _ in range(int(n_splay)):
            cs_node = int(np.random.choice(len(p), p=p))
            curv_at = float(self.curv[cs_node])
            cs_azi = float(self.azi[cs_node])
            # Cutbank rule (streamsim.for:907-911): curv > 0 → CSazi -= 90
            if curv_at > 0.0:
                cs_azi = cs_azi - 90.0
            else:
                cs_azi = cs_azi + 90.0
            for _l in range(int(n_lobe_per_splay)):
                cs_LL = _gauss_clip(self.mCSLOLL, self.stdevCSLOLL, lo=1e-3)
                cs_WW = _gauss_clip(self.mCSLOWW, self.stdevCSLOWW, lo=1e-3)
                cs_l = _gauss_clip(self.mCSLOl, self.stdevCSLOl, lo=1e-3, hi=cs_LL)
                cs_w = _gauss_clip(self.mCSLOw, self.stdevCSLOw, lo=1e-3)
                cs_hw = _gauss_clip(self.mCSLO_hwratio, self.stdevCSLO_hwratio, lo=1e-3)
                cs_dw = _gauss_clip(self.mCSLO_dwratio, self.stdevCSLO_dwratio, lo=1e-3)
                # Generate the splay random-walk centerline (gensplay.for)
                cx_lobe, cy_lobe = self._build_splay_walker(
                    float(cx_r[cs_node]), float(cy_r[cs_node]),
                    cs_azi, cs_LL,
                )
                if cx_lobe is None or cx_lobe.size < 3:
                    continue
                # Reuse the host channel event's K-C pair — splays form
                # contemporaneously with the channel that fed them, so
                # they inherit the same poro/perm regime.
                ev_pm = self._event_poro_mult
                ev_po = self._event_log_perm_offset
                # Paint lobe envelope along this walker
                paint_lobe(
                    cx_lobe, cy_lobe,
                    cs_LL, cs_WW, cs_l, cs_w, cs_hw, cs_dw,
                    self.chelev,
                    self.x, self.y,
                    self.nx, self.ny, self.nz,
                    self.xsiz, self.ysiz, self.zsiz,
                    self.facies, self.ntg_counter, lk_cs=CS,
                    xmn=self.xmn, ymn=self.ymn,
                    depth_norm=self.depth_norm,
                    poro_mult_field=self.poro_mult_field,
                    log_perm_offset_field=self.log_perm_offset_field,
                    ev_poro_mult=ev_pm, ev_log_perm_offset=ev_po,
                )
                # Also paint the thin gensplay sheet (`facies = 5` → CS) at
                # iz_chelev-1 with linear taper (item 1.10/2.26)
                paint_splay(
                    cx_lobe, cy_lobe, self.chelev, self.CHdepth,
                    self.x, self.y,
                    self.nx, self.ny, self.nz,
                    self.xsiz, self.ysiz, self.zsiz,
                    self.facies, self.ntg_counter,
                    xmn=self.xmn, ymn=self.ymn, lk_cs=CS,
                    depth_norm=self.depth_norm,
                    poro_mult_field=self.poro_mult_field,
                    log_perm_offset_field=self.log_perm_offset_field,
                    ev_poro_mult=ev_pm, ev_log_perm_offset=ev_po,
                )

    def _build_splay_walker(self, x0: float, y0: float, azi0: float, dist: float):
        """One splay random-walk center (port of ``gensplay.for:78-115``).

        ``onedrf(l = max(1, nst/5), nst, angle, azi0, 40.0)`` then walk
        ``x = x + step*cosd(angle); y = y + step*sind(angle)`` until the
        walker leaves the grid or completes ``nst`` steps.
        Length jitter ±10% via ``dist0 = (p-0.5)*dist*0.2 + dist``.
        """
        st = (self.xsiz + self.ysiz) / 2.0
        p_jitter = float(np.random.uniform())
        dist0 = (p_jitter - 0.5) * dist * 0.2 + dist
        nst = max(int(dist0 / st), 5)
        L = max(1, nst // 5)
        # onedrf around mean azi0, stdev 40°
        depth_arr = self._onedrf_with_correlation(nst, azi0, 40.0, L)
        cx = np.empty(nst, dtype=np.float64)
        cy = np.empty(nst, dtype=np.float64)
        x = x0
        y = y0
        n_ok = 0
        for ist in range(nst):
            ang = depth_arr[ist]
            # AL: x = x + step*cosd(ang); y = y + step*sind(ang) (math angle)
            ang_math = 450.0 - ang  # convert compass → math
            x2 = x + st * np.cos(np.radians(ang_math))
            y2 = y + st * np.sin(np.radians(ang_math))
            if x2 < self.xmin or x2 > self.xmax or y2 < self.ymin or y2 > self.ymax:
                break
            cx[ist] = x2
            cy[ist] = y2
            x = x2
            y = y2
            n_ok = ist + 1
        if n_ok < 3:
            return None, None
        return cx[:n_ok], cy[:n_ok]

    def _onedrf_with_correlation(self, n: int, tmean: float, tstdev: float, L: int) -> np.ndarray:
        """``onedrf`` variant with explicit correlation length L (used by gensplay)."""
        L = max(int(L), 1)
        white = np.random.normal(0.0, 1.0, n + 2 * L)
        kernel = np.array([(L - abs(i)) / float(L * L) for i in range(-L, L + 1)],
                          dtype=np.float64)
        smoothed = np.convolve(white, kernel, mode='same')[L:L + n]
        var = float(smoothed.var())
        std = np.sqrt(max(var, 0.001))
        return (smoothed - smoothed.mean()) * (max(tstdev, 0.0) / std) + tmean

    def _stamp_levee(self, LV_depth: float, LV_width: float, LV_height: float,
                     LV_asym: float, LV_thin: float):
        if self.cx is None or self.cx.size < 3:
            return
        if LV_width <= 0.0 or (LV_height + LV_depth) <= 0.0:
            return
        if (self.vx is None or self.vx.size != self.cx.size
                or self.curv is None or self.curv.size != self.cx.size):
            self.cal_curv()
        self._refresh_chwidth()
        cx_r, cy_r, _vx, _vy = self._rotated_stream()
        chwidth_arr = self._chwidth_arr
        chelev_arr = self.chelev_arr if (self.chelev_arr is not None
                                          and self.chelev_arr.size == self.cx.size) \
                     else np.full(self.cx.size, self.chelev, dtype=np.float64)
        # Reuse the host channel event's K-C pair — natural levees
        # form contemporaneously with the channel they border.
        ev_pm = self._event_poro_mult
        ev_po = self._event_log_perm_offset
        paint_levee(
            cx_r, cy_r, self.curv, chwidth_arr, chelev_arr,
            float(LV_depth), float(LV_width), float(LV_height), float(LV_asym), float(LV_thin),
            self.x, self.y, self.xsiz, self.ysiz, self.zsiz,
            self.nx, self.ny, self.nz, self.facies, self.ntg_counter,
            float(self.maxCHhalfwidth),
            xmn=self.xmn, ymn=self.ymn, lk_lv=LV,
            depth_norm=self.depth_norm,
            poro_mult_field=self.poro_mult_field,
            log_perm_offset_field=self.log_perm_offset_field,
            ev_poro_mult=ev_pm, ev_log_perm_offset=ev_po,
        )

    def _stamp_neck_oxbows(self, surviving_idx, n_pre,
                            cx_pre, cy_pre, vx_pre, vy_pre,
                            thalweg_pre, chwidth_pre, chelev_pre,
                            mud_prop):
        """Paint each dropped neck-cutoff oxbow loop as an FFCH mud plug.

        ``surviving_idx`` is the slice of the post-cutoff ``idx_map``
        (original indices that survived). Any contiguous run of original
        indices missing from that set is one oxbow loop — we rotate
        (delta-mode) and slice the saved per-node arrays for that loop
        and call ``paint_abandoned`` with ``mud_prop = mNeckFFCHprop``.
        """
        survive_set = set(int(i) for i in surviving_idx.tolist())
        dropped_ranges = []
        i = 0
        while i < n_pre:
            if i not in survive_set:
                start = i
                while i < n_pre and i not in survive_set:
                    i += 1
                dropped_ranges.append((start, i - 1))
            else:
                i += 1
        if not dropped_ranges:
            return

        if self.azimuth_rad == 0.0:
            cx_rot, cy_rot, vx_rot, vy_rot = cx_pre, cy_pre, vx_pre, vy_pre
        else:
            c, s = self._cos_az, self._sin_az
            px, py = self._pivot_x, self._pivot_y
            dx = cx_pre - px
            dy = cy_pre - py
            cx_rot = px + dx * c + dy * s
            cy_rot = py - dx * s + dy * c
            vx_rot = vx_pre * c + vy_pre * s
            vy_rot = -vx_pre * s + vy_pre * c

        for (lo, hi) in dropped_ranges:
            # Pad by one node on each side so the loop's pinch points
            # (the surviving idis and jdis+1) join smoothly with the
            # new active centerline rather than leaving a 1-node gap.
            s_idx = max(0, lo - 1)
            e_idx = min(n_pre, hi + 2)
            cx_l = cx_rot[s_idx:e_idx].copy()
            cy_l = cy_rot[s_idx:e_idx].copy()
            vx_l = vx_rot[s_idx:e_idx].copy()
            vy_l = vy_rot[s_idx:e_idx].copy()
            thalweg_l = thalweg_pre[s_idx:e_idx].copy()
            chwidth_l = chwidth_pre[s_idx:e_idx].copy()
            chelev_l = chelev_pre[s_idx:e_idx].copy()
            if cx_l.size < 3:
                continue
            # Neck-cutoff oxbow plug is a separate later event from the
            # original channel deposition — fresh K-C draw.
            ev_pm, ev_po = self._draw_event_mults()
            paint_abandoned(
                float(chwidth_l.max()), cx_l, cy_l, vx_l, vy_l, thalweg_l,
                chwidth_l, chelev_l, self.gr_dwratio, mud_prop,
                self.x, self.y, self.xsiz, self.ysiz, self.zsiz,
                self.nx, self.ny, self.nz, self.facies,
                self.ntg_counter, self.ffch_counter,
                xmn=self.xmn, ymn=self.ymn, lk_ffch=FFCH, lk_ch=CH,
                depth_norm=self.depth_norm,
                poro_mult_field=self.poro_mult_field,
                log_perm_offset_field=self.log_perm_offset_field,
                ev_poro_mult=ev_pm, ev_log_perm_offset=ev_po,
            )

    def _stamp_abandoned(self, mud_prop: float):
        if self.cx is None or self.cx.size < 3:
            return
        if (self.vx is None or self.vx.size != self.cx.size
                or self.curv is None or self.curv.size != self.cx.size):
            self.cal_curv()
        self._refresh_chwidth()
        cx_r, cy_r, vx_r, vy_r = self._rotated_stream()
        chwidth_arr = self._chwidth_arr
        chelev_arr = self.chelev_arr if (self.chelev_arr is not None
                                          and self.chelev_arr.size == self.cx.size) \
                     else np.full(self.cx.size, self.chelev, dtype=np.float64)
        # End-of-level abandonment is a later event (channel was
        # abandoned, mud filled in) — fresh K-C draw, distinct from the
        # original channel-deposition regime.
        ev_pm, ev_po = self._draw_event_mults()
        paint_abandoned(
            float(self.maxCHhalfwidth), cx_r, cy_r, vx_r, vy_r, self.thalweg,
            chwidth_arr, chelev_arr, self.gr_dwratio, mud_prop,
            self.x, self.y, self.xsiz, self.ysiz, self.zsiz,
            self.nx, self.ny, self.nz, self.facies,
            self.ntg_counter, self.ffch_counter,
            xmn=self.xmn, ymn=self.ymn, lk_ffch=FFCH, lk_ch=CH,
            depth_norm=self.depth_norm,
            poro_mult_field=self.poro_mult_field,
            log_perm_offset_field=self.log_perm_offset_field,
            ev_poro_mult=ev_pm, ev_log_perm_offset=ev_po,
        )

    # ----------------------------------------------------------------- main event loop

    def simulation(self):
        """Per-level event loop — port of ``streamsim.for:737-1004``."""
        # Per-level entry-x offset (used for delta-style progradation).
        # When ``mCHentry_x_offset_per_level`` is None the offset stays
        # at 0 throughout; otherwise we apply level 0's offset before
        # building the pool so the proximal trunk position matches.
        if self.mCHentry_x_offset_per_level is not None:
            self._entry_x_offset = float(self.mCHentry_x_offset_per_level[0])
        if self.n_sources > 1:
            self._draw_sources()
        # Build the streamline pool once at sim start (Alluvsim:702 buildCHtable)
        self._build_streamline_pool()

        self.chelev = self.level_z[0]
        # Initial streamline at level 0 (Alluvsim:727-731)
        if not self._draw_from_pool():
            return
        self.cal_curv()
        self._is_first_streamline = True

        level_target = self._level_targets()
        ev_counter = 0
        last_ffchprop = 0.0

        for ilevel in range(self.nlevel):
            self.chelev = float(self.level_z[ilevel])
            self.chelev_arr = np.full(self.ndis, self.chelev, dtype=np.float64)
            # When ntime is per-level, reset the event counter so each level
            # gets its own full ``ntime`` budget instead of being starved by
            # whatever the lower levels consumed.
            if self.ntime_per_level:
                ev_counter = 0
            # Snapshot the global ntg / ffch counters at the start of this
            # level so the inner ``while`` loop can stop when *this level*
            # has deposited its independent NTG share — regardless of how
            # much lower levels overshot. Without this delta-tracking the
            # cumulative target would let lower-level overshoot satisfy
            # upper levels and produce bottom-heavy cubes at low NTG.
            ntg_at_level_start = int(self.ntg_counter[0])
            ffch_at_level_start = int(self.ffch_counter[0])
            # Progradation: shift entry x and rebuild the pool so all
            # subsequent draws start at the new apex position.
            if (self.mCHentry_x_offset_per_level is not None
                    and ilevel > 0):
                new_offset = float(self.mCHentry_x_offset_per_level[ilevel])
                if new_offset != self._entry_x_offset:
                    self._entry_x_offset = new_offset
                    self._build_streamline_pool()
            # Always reseed at level top (item 2.20 — matches AL:727-731)
            if ilevel > 0:
                if not self._draw_from_pool():
                    return
                self.cal_curv()

            if self.bifurcate:
                # Delta distributary trees: ``n_trees`` networks per level
                # (the level's avulsion history), each from its own trunk;
                # no event loop, no end-of-level abandonment.
                for itree in range(self.n_trees):
                    if itree > 0:
                        if not self._draw_from_pool():
                            break
                        self.cal_curv()
                    self._simulate_tree(old=itree < self.n_trees - 1)
                continue

            while ((self.ntg_counter[0] - ntg_at_level_start)
                   - (self.ffch_counter[0] - ffch_at_level_start)) < level_target[ilevel]:
                if ev_counter >= self.ntime:
                    # ntime cap exit (AL:988-991 — no extra abandon; the
                    # end-of-level path below handles abandonment).
                    break
                ev_counter += 1
                self.totalid = ev_counter

                ffchprop = _gauss_clip(self.mFFCHprop, self.stdevFFCHprop, lo=0.0, hi=1.0)
                last_ffchprop = ffchprop
                p = float(np.random.uniform())

                if p < self.probAvulOutside:
                    # Avulsion outside (AL:757-774)
                    if not self._is_first_streamline:
                        self._stamp_abandoned(ffchprop)
                    if not self._draw_from_pool():
                        continue
                    self._redraw_event_geometry()
                    self.cal_curv()
                    self._is_first_streamline = False
                elif p < (self.probAvulOutside + self.probAvulInside):
                    # Avulsion inside (AL:782-796)
                    self._stamp_abandoned(ffchprop)
                    if not self._avulse_inside():
                        continue
                    self._redraw_event_geometry()
                    self.cal_curv()
                    self._is_first_streamline = False
                else:
                    # Migration (AL:804-820): LA on OLD path, then redraw,
                    # migrate, neckcutoff, then CH on NEW path.
                    self._stamp_channel(facies_code=LA, erode_above=False)
                    self._redraw_event_geometry()
                    distMigrate = _gauss_clip(self.mdistMigrate, self.stdevdistMigrate, lo=0.0)
                    if self._migrate_one_step(distMigrate) == 0:
                        if not self._draw_from_pool():
                            return
                        self.cal_curv()
                    self._is_first_streamline = False

                # Per-event CS draws + placement (AL:889-945)
                cs_num = int(round(_gauss_clip(self.mCSnum, self.stdevCSnum, lo=0.0)))
                cs_numlobe = int(round(_gauss_clip(self.mCSnumlobe, self.stdevCSnumlobe, lo=0.0)))
                if cs_num > 0 and cs_numlobe > 0:
                    self._stamp_splays(cs_num, cs_numlobe)

                # Per-event LV draws (AL:842-865) and placement
                lv_depth = _gauss_clip(self.mLVdepth, self.stdevLVdepth, lo=0.0)
                lv_width = _gauss_clip(self.mLVwidth, self.stdevLVwidth, lo=0.0)
                lv_height = _gauss_clip(self.mLVheight, self.stdevLVheight, lo=0.0)
                lv_asym = _gauss_clip(self.mLVasym, self.stdevLVasym, lo=0.0)
                lv_thin = _gauss_clip(self.mLVthin, self.stdevLVthin, lo=0.0)
                self._stamp_levee(lv_depth, lv_width, lv_height, lv_asym, lv_thin)

                # Active-channel stamp (AL:966-967)
                self._stamp_channel(facies_code=CH, erode_above=True)

            # End-of-level abandonment (AL:1002-1004)
            self._stamp_abandoned(last_ffchprop)
            # Record the distal tip of the active streamline at level
            # close (x, y, chelev, heading). DeltaLayer optionally
            # paints a calc_lobe mouth bar at every recorded tip.
            if self.cx is not None and self.cx.size > 1:
                head = float(np.arctan2(self.cy[-1] - self.cy[-2],
                                         self.cx[-1] - self.cx[-2]))
                self.distal_tips.append((
                    float(self.cx[-1]), float(self.cy[-1]),
                    float(self.chelev), head,
                ))
            # Stop the whole simulation only when ntime is global (Alluvsim
            # default). With ``ntime_per_level=True`` each level gets a
            # fresh budget and we always continue to the next level.
            if not self.ntime_per_level and ev_counter >= self.ntime:
                return

    # ----------------------------------------------------------------- distributary tree

    def _simulate_tree(self, old: bool = False):
        """Grow and stamp one distributary network for the current level.

        ``old``: an earlier network of the level, abandoned when the next one
        formed; its channels are stamped with the level's mud-fill fraction
        (``mFFCHprop``) like any abandoned channel.

        Starts from the trunk just drawn from the pool, carrying discharge
        ``Q = 1``. Each of ``n_bifurcations`` splits picks a branch with
        probability proportional to its discharge (big channels split
        more often), a node beyond the branch's protected reach, and a
        share ``f ~ U(split_frac_lo, split_frac_hi)`` that leaves the
        parent: the parent keeps ``(1 - f) Q`` downstream of the node and
        a new walk carrying ``f Q`` departs at ``branch_angle`` from the
        local heading (never more than 80 degrees off the regional flow).
        Width follows ``Q ** width_exp`` and depth ``Q ** depth_exp``, so
        every order of the tree is narrower and shallower than its parent;
        a branch runs at most ``branch_length_scale * diagonal * sqrt(Q)``
        steps and then ends on the plain, where its tip is recorded for a
        mouth bar; a walk that runs into another branch (``merge_branches``)
        joins it, and the receiving branch carries the sum downstream of the
        junction. Branches are stamped once each, largest discharge first,
        with the level's levees. Splits land anywhere on the network, so
        the result is a tree rather than a fan of equal ribbons.
        """
        if self.cx is None or self.cx.size < 20:
            return
        base_depth, base_wd = float(self.CHdepth), float(self.CHwdratio)
        chsinu = float(getattr(self, '_chsinu', self.mCHsinu))
        diag = float(np.hypot(self.xmax - self.xmin, self.ymax - self.ymin))
        w0 = base_depth * base_wd                         # trunk full width

        def half_of(q):
            return 0.5 * w0 * q ** self.width_exp

        # the delta front: a circle about the apex; a branch reaching it ends
        # there in a mouth bar
        apex = (float(self.cx[0]), float(self.cy[0]))
        R_front = self.front_radius * (self.xmax - self.xmin)

        def front(q):                                      # where this branch's lobe ends
            jit = float(np.exp(np.random.normal(0.0, self.front_jitter))) if self.front_jitter > 0 else 1.0
            return R_front * (1.0 + self.front_bulge * np.sqrt(q)) * jit

        tcx, tcy, at_front = self._cut_at_front(self.cx.copy(), self.cy.copy(), apex, front(1.0))
        n = tcx.size
        if n < 20:
            return
        branches = [dict(cx=tcx, cy=tcy, q=1.0, order=0, splits=0,
                         protect=max(1, int(self.min_avul_node_frac * n)), tip=at_front, merged=False)]
        bars = []                                          # (x, y, heading, width): a bar at every bifurcation
        reg = np.radians(450.0 - self.mCHazi)             # regional flow, walk frame, math radians
        lim = np.radians(80.0)
        for _ in range(self.n_bifurcations):
            # a segment gives off at most ``max_splits_per_branch`` branches, so
            # the tree cascades (trunk, then its children, then theirs) instead
            # of every branch leaving the trunk
            cand = [b for b in branches
                    if b['q'] >= 2.0 * self.q_min and b['cx'].size - b['protect'] > 12
                    and b['splits'] < self.max_splits_per_branch]
            if not cand:
                break
            wq = np.array([b['q'] for b in cand])
            parent = cand[int(np.random.choice(len(cand), p=wq / wq.sum()))]
            pcx, pcy = parent['cx'], parent['cy']
            span = pcx.size - 6 - parent['protect']
            k = parent['protect'] + int(span * np.random.uniform() ** self.split_pos_exp)
            k = int(np.clip(k, parent['protect'], pcx.size - 7))
            f = float(np.random.uniform(self.split_frac_lo, self.split_frac_hi))
            q_child, q_down = f * parent['q'], (1.0 - f) * parent['q']
            head = float(np.arctan2(pcy[k + 1] - pcy[k - 1], pcx[k + 1] - pcx[k - 1]))
            theta = np.radians(float(np.clip(np.random.normal(self.branch_angle_mean,
                                                              max(self.branch_angle_sd, 1e-9)), 15.0, 90.0)))
            side = 1.0 if np.random.uniform() < 0.5 else -1.0
            child_math = head + side * theta
            off = (child_math - reg + np.pi) % (2.0 * np.pi) - np.pi
            child_math = reg + float(np.clip(off, -lim, lim))
            launch = (450.0 - np.degrees(child_math)) % 360.0
            # mean heading between the launch angle and the regional flow:
            # the branch leaves at the full angle and eases back
            mean_math = reg + (1.0 - self.branch_relax) * ((child_math - reg + np.pi) % (2.0 * np.pi) - np.pi)
            chazi_child = (450.0 - np.degrees(mean_math)) % 360.0
            terminal = q_child < self.q_min
            cap = int(max(12, min(self.ndis_cap, self.branch_length_scale * diag / self.step * np.sqrt(q_child)))) \
                if terminal else int(self.ndis_cap)
            cx_t, cy_t = self._ar2_walk(float(pcx[k]), float(pcy[k]), chazi_child, chsinu, cap, azi0=launch)
            if cx_t is None or cx_t.size < 8:
                continue
            ended_on_plain = terminal and cx_t.size >= cap
            cx_t, cy_t, reached = self._cut_at_front(cx_t, cy_t, apex, front(q_child))
            if cx_t.size < 8:
                continue
            ended_on_plain = ended_on_plain or reached
            hit = None
            if self.merge_branches:
                hit = self._first_contact(branches, parent, cx_t, cy_t, half_of(q_child), half_of)
                if hit is not None:
                    j, recv, jn = hit
                    if j < 8:
                        continue
                    cx_t, cy_t = cx_t[:j + 1], cy_t[:j + 1]
                    ended_on_plain = False
            # parent keeps Q upstream of the node, (1 - f) Q downstream of it
            up = dict(parent, cx=pcx[:k + 1].copy(), cy=pcy[:k + 1].copy(), tip=False,
                      splits=parent['splits'] + 1)
            down = dict(cx=pcx[k:].copy(), cy=pcy[k:].copy(), q=q_down, order=parent['order'],
                        protect=6, tip=parent['tip'], merged=parent['merged'], splits=0)
            if self.y_split and hit is None and not self._reach_has_branches(branches, parent, pcx, pcy, k):
                # a Y: the surviving reach bends away from the new branch by
                # the share the branch took, so neither side runs straight on
                d_math = head - side * theta * f
                d_off = (d_math - reg + np.pi) % (2.0 * np.pi) - np.pi
                d_math = reg + float(np.clip(d_off, -lim, lim))
                d_launch = (450.0 - np.degrees(d_math)) % 360.0
                d_mean = reg + (1.0 - self.branch_relax) * ((d_math - reg + np.pi) % (2.0 * np.pi) - np.pi)
                d_cap = int(self.ndis_cap)
                dcx, dcy = self._ar2_walk(float(pcx[k]), float(pcy[k]), (450.0 - np.degrees(d_mean)) % 360.0,
                                          chsinu, d_cap, azi0=d_launch)
                if dcx is not None and dcx.size >= 8:
                    dcx, dcy, d_front = self._cut_at_front(dcx, dcy, apex, front(q_down))
                    if dcx.size >= 8:
                        down = dict(cx=dcx, cy=dcy, q=q_down, order=parent['order'], protect=6,
                                    tip=bool(d_front), merged=False, splits=0)
            # the bar that split the channel sits between the two branches: about
            # half the parent's width, so record it as a tip of that width
            bars.append((float(pcx[k]), float(pcy[k]), head, 1.0 * half_of(parent['q'])))
            branches[self._index_of(branches, parent)] = up
            branches.append(down)
            branches.append(dict(cx=cx_t, cy=cy_t, q=q_child, order=parent['order'] + 1, splits=0,
                                 protect=6, tip=bool(ended_on_plain), merged=hit is not None))
            if hit is not None:
                j, recv, jn = hit
                if recv is parent:
                    recv = down if jn >= k else up
                    jn = jn - k if jn >= k else jn
                if 1 <= jn < recv['cx'].size - 1:
                    r_up = dict(recv, cx=recv['cx'][:jn + 1].copy(), cy=recv['cy'][:jn + 1].copy(), tip=False)
                    r_down = dict(recv, cx=recv['cx'][jn:].copy(), cy=recv['cy'][jn:].copy(),
                                  q=recv['q'] + q_child, protect=6, splits=0)
                    branches[self._index_of(branches, recv)] = r_up
                    branches.append(r_down)

        # mouth-bar complex: a channel that reached its front splits around its
        # bar into two short terminal channels, each ending in a smaller bar
        fringe = []
        for b in branches:
            if not b['tip'] or b['cx'].size < 4 or 2.0 * half_of(b['q']) < 2.0 * self.step:
                continue
            ex, ey = float(b['cx'][-1]), float(b['cy'][-1])
            h = float(np.arctan2(b['cy'][-1] - b['cy'][-3], b['cx'][-1] - b['cx'][-3]))
            for side in (-1.0, 1.0):
                th = np.radians(float(np.clip(np.random.normal(35.0, 10.0), 15.0, 60.0)))
                m = h + side * th
                q_t = 0.5 * b['q']
                cap = int(max(6, 3.0 * 2.0 * half_of(b['q']) / self.step))
                tcx2, tcy2 = self._ar2_walk(ex, ey, (450.0 - np.degrees(m)) % 360.0, chsinu, cap)
                if tcx2 is not None and tcx2.size >= 4:
                    fringe.append(dict(cx=tcx2, cy=tcy2, q=q_t, order=b['order'] + 1, splits=self.max_splits_per_branch,
                                       protect=0, tip=True, merged=False))
        branches.extend(fringe)
        self.tree_branches.extend(dict(order=b['order'], q=b['q'], n=int(b['cx'].size),
                                       tip=b['tip'], merged=b['merged']) for b in branches)
        for b in sorted(branches, key=lambda b: -b['q']):
            if b['cx'].size < 4:
                continue
            self._set_branch(b, base_depth, base_wd)
            self.totalid += 1
            if old:
                self._stamp_abandoned(_gauss_clip(self.mFFCHprop, self.stdevFFCHprop, lo=0.0, hi=1.0))
            else:
                self._stamp_channel(facies_code=CH, erode_above=False)
            lv_depth = _gauss_clip(self.mLVdepth, self.stdevLVdepth, lo=0.0)
            lv_width = _gauss_clip(self.mLVwidth, self.stdevLVwidth, lo=0.0)
            lv_height = _gauss_clip(self.mLVheight, self.stdevLVheight, lo=0.0)
            lv_asym = _gauss_clip(self.mLVasym, self.stdevLVasym, lo=0.0)
            lv_thin = _gauss_clip(self.mLVthin, self.stdevLVthin, lo=0.0)
            self._stamp_levee(lv_depth, lv_width, lv_height, lv_asym, lv_thin)
            if b['tip'] and self.cx.size > 1:
                dx, dy = self.cx[-1] - self.cx[-2], self.cy[-1] - self.cy[-2]
                tx, ty = self._rot_xy(float(self.cx[-1]), float(self.cy[-1]))
                hx = dx * self._cos_az + dy * self._sin_az
                hy = -dx * self._sin_az + dy * self._cos_az
                tip_half = float(self._chwidth_arr[-1]) if self._chwidth_arr is not None else self.CHhalfwidth
                self.distal_tips.append((float(tx), float(ty), float(self.chelev), float(np.arctan2(hy, hx)),
                                         2.0 * tip_half))
        for bx, by, bhead, bw in bars:
            tx, ty = self._rot_xy(bx, by)
            hx = np.cos(bhead) * self._cos_az + np.sin(bhead) * self._sin_az
            hy = -np.cos(bhead) * self._sin_az + np.sin(bhead) * self._cos_az
            self.distal_tips.append((float(tx), float(ty), float(self.chelev), float(np.arctan2(hy, hx)), float(bw)))

    @staticmethod
    def _index_of(branches, b):
        return next(i for i, x in enumerate(branches) if x is b)

    @staticmethod
    def _cut_at_front(cx, cy, apex, R):
        """Truncate a walk at the delta front (distance R from the apex).
        Returns (cx, cy, reached)."""
        r = np.hypot(cx - apex[0], cy - apex[1])
        beyond = np.flatnonzero(r >= R)
        if beyond.size == 0:
            return cx, cy, False
        j = int(beyond[0])
        return cx[:max(j, 2)], cy[:max(j, 2)], True

    @staticmethod
    def _reach_has_branches(branches, parent, pcx, pcy, k):
        """True when another branch starts on the parent's nodes beyond k."""
        starts = np.array([(b['cx'][0], b['cy'][0]) for b in branches if b is not parent])
        if starts.size == 0 or pcx.size - k - 1 <= 0:
            return False
        reach = np.column_stack([pcx[k + 1:], pcy[k + 1:]])
        d = np.sqrt(((starts[:, None, :] - reach[None, :, :]) ** 2).sum(-1))
        return bool((d < 1e-6).any())

    def _first_contact(self, branches, parent, cx_t, cy_t, half_child, half_of):
        """First node of a new walk that touches another branch.

        Returns ``(j, branch, node)`` for the earliest walk node ``j >= 6``
        closer to a node of a branch other than its parent than the two
        half-widths, or None.
        """
        pts, owner = [], []
        for b in branches:
            if b is parent:
                continue
            pts.append(np.column_stack([b['cx'], b['cy']]))
            owner.extend((b, i) for i in range(b['cx'].size))
        if not pts:
            return None
        pts = np.vstack(pts)
        tree = cKDTree(pts)
        q = np.column_stack([cx_t[6:], cy_t[6:]])
        d, idx = tree.query(q, k=1)
        for jj in range(q.shape[0]):
            b, i = owner[int(idx[jj])]
            if d[jj] < half_child + half_of(b['q']):
                return jj + 6, b, i
        return None

    def _set_branch(self, b, base_depth, base_wd):
        """Make one tree branch the engine's current streamline."""
        self.cx = np.asarray(b['cx'], dtype=np.float64).copy()
        self.cy = np.asarray(b['cy'], dtype=np.float64).copy()
        self.ndis = self.cx.size
        q = float(b['q'])
        self.CHdepth = base_depth * q ** self.depth_exp
        self.CHwdratio = base_wd * q ** (self.width_exp - self.depth_exp)
        self.CHhalfwidth = 0.5 * self.CHdepth * self.CHwdratio
        self.gr_dwratio = 2.0 / max(self.CHwdratio, 1e-6)
        self._chwidth_arr = None
        self._chwidth_state_n = -1
        self.chelev_arr = np.full(self.ndis, self.chelev, dtype=np.float64)
        self.vx = self.vy = self.curv = self.thalweg = None
        self.cal_curv()
        if b['order'] > 0 and self.branch_taper < 1.0 and self._chwidth_arr is not None:
            # linear taper from the split to the end of the branch
            ramp = np.linspace(1.0, self.branch_taper, self._chwidth_arr.size)
            self._chwidth_arr = np.clip(self._chwidth_arr * ramp, 0.01, None)
            self.maxCHhalfwidth = float(self._chwidth_arr.max())

    # ----------------------------------------------------------------- helpers

    def _level_targets(self) -> np.ndarray:
        """Per-level NTG-cell target — *independent* across levels.

        Each level is responsible for depositing
        ``NTGtarget × cells_per_level`` sand cells in its own z-slice;
        the simulation loop tracks that as a delta against an
        ``ntg_at_level_start`` snapshot, so a level's event count is
        independent of how much lower levels deposited. This gives
        full-z population at any NTGtarget — without the original
        Pyrcz-Alluvsim cumulative-target artefact where lower levels
        could "skip" upper levels when they overshot the cumulative
        target.

        The returned array is the *delta target per level* (each
        element is the same value when nlevel divides evenly).
        """
        total_cells = self.nx * self.ny * self.nz
        NTGcount = int(round(self.NTGtarget * total_cells))
        per_level = max(1, int(round(NTGcount / max(self.nlevel, 1))))
        return np.full(self.nlevel, per_level, dtype=np.int64)
