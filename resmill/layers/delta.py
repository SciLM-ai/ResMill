"""Delta layer — distributary-fan architecture (Alluvsim fluvial engine).

:py:class:`DeltaLayer` is a thin subclass of
:py:class:`~resmill.layers.channel.ChannelLayer` that drives
the full Alluvsim event-loop fluvial engine (AR(2) walks +
Sun-1996 bank-retreat migration + avulsion-inside + neck cutoff + level
aggradation) with delta-tuned defaults (:data:`DELTA_FAN` preset).

Why the fan emerges naturally
-----------------------------
The fluvial engine builds a streamline pool in which every candidate
AR(2) walk starts at ``(xmin + entry_x_offset, y0)`` with
``y0 ~ N(mCHsource, stdevCHsource)``. With ``stdevCHsource ≈ 0`` every
streamline shares the same proximal apex pixel and AR(2) noise fans
the trajectories out downstream.

Each avulsion-inside event splices a fresh AR(2) tail at a curvature-
weighted node (downstream of ``trunk_length_fraction`` if set) and
leaves the old downstream tail abandoned. With ``mFFCHprop = 0`` that
abandoned tail stays as residual CH (sand), so each event leaves one
permanent visible distributary in the cube. After ``ntime_per_gen``
events the accumulated tails draw out a bifurcating distributary
network at a single chelev. ``n_generations`` independent runs at
stacked chelev fill the column vertically (aggradation). Optional
``progradation_fraction`` advances the apex along +flow per generation
to build a clinoform.
"""
from __future__ import annotations

import numpy as np

from .channel import ChannelLayer


__all__ = ["DeltaLayer", "DELTA_FAN"]


# ---------------------------------------------------------------------------
# DELTA_FAN preset — defaults tuned for "trunk meanders before
# bifurcating into a clinoform-stacked fan with full vertical
# aggradation".
#
# The DeltaLayer drives ``n_generations`` separate single-level engine
# runs (rather than one multi-level run) so each generation gets a
# guaranteed event budget regardless of the global NTG target. This
# gives explicit control over both aggradation (n_generations + z
# stacking) and progradation (apex advance per generation), and avoids
# the NTG-cap problem where the engine exhausts its ntime budget on
# level 0 and never reaches upper levels.
# ---------------------------------------------------------------------------
DELTA_FAN = dict(
    # ---- aggradation: stacked generations (single-level each) -----
    # 8 generations × 4-m channels = 32 m of vertical coverage on the
    # standard 32-m grid, no gaps. Channels are 4 cells thick on the
    # default ``dz=1 m`` so distributary ribbons read clearly in XZ/YZ.
    n_generations=8,
    ntime_per_gen=80,         # ≈ 56 avulsion-inside + ≈ 24 migration / gen
    NTGtarget=0.99,           # never NTG-cap inside a generation
    # ---- avulsion: heavy avulsion-inside, no avulsion-outside ------
    probAvulOutside=0.0,
    probAvulInside=0.7,
    # ---- channel geometry: nearly straight, narrow distributaries -
    mCHsinu=1.10, stdevCHsinu=0.02,
    mCHwdratio=14.0, stdevCHwdratio=2.0,
    mCHdepth=4.0, stdevCHdepth=0.5,
    # ---- single-feeder source: tight, dead-centre ------------------
    stdevCHsource=0.2,
    mCHazi=90.0, stdevCHazi=0.2,
    # ---- migration small (we want avulsion, not bend amplification)
    mdistMigrate=3.0, stdevdistMigrate=1.0,
    # ---- abandoned tails stay as sand → visible distributaries -----
    mFFCHprop=0.0, stdevFFCHprop=0.0,
    # ---- levees on (delta-distributary natural levees are common) -
    mLVdepth=0.5, stdevLVdepth=0.1,
    mLVwidth=20.0, stdevLVwidth=5.0,
    mLVheight=0.3, stdevLVheight=0.1,
    mLVasym=0.0, stdevLVasym=0.1,
    # ---- crevasse splays disabled by default; user can enable -----
    mCSnum=0.0, stdevCSnum=0.0,
    # ---- hydraulic — Alluvsim makepar central values --------------
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
    # ---- distributary tree (off by default; the published dataset ---
    # was made without it). With ``bifurcate=True`` a generation is one
    # branching network instead of an avulsion history: discharge is
    # split at every bifurcation, width goes as Q**width_exp and depth
    # as Q**depth_exp, branches end on the plain or rejoin. See
    # ``_fluvial._simulate_tree``.
    bifurcate=False,
    n_trees=1,
    n_bifurcations=8,
    split_frac_lo=0.25, split_frac_hi=0.5,
    branch_angle_mean=55.0, branch_angle_sd=15.0,
    q_min=0.05,
    branch_length_scale=0.6,
    max_splits_per_branch=2,
    split_pos_exp=1.0,
    y_split=True,
    branch_relax=0.5,
    branch_taper=0.7,
    merge_branches=True,
    width_exp=0.5, depth_exp=0.4,
)


def _paint_mouth_bar_into_engine(engine, tip_x, tip_y, tip_z, heading,
                                  LL, WW, hw_ratio, dw_ratio,
                                  facies_code=3):
    """Paint a calc_lobe mouth-bar envelope at a distributary terminus.

    Direct port of Alluvsim ``calc_lobe.for`` lines 190-219:
    parabolic-proximal / elliptical-distal envelope. Writes into the
    engine's facies array using an Alluvsim facies code (default
    LA = 3 — lateral accretion / bar deposit; sand in binary mode,
    with realistic poro / perm via ``FACIES_PROPS``).

    Cells already CH (active channel, code 4) are not overwritten —
    the bar onlaps the channel mouth rather than replacing it.
    """
    cos_h, sin_h = float(np.cos(heading)), float(np.sin(heading))
    s_l = LL / 3.0
    pad = max(LL, WW)
    nx_, ny_, nz_ = engine.facies.shape
    xsiz, ysiz, zsiz = engine.xsiz, engine.ysiz, engine.zsiz
    x_grid, y_grid = engine.x, engine.y

    xmin_ = tip_x - pad; xmax_ = tip_x + pad
    ymin_ = tip_y - pad; ymax_ = tip_y + pad
    ix0 = max(0, int((xmin_ - x_grid[0]) / xsiz))
    ix1 = min(nx_, int((xmax_ - x_grid[0]) / xsiz) + 1)
    iy0 = max(0, int((ymin_ - y_grid[0]) / ysiz))
    iy1 = min(ny_, int((ymax_ - y_grid[0]) / ysiz) + 1)

    for ix in range(ix0, ix1):
        for iy in range(iy0, iy1):
            dx = x_grid[ix] - tip_x
            dy = y_grid[iy] - tip_y
            s = dx * cos_h + dy * sin_h
            if s < 0.0 or s > LL:
                continue
            d = -dx * sin_h + dy * cos_h
            if s <= s_l:
                # Parabolic proximal ramp: WW/4 at apex → WW at s_l.
                y_fn = WW * (0.25 + 0.75 * (s / s_l) ** 2)
            else:
                u = (s - s_l) / (LL - s_l)
                y_fn = WW * np.sqrt(max(0.0, 1.0 - u * u))
            if y_fn < 1e-3 or abs(d) > y_fn:
                continue
            env = 1.0 - (d / y_fn) ** 2
            top_z = tip_z + y_fn * hw_ratio * env
            bot_z = tip_z - y_fn * dw_ratio * env
            iz_bot = max(0, int(bot_z / zsiz))
            iz_top = min(nz_ - 1, int(top_z / zsiz))
            if iz_top < iz_bot:
                continue
            for iz in range(iz_bot, iz_top + 1):
                if engine.facies[ix, iy, iz] < 4:   # don't overwrite CH
                    engine.facies[ix, iy, iz] = facies_code


class DeltaLayer(ChannelLayer):
    """Distributary-fan delta — Alluvsim-faithful event-loop simulation.

    Subclass of :py:class:`ChannelLayer` that drives the same
    fluvial engine with delta-tuned defaults (:data:`DELTA_FAN`).

    Architecture knobs (delta-only):

    * ``trunk_length_fraction`` — fraction of proximal streamline nodes
      that avulsion-inside is forbidden from picking. Forces the trunk
      to meander as a single channel for the upstream
      ``trunk_length_fraction`` of every streamline before any
      bifurcation can occur. ``0.0`` = bifurcation can happen
      anywhere (single-meandering-channel Alluvsim behaviour).
    * ``progradation_fraction`` — fractional advance of the apex along
      +flow across all aggradation levels. ``0.0`` = no progradation
      (apex pinned at the entry); ``0.4`` = apex advances by 40 % of
      the grid along the flow direction over ``n_generations`` levels →
      classic clinoform.
    * ``branch_spread_deg`` — std-dev (compass °) of a Gaussian
      perturbation added to each new branch's launch azimuth at every
      avulsion-inside splice. ``0.0`` = parent direction (Alluvsim
      default); ``10–15`` = noticeable angular fan-out.
    * ``paint_mouth_bars`` — if ``True``, paint Alluvsim ``calc_lobe``
      envelopes (LA facies, code 3) at every distal streamline tip
      recorded by the engine. One bar per closed level →
      ``n_generations`` bars at the prograding front.

    All other Alluvsim fluvial kwargs are accepted as passthroughs
    (override DELTA_FAN).
    """

    def create_geology(self, *,
                       trunk_length_fraction: float = 0.4,
                       progradation_fraction: float = 0.0,
                       branch_spread_deg: float = 0.0,
                       paint_mouth_bars: bool = False,
                       mouth_bar_length_factor: float = 2.5,
                       mouth_bar_width_factor: float = 1.6,
                       mouth_bar_hw_ratio: float = 0.06,
                       mouth_bar_dw_ratio: float = 0.08,
                       facies_props: dict | None = None,
                       poro_realization_mult: float = 1.0,
                       perm_realization_mult: float = 1.0,
                       seed: int | None = None,
                       **kwargs):
        """Generate a prograding distributary-fan delta.

        Drives ``n_generations`` independent single-level fluvial
        simulations and merges their facies cubes by per-cell max
        (Alluvsim facies codes are ordered FF=-1 < FFCH=0 < CS=1 <
        LV=2 < LA=3 < CH=4, so max preserves the highest-quality
        facies at every cell). This guarantees every generation gets
        the configured event budget regardless of the global NTG
        target.

        Parameters
        ----------
        trunk_length_fraction : float ∈ [0, 0.95]
            Fraction of proximal streamline nodes protected from
            avulsion-inside splicing. The trunk meanders as a single
            channel for that proximal fraction before any
            distributary can branch off. ``0.0`` lets the channel
            bifurcate from the very first node (Alluvsim default —
            i.e. a meandering channel, not a delta).
        progradation_fraction : float ∈ [0, 0.95 - trunk_length_fraction]
            Per-generation **trunk-length advance**. Every streamline
            still starts at the upstream-boundary entry point picked
            by ``azimuth`` (xmin for az=0°, ymax for az=90°, …) and
            walks to the opposite boundary — no channel ever starts
            mid-grid. What changes per generation is the proximal
            node fraction protected from avulsion-inside: generation
            0 uses ``trunk_length_fraction``; the last generation
            uses ``trunk_length_fraction + progradation_fraction``.
            The bifurcation locus (apex of the visible fan) therefore
            advances toward +flow as generations stack vertically —
            classic clinoform with all channels still
            boundary-to-boundary.
        branch_spread_deg : float
            Standard deviation (compass degrees) of the random
            perturbation added to a new branch's launch azimuth at
            every avulsion-inside splice. ``0.0`` (default) ⇒ each
            new tail launches exactly along the parent's local
            direction (so siblings only spread apart through
            subsequent AR(2) wandering). ``>0`` gives direct angular
            fan-out: ``10–15°`` = subtle widening, ``30°+`` = wide
            fan that quickly looks chaotic.
        paint_mouth_bars : bool
            If True, paint a calc_lobe envelope (LA facies, code 3)
            at every distal streamline tip recorded by the engine —
            one bar per generation, at the prograding front.
        mouth_bar_length_factor / mouth_bar_width_factor :
            Bar along-axis length / peak half-width relative to the
            reference channel full-width.
        mouth_bar_hw_ratio / mouth_bar_dw_ratio :
            Dimensionless bar thickness above / depth below the
            channel datum at the bar axis.
        seed : int | None
            Master seed; each generation seeds with ``seed + igen`` so
            generations are independent but reproducible.
        **kwargs
            Override any DELTA_FAN entry (or any fluvial kwarg).
        """
        from ._fluvial import fluvial

        # DELTA_FAN baseline + user overrides
        cfg = dict(DELTA_FAN)
        cfg.update(kwargs)

        # Pull delta-only / user-facing kwargs that fluvial doesn't accept
        n_generations = int(cfg.pop('n_generations', 8))
        ntime_per_gen = int(cfg.pop('ntime_per_gen', 80))
        scour_factor = cfg.pop('scour_factor', 10.0)
        gradient = cfg.pop('gradient', 0.001)

        # Wire direct branch-spread control
        cfg['stdev_branch_azi'] = float(max(branch_spread_deg, 0.0))

        # Per-generation chelev: span the full z column. ``level_z``
        # passed by the user wins; otherwise spread linearly between
        # zsiz and z_len.
        z_len = self.nz * self.dz
        if 'level_z' in cfg and cfg['level_z'] is not None:
            chelev_per_gen = list(cfg.pop('level_z'))
            if len(chelev_per_gen) != n_generations:
                raise ValueError(
                    f"level_z has {len(chelev_per_gen)} entries but "
                    f"n_generations={n_generations}")
        else:
            cfg.pop('level_z', None)
            chelev_per_gen = list(np.linspace(self.dz, z_len, n_generations))

        # Progradation = per-generation **trunk-length** advance. Every
        # streamline still starts at the upstream-boundary entry point
        # picked by ``azimuth`` (xmin for az=0, ymax for az=90, …) and
        # walks to the opposite boundary — no channel ever starts
        # mid-grid. What changes per generation is *how far down the
        # streamline* avulsion-inside is allowed to splice: late
        # generations have a longer protected trunk, so the
        # bifurcation locus (= the apex of the visible fan) advances
        # toward +flow as generations stack vertically. Same clinoform
        # signature as the legacy delta, but the channel is always
        # boundary-to-boundary.
        base_trunk = float(np.clip(trunk_length_fraction, 0.0, 0.95))
        if progradation_fraction > 0.0 and n_generations > 1:
            trunk_per_gen = np.linspace(
                base_trunk,
                float(np.clip(base_trunk + progradation_fraction, 0.0, 0.95)),
                n_generations,
            )
        else:
            trunk_per_gen = np.full(n_generations, base_trunk)

        # Run n_generations independent simulations and merge.
        # Merge rule: per cell, take the generation with the
        # highest-quality facies (CH=4 > LA=3 > LV=2 > CS=1 > FFCH=0
        # > FF=-1). The aux fields (depth_norm, poro_mult, log_perm_offset)
        # also propagate from that winning generation so the upward-fining
        # ramp + per-event noise stay consistent with the visible facies.
        nx_, ny_, nz_ = self.nx, self.ny, self.nz
        accum_facies = np.full((nx_, ny_, nz_), -1, dtype=np.int8)
        accum_depth_norm = np.full((nx_, ny_, nz_), 0.5, dtype=np.float32)
        accum_poro_mult = np.ones((nx_, ny_, nz_), dtype=np.float32)
        accum_log_perm_offset = np.zeros((nx_, ny_, nz_), dtype=np.float32)
        accum_distal_tips: list[tuple[float, float, float, float]] = []
        last_engine = None
        # Every branch of every generation's distributary tree
        # (``bifurcate=True``): order, discharge share, node count, whether
        # it ended on the plain or joined another branch.
        self.tree_branches: list[dict] = []
        for igen in range(n_generations):
            chelev = float(chelev_per_gen[igen])
            gen_seed = None if seed is None else int(seed) + igen
            engine = fluvial(
                nx=nx_, ny=ny_, nz=nz_,
                xsiz=self.dx, ysiz=self.dy, zsiz=self.dz,
                xmn=self.dx / 2, ymn=self.dy / 2,
                nlevel=1, level_z=[chelev], ntime=ntime_per_gen,
                A=scour_factor, I=gradient,
                min_avul_node_frac=float(trunk_per_gen[igen]),
                seed=gen_seed,
                **cfg,
            )
            engine.simulation()
            # Cells where this generation outranks the accumulated facies
            # adopt this generation's aux values; ties keep the prior gen.
            takeover = engine.facies > accum_facies
            accum_facies = np.where(takeover, engine.facies, accum_facies)
            accum_depth_norm = np.where(takeover, engine.depth_norm, accum_depth_norm)
            accum_poro_mult = np.where(takeover, engine.poro_mult_field, accum_poro_mult)
            accum_log_perm_offset = np.where(takeover, engine.log_perm_offset_field,
                                             accum_log_perm_offset)
            accum_distal_tips.extend(engine.distal_tips)
            self.tree_branches.extend(dict(gen=igen, **b) for b in getattr(engine, 'tree_branches', []))
            last_engine = engine

        # Optional mouth-bar painting at every recorded distal tip
        if paint_mouth_bars and accum_distal_tips and last_engine is not None:
            leaf_full_width = last_engine.mCHwdratio * last_engine.mCHdepth
            MB_L = mouth_bar_length_factor * leaf_full_width * 2.0
            MB_W = mouth_bar_width_factor * leaf_full_width
            # Use a thin shim object so _paint_mouth_bar_into_engine sees
            # the same .facies / .x / .y / .xsiz / .ysiz / .zsiz attrs.
            class _Shim: pass
            shim = _Shim()
            shim.facies = accum_facies
            shim.x = last_engine.x; shim.y = last_engine.y
            shim.xsiz = self.dx; shim.ysiz = self.dy; shim.zsiz = self.dz
            for tip in accum_distal_tips:
                tx, ty, tz, head = tip[:4]
                # a tree tip carries its own channel width; size its bar by it
                L_i, W_i = (MB_L, MB_W) if len(tip) < 5 else (
                    mouth_bar_length_factor * float(tip[4]) * 2.0, mouth_bar_width_factor * float(tip[4]))
                _paint_mouth_bar_into_engine(
                    shim, tx, ty, tz, head, L_i, W_i,
                    mouth_bar_hw_ratio, mouth_bar_dw_ratio,
                    facies_code=3,   # LA = lateral-accretion / bar
                )

        self._finalize_facies_table(
            accum_facies, facies_props=facies_props,
            depth_norm=accum_depth_norm,
            poro_mult_field=accum_poro_mult,
            log_perm_offset_field=accum_log_perm_offset,
            poro_realization_mult=poro_realization_mult,
            perm_realization_mult=perm_realization_mult,
        )
        # Expose final engine + accumulated distal tips for tutorial / debug
        self._engine = last_engine
        self._distal_tips = accum_distal_tips
        if last_engine is not None:
            self.poro_mult_std = float(last_engine.poro_mult_std)
            self.log_perm_offset_std = float(last_engine.log_perm_offset_std)
