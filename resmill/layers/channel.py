"""Channel layer classes — Alluvsim-faithful fluvial reservoir generation.

A single engine (``_fluvial.fluvial``) drives all reservoir architectures.
The same parameter space that produces Alluvsim's PV-shoestring,
CB-jigsaw, CB-labyrinth, SH-distal and SH-proximal architectures is exposed
as kwargs on :py:meth:`ChannelLayer.create_geology` —
``ChannelLayer`` is a thin subclass that picks the CB-jigsaw
defaults.

The engine internally tracks all six Alluvsim facies
(FF=-1, FFCH=0, CS=1, LV=2, LA=3, CH=4) and exposes them on
``self.facies`` (always the 6-class array). ``self.active`` is the
binary 0/1 sand mask (``self.facies >= 1``). Per-cell porosity and
permeability are pulled from a small constant table (``FACIES_PROPS``).

Importable parameter presets ``PV_SHOESTRING``, ``CB_JIGSAW``,
``CB_LABYRINTH``, ``SH_DISTAL``, ``SH_PROXIMAL`` mirror Alluvsim's
``runs/run_presets.py`` so users can call::

    layer = ChannelLayer(nx=64, ny=64, nz=32, x_len=640,
                                   y_len=640, z_len=32, top_depth=0.0)
    layer.create_geology(**PV_SHOESTRING)
"""
from __future__ import annotations

import numpy as np

from .base import Layer

__all__ = [
    "ChannelLayer",
    "FACIES_PROPS",
    "PV_SHOESTRING", "CB_JIGSAW", "CB_LABYRINTH", "SH_DISTAL", "SH_PROXIMAL",
    "MEANDER_OXBOW",
]


# ---------------------------------------------------------------------------
# Per-facies poro / log10(perm in mD) lookup table.
#
# Values pick a monotonic CH > LA > LV > CS > FFCH > FF ordering with a
# ~3-decade perm range, typical of fluvial reservoir characterisations.
# Override per-call via the ``facies_props`` kwarg on create_geology.
# ---------------------------------------------------------------------------
FACIES_PROPS: dict[int, dict[str, float]] = {
    -1: {"poro": 0.05, "log10_perm": -1.0},  # FF      overbank fines
     0: {"poro": 0.08, "log10_perm":  0.0},  # FFCH   abandoned mud plug
     1: {"poro": 0.18, "log10_perm":  1.5},  # CS     crevasse splay
     2: {"poro": 0.20, "log10_perm":  2.0},  # LV     levee
     3: {"poro": 0.25, "log10_perm":  2.7},  # LA     lateral accretion
     4: {"poro": 0.30, "log10_perm":  3.3},  # CH     active channel
}


class ChannelLayer(Layer):
    """Fluvial channel-belt layer driving the Alluvsim-port engine.

    One layer class for every channel architecture. Pick the
    architecture via ``preset=`` in ``create_geology`` (PV_SHOESTRING,
    CB_JIGSAW, CB_LABYRINTH, SH_DISTAL, SH_PROXIMAL, MEANDER_OXBOW),
    or override individual fluvial kwargs directly.
    """

    def _finalize_facies_table(self, engine_facies: np.ndarray,
                               facies_props: dict | None = None,
                               depth_norm: np.ndarray | None = None,
                               poro_mult_field: np.ndarray | None = None,
                               log_perm_offset_field: np.ndarray | None = None,
                               poro_realization_mult: float = 1.0,
                               perm_realization_mult: float = 1.0):
        """Build ``self.facies / active / poro_mat / perm_mat`` from engine outputs.

        Inputs:

        * ``engine_facies`` — 6-class facies cube (-1..4).
        * ``depth_norm`` — per-cell, ∈ [0, 1]: 0 at top of channel
          cross-section, 1 at base. Filled with 0.5 (neutral; ramp = 1.0)
          for non-channel facies and inactive cells. Written by the
          fluvial engine PER EVENT (no per-column leakage). When
          ``None`` (e.g. delta merge path), uses 0.5 everywhere.
        * ``poro_mult_field`` — per-cell, defaults to 1.0. Per-event poro
          scalar drawn at each stamp call.
        * ``log_perm_offset_field`` — per-cell, defaults to 0.0. Per-event
          additive offset in log10(perm).
        * ``poro_realization_mult`` — single scalar applied uniformly to
          all cells in the realization (Sobol-controlled "regional rock
          quality"). Default 1.0 (no shift).
        * ``perm_realization_mult`` — single scalar (linear) applied
          uniformly. log10(perm_realization_mult) is added to log_perm
          for every cell. Sobol-sampled log-uniformly, default 1.0.

        Combined formula per cell::

            ramp     = 0.7 + 0.6 × depth_norm                         # [0.7, 1.3]
            poro     = FACIES[f].poro × ramp × poro_mult_field
                       × poro_realization_mult
            log_perm = FACIES[f].log10_perm
                       + KC_SLOPE × log10(ramp)                       # within-event amplified
                       + log_perm_offset_field                        # per-event K-C offset
                       + log10(perm_realization_mult)                 # per-realization shift
            perm     = 10**log_perm

        ``KC_SLOPE = 3.0`` matches the per-event K-C slope (= per-event
        ``log_perm_offset_std / poro_mult_std``), so within-event poro
        and perm vary at the same K-C ratio across the cube.

        FF / FFCH cells are post-clamped to their FACIES_PROPS base
        values (no ramp / no per-event mult / no per-realization mult);
        they are mud, not sand, so within-deposit fining and per-event
        K-C don't apply.
        """
        # K-C slope for within-event ramp (matches per-event slope:
        # log_perm_offset_std / poro_mult_std = 0.12 / 0.04 = 3.0).
        KC_SLOPE = 3.0
        props = dict(FACIES_PROPS)
        if facies_props:
            for k, v in facies_props.items():
                props.setdefault(int(k), {}).update(v)

        self.facies = engine_facies.astype(np.int8)
        self.active = (self.facies >= 1).astype(np.int8)
        nx_, ny_, nz_ = self.facies.shape

        # Aux fields default to neutral when caller omits them
        # (e.g. unit tests that don't go through the full engine).
        if depth_norm is None:
            depth_norm = np.full(self.facies.shape, 0.5, dtype=np.float32)
        if poro_mult_field is None:
            poro_mult_field = np.ones(self.facies.shape, dtype=np.float32)
        if log_perm_offset_field is None:
            log_perm_offset_field = np.zeros(self.facies.shape, dtype=np.float32)

        # Base poro / log_perm by facies code from FACIES_PROPS lookup.
        base_poro = np.zeros(self.facies.shape, dtype=np.float32)
        base_log_perm = np.zeros(self.facies.shape, dtype=np.float32)
        for code, vals in props.items():
            mask = (self.facies == code)
            if mask.any():
                base_poro[mask] = vals["poro"]
                base_log_perm[mask] = vals["log10_perm"]

        # Per-event Walker-1992 upward-fining ramp.
        ramp = (0.7 + 0.6 * depth_norm).astype(np.float32)

        # Compute poro and log_perm across the full cube. Per-realization
        # mults are constants applied uniformly. KC_SLOPE amplifies the
        # within-event ramp on log_perm so it visibly tracks the poro
        # gradient on a log10(perm) colormap.
        poro_realization_mult_f32 = np.float32(poro_realization_mult)
        log_perm_realization_offset_f32 = np.float32(
            np.log10(max(float(perm_realization_mult), 1e-9))
        )
        poro_mat = (base_poro * ramp * poro_mult_field
                    * poro_realization_mult_f32)
        log_perm = (base_log_perm
                    + KC_SLOPE * np.log10(np.maximum(ramp, 1e-6))
                    + log_perm_offset_field
                    + log_perm_realization_offset_f32)

        # Mud cells (FF, FFCH) get base FACIES_PROPS values — no ramp,
        # no per-event mult, no per-realization shift (mud is not the
        # reservoir-quality control variable in this scheme).
        mud_mask = (self.facies == -1) | (self.facies == 0)
        if mud_mask.any():
            poro_mat[mud_mask] = base_poro[mud_mask]
            log_perm[mud_mask] = base_log_perm[mud_mask]

        # Inactive cells (FF=-1) get poro = base FF value; perm tracks.
        # Clip poro to a physical range to avoid float16 overflow / negatives.
        poro_mat = np.clip(poro_mat, 0.0, 0.5)
        perm_mat = (10.0 ** log_perm).astype(np.float32)

        self.poro_mat = poro_mat.astype(np.float32)
        self.perm_mat = perm_mat.astype(np.float32)


    def create_geology(
        self,
        # ---- aggradation -----------------------------------------------
        nlevel: int = 8,
        level_z: list[float] | None = None,
        NTGtarget: float = 0.10,
        ntime: int = 240,
        # ``True`` ⇒ ``ntime`` is interpreted per-level (counter resets
        # at every level); ``False`` ⇒ ``ntime`` is the total event cap
        # across all levels (Alluvsim default).
        ntime_per_level: bool = False,
        # ---- avulsion --------------------------------------------------
        probAvulOutside: float = 0.10, probAvulInside: float = 0.05,
        # ---- channel geometry ------------------------------------------
        mCHdepth: float = 4.0, stdevCHdepth: float = 0.4, stdevCHdepth2: float = 0.3,
        mCHwdratio: float = 10.0, stdevCHwdratio: float = 1.0,
        mCHsinu: float = 1.6, stdevCHsinu: float = 0.15,
        mCHazi: float = 90.0, stdevCHazi: float = 1.0,
        mCHsource: float | None = None, stdevCHsource: float = 80.0,
        n_sources: int = 1, source_spacing_min: float = 0.15,
        # ---- migration --------------------------------------------------
        mdistMigrate: float = 35.0, stdevdistMigrate: float = 10.0,
        # ---- levee (LV) — Alluvsim makepar central values --------------
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
        # ---- abandoned-channel fill (FFCH) -----------------------------
        mFFCHprop: float = 0.0, stdevFFCHprop: float = 0.0,
        # ---- neck-cutoff oxbow → mud plug -----------------------------
        # 0.0 = pure Alluvsim (cutoff is geometric only, abandoned bend
        # keeps prior LA/CH stamps). >0 = each excised oxbow loop is
        # painted as an FFCH mud plug over the upper ``mNeckFFCHprop``
        # of the channel column at that node — gives the
        # neck-cutoff → oxbow lake → mud plug succession in cross-section.
        mNeckFFCHprop: float = 0.0,
        # ---- hydraulic — Alluvsim makepar central values ---------------
        Cf: float = 0.0078, scour_factor: float = 2.0,
        gradient: float = 0.001, Q: float = 5.0,
        # ---- pool / discretisation -------------------------------------
        CHndraw: int = 50, ndiscr: int = 5, nCHcor: int = 10,
        # ---- presentation ----------------------------------------------
        azimuth: float = 0.0,
        facies_props: dict | None = None,
        # ---- per-realization rock-quality (sampler-controlled) ---------
        # Multiplies poro and perm uniformly across the cube — the
        # "regional reservoir quality" knob, analogous to lobes' direct
        # ``poro_ave`` / ``perm_ave`` Sobol sampling. Per-event mults
        # (drawn at each stamp call inside the engine) wiggle on top.
        poro_realization_mult: float = 1.0,
        perm_realization_mult: float = 1.0,
        seed: int | None = None,
    ):
        """Generate channel geology with Alluvsim-faithful semantics.

        All ``mFoo`` / ``stdevFoo`` kwargs are direct ports of Alluvsim's
        per-event Gaussian-draw parameters (``streamsim.par`` field set);
        see ``$HOME/Alluvsim/CLAUDE.md`` §4 for full docs.

        Notable mappings:

        * ``mCHwdratio`` is Alluvsim's full width / depth ratio (so 10
          means a 10:1 W:D channel). ResMill converts internally to
          ``depth / half_width``.
        * Default kwargs reproduce a PV-shoestring-ish reservoir on a
          typical 64-cell grid; override individual params or pass
          ``**PV_SHOESTRING`` / ``**CB_JIGSAW`` / etc. for canonical
          architectures.
        * Output: ``self.facies`` is the full Alluvsim 6-class array
          (-1..4); ``self.active`` is the binary 0/1 sand mask
          (``self.facies >= 1``).
        """
        from ._fluvial import fluvial

        engine = fluvial(
            nx=self.nx, ny=self.ny, nz=self.nz,
            xsiz=self.dx, ysiz=self.dy, zsiz=self.dz,
            xmn=self.dx / 2, ymn=self.dy / 2,
            nlevel=nlevel, level_z=level_z,
            NTGtarget=NTGtarget, ntime=ntime,
            probAvulOutside=probAvulOutside, probAvulInside=probAvulInside,
            mCHdepth=mCHdepth, stdevCHdepth=stdevCHdepth, stdevCHdepth2=stdevCHdepth2,
            mCHwdratio=mCHwdratio, stdevCHwdratio=stdevCHwdratio,
            mCHsinu=mCHsinu, stdevCHsinu=stdevCHsinu,
            mCHazi=mCHazi, stdevCHazi=stdevCHazi,
            mCHsource=mCHsource, stdevCHsource=stdevCHsource,
            n_sources=n_sources, source_spacing_min=source_spacing_min,
            mdistMigrate=mdistMigrate, stdevdistMigrate=stdevdistMigrate,
            mLVdepth=mLVdepth, stdevLVdepth=stdevLVdepth,
            mLVwidth=mLVwidth, stdevLVwidth=stdevLVwidth,
            mLVheight=mLVheight, stdevLVheight=stdevLVheight,
            mLVasym=mLVasym, stdevLVasym=stdevLVasym,
            mLVthin=mLVthin, stdevLVthin=stdevLVthin,
            mCSnum=mCSnum, stdevCSnum=stdevCSnum,
            mCSnumlobe=mCSnumlobe, stdevCSnumlobe=stdevCSnumlobe,
            mCSsource=mCSsource, stdevCSsource=stdevCSsource,
            mCSLOLL=mCSLOLL, stdevCSLOLL=stdevCSLOLL,
            mCSLOWW=mCSLOWW, stdevCSLOWW=stdevCSLOWW,
            mCSLOl=mCSLOl, stdevCSLOl=stdevCSLOl,
            mCSLOw=mCSLOw, stdevCSLOw=stdevCSLOw,
            mCSLO_hwratio=mCSLO_hwratio, stdevCSLO_hwratio=stdevCSLO_hwratio,
            mCSLO_dwratio=mCSLO_dwratio, stdevCSLO_dwratio=stdevCSLO_dwratio,
            mFFCHprop=mFFCHprop, stdevFFCHprop=stdevFFCHprop,
            mNeckFFCHprop=mNeckFFCHprop,
            ntime_per_level=ntime_per_level,
            Cf=Cf, A=scour_factor, I=gradient, Q=Q,
            CHndraw=CHndraw, ndiscr=ndiscr, nCHcor=nCHcor,
            azimuth=azimuth, seed=seed,
        )
        engine.simulation()
        self._finalize_facies_table(
            engine.facies, facies_props=facies_props,
            depth_norm=engine.depth_norm,
            poro_mult_field=engine.poro_mult_field,
            log_perm_offset_field=engine.log_perm_offset_field,
            poro_realization_mult=poro_realization_mult,
            perm_realization_mult=perm_realization_mult,
        )
        # Stash for downstream tooling (parquet writers can record the
        # engine-level multiplier std values, generate.py uses these to
        # derive realized poro_ave / perm_ave for the slim parquet).
        self._engine = engine
        self.poro_mult_std = float(engine.poro_mult_std)
        self.log_perm_offset_std = float(engine.log_perm_offset_std)


# ---------------------------------------------------------------------------
# Importable parameter presets (mirror $HOME/Alluvsim/runs/run_presets.py).
#
# Use as `ChannelLayer.create_geology(**PV_SHOESTRING)` or pass individual
# overrides on top.
# ---------------------------------------------------------------------------
# NOTE on preset ``nlevel`` / ``level_z`` / ``mCHdepth`` choice
# -------------------------------------------------------------
# All presets ship **no explicit ``level_z``** — the engine spreads the
# requested ``nlevel`` levels evenly across the layer's ``z_len`` via
# ``np.linspace(zsiz, nz·zsiz, nlevel)``. ``mCHdepth`` is set to ~4 m
# so each channel is 4 cells thick on the standard ``dz=1 m`` grid
# (nicely visible in cross-section), and ``nlevel`` is chosen so chelev
# spacing ≈ channel depth — adjacent levels just touch, and the column
# fills continuously without hand-tuning ``level_z`` per grid.
PV_SHOESTRING = dict(
    ntime=240, nlevel=8,
    NTGtarget=0.10,
    probAvulOutside=0.10, probAvulInside=0.05,
    mCHsinu=1.6, stdevCHsinu=0.15,
    mCHwdratio=10.0, stdevCHwdratio=1.0,
    mCHdepth=4.0, stdevCHdepth=0.4,
    mLVdepth=0.6, stdevLVdepth=0.1,
    mLVwidth=40.0, stdevLVwidth=5.0,
    mLVheight=0.4, stdevLVheight=0.1,
    mLVasym=0.3, stdevLVasym=0.1,
    mFFCHprop=0.0, stdevFFCHprop=0.0,
    mdistMigrate=35.0, stdevdistMigrate=10.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

CB_JIGSAW = dict(
    ntime=400, nlevel=8,
    NTGtarget=0.30,
    probAvulOutside=0.05, probAvulInside=0.40,
    mCHsinu=1.3, stdevCHsinu=0.1,
    mCHwdratio=16.0, stdevCHwdratio=2.0,
    mCHdepth=4.0, stdevCHdepth=0.4,
    mLVdepth=0.6, stdevLVdepth=0.1,
    mLVwidth=40.0, stdevLVwidth=10.0,
    mLVheight=0.4, stdevLVheight=0.1,
    mLVasym=0.3, stdevLVasym=0.1,
    mFFCHprop=0.5, stdevFFCHprop=0.15,
    mdistMigrate=25.0, stdevdistMigrate=8.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

CB_LABYRINTH = dict(
    ntime=600, nlevel=8,
    NTGtarget=0.25,
    probAvulOutside=0.05, probAvulInside=0.05,
    mCHsinu=1.3, stdevCHsinu=0.1,
    mCHwdratio=16.0, stdevCHwdratio=2.0,
    mCHdepth=4.0, stdevCHdepth=0.4,
    mLVdepth=0.5, stdevLVdepth=0.1,
    mLVwidth=30.0, stdevLVwidth=5.0,
    mLVheight=0.3, stdevLVheight=0.1,
    mLVasym=0.3, stdevLVasym=0.1,
    mFFCHprop=0.4, stdevFFCHprop=0.15,
    mdistMigrate=25.0, stdevdistMigrate=8.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

SH_DISTAL = dict(
    ntime=400, nlevel=6,
    NTGtarget=0.50,
    probAvulOutside=0.02, probAvulInside=0.05,
    mCHsinu=1.3, stdevCHsinu=0.1,
    mCHwdratio=18.0, stdevCHwdratio=2.0,
    mCHdepth=5.0, stdevCHdepth=0.4,
    mLVdepth=1.2, stdevLVdepth=0.2,
    mLVwidth=150.0, stdevLVwidth=25.0,
    mLVheight=1.0, stdevLVheight=0.2,
    mLVasym=0.2, stdevLVasym=0.1,
    mFFCHprop=0.0, stdevFFCHprop=0.0,
    mdistMigrate=35.0, stdevdistMigrate=10.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

SH_PROXIMAL = dict(
    ntime=400, nlevel=8,
    NTGtarget=0.40,
    probAvulOutside=0.08, probAvulInside=0.35,
    mCHsinu=1.2, stdevCHsinu=0.1,
    mCHwdratio=22.0, stdevCHwdratio=3.0,
    mCHdepth=4.0, stdevCHdepth=0.4,
    mLVdepth=0.4, stdevLVdepth=0.1,
    mLVwidth=20.0, stdevLVwidth=5.0,
    mLVheight=0.2, stdevLVheight=0.1,
    mLVasym=0.0, stdevLVasym=0.0,
    mFFCHprop=0.15, stdevFFCHprop=0.05,
    mdistMigrate=25.0, stdevdistMigrate=8.0,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=5.0,
)

# Single sinuous meandering channel that grows tight bends, neck-cuts
# off the oxbow loops, and fills each abandoned bend with a mud plug —
# the classic neck-cutoff → oxbow lake → mud plug succession of a
# meandering river belt, stacked vertically as a multi-storey channel
# sandstone. Per-event geometry is identical to ``tutorial_alluvsim``
# Showcase B (the canonical highly-sinuous case): pure migration
# (``probAvul* = 0``), ``mCHsinu = 1.5``, ``mdistMigrate = 2.0``,
# Pyrcz-Table-1 hydraulics — those values are already known to grow
# tight bends and trigger multiple neck cutoffs per level. We stack
# ``nlevel`` such belts on top of each other (so ``ntime`` is set to
# ~200 events per level) and turn on ``mNeckFFCHprop = 0.5`` so each
# excised oxbow loop is painted as an FFCH mud plug over the upper
# half of its column.
MEANDER_OXBOW = dict(
    # ``nlevel * mCHdepth >= nz·zsiz`` keeps the column saturated:
    # adjacent channels overlap so every Z slice shows the meander belt
    # rather than barren floodplain. The default ``level_z`` spread
    # (engine: ``linspace(mCHdepth, nz·zsiz, nlevel)``) puts the top
    # channel at the grid top and the bottom channel one full depth up.
    nlevel=8, ntime=50, ntime_per_level=True,
    NTGtarget=0.99,
    probAvulOutside=0.0, probAvulInside=0.0,
    mCHsinu=1.5, stdevCHsinu=0.03,
    mCHwdratio=11.0, stdevCHwdratio=0.5,
    mCHdepth=5.0, stdevCHdepth=0.3,
    mLVdepth=0.8, stdevLVdepth=0.1,
    mLVwidth=40.0, stdevLVwidth=5.0,
    mLVheight=0.5, stdevLVheight=0.05,
    mLVasym=0.0, mLVthin=0.0,
    mFFCHprop=0.0, stdevFFCHprop=0.0,
    mNeckFFCHprop=0.5,
    mCSnum=0.0, mCSnumlobe=0.0, stdevCSnum=0.0, stdevCSnumlobe=0.0,
    mdistMigrate=3.0, stdevdistMigrate=0.6,
    Cf=0.0036, scour_factor=10.0, gradient=0.001, Q=3.256,
    stdevCHsource=1.0,
)
