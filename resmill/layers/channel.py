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


# White noise smoothed by a Gaussian of s.d. s has correlation exp(-h^2 / 4 s^2), which falls to
# 5 % (the practical range) at h = 2 sqrt(3) s.
_RANGE_PER_SIGMA = 2.0 * float(np.sqrt(3.0))


def _correlated_noise(shape: tuple, range_xy: float, sigma: tuple | None = None) -> np.ndarray:
    """Unit-variance Gaussian field with lateral correlation ``range_xy``
    cells and a third of that vertically (smoothed white noise), drawn from
    the global numpy state so it follows the layer's seed. ``sigma`` gives the
    smoothing per axis in cells instead."""
    from scipy.ndimage import gaussian_filter
    white = np.random.normal(0.0, 1.0, shape).astype(np.float32)
    if sigma is None:
        r = max(float(range_xy), 1e-3)
        sigma = (r, r, max(r / 3.0, 0.5))
    field = gaussian_filter(white, sigma=sigma, mode="reflect")
    sd = float(field.std())
    return (field / sd if sd > 0 else field).astype(np.float32)


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
                               perm_realization_mult: float = 1.0,
                               poro_noise_std: float = 0.0,
                               poro_noise_range: float = 3.0,
                               perm_poro_slope: float | None = None,
                               poro_max: float | None = None,
                               perm_max: float | None = None,
                               fining_amplitude: float | None = None,
                               fining_perm_decades: float | None = None,
                               fining_clean_fraction: float = 0.5,
                               fining_poro_per_decade: float = 0.0,
                               fining_probability: float = 1.0,
                               noise_range_m: tuple | None = None,
                               fining_top_kvkh: float | None = None,
                               event_group: dict | None = None,
                               levee_ntg_decay_m: float | None = None):
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
        * ``poro_noise_std`` — relative standard deviation of a spatially
          correlated Gaussian field multiplied into the porosity of every
          sand cell (facies >= 1); 0 (default) leaves the bodies as the
          smooth ramp. ``poro_noise_range`` is its lateral correlation
          length in cells (vertical = a third of that). Permeability
          follows through the same K-C slope. Mud cells stay untouched.
        * ``perm_realization_mult`` — single scalar (linear) applied
          uniformly. log10(perm_realization_mult) is added to log_perm
          for every cell. Sobol-sampled log-uniformly, default 1.0.
        * ``facies_props`` entries may also carry ``poro_sd`` (log-normal
          relative porosity spread) and
          ``log10_perm_sd`` (permeability spread at a given porosity, in
          decades): correlated fields (range ``poro_noise_range``) applied
          to that facies' porosity and log-permeability, mud included.
          Measured rock has 0.2-0.56 decades of spread by facies (Corbett &
          Jensen 1992); without these keys a facies has none beyond the
          ramp and the event draws.
        * ``perm_poro_slope`` — decades of permeability per porosity unit
          (0.01) for sand facies, replacing the K-C terms below: log_perm
          = facies value + realization shift + the event's offset + slope
          x (poro - the facies' mean porosity); each facies' porosity is
          rescaled so its average is its facies value, so a facies' values
          are its averages and each flow event keeps its own quality.
          Within formations the slope is 0.10-0.31 (median 0.19; Nelson &
          Kibler 2003), steeper than K-C's ~0.05. None keeps the K-C terms.
        * ``poro_max`` — a soft porosity cap: porosity above 90 % of it
          bends smoothly towards it, with no pile-up at the cap. Applied
          before the slope. None keeps the hard 0.5 clip.
        * ``perm_max`` — a soft permeability cap (mD): log-permeability
          within half a decade of it bends smoothly towards it. None: no cap.
        * ``fining_amplitude`` — the fining-upward ramp is 1 -+ this
          (base to top of each event); None keeps Alluvsim's 0.7-1.3 (0.3).
        * ``fining_perm_decades`` — grain-size fining upward inside each
          channel fill (CH, LA), in place of the porosity ramp: permeability
          is flat in the fill's clean lower part (``fining_clean_fraction`` of
          its depth) and falls linearly above it, by this many decades at the
          top; porosity falls ``fining_poro_per_decade`` units per decade
          (clay-rich tops). A fraction ``fining_probability`` of the channel
          storeys fine upward, the others stay blocky: ``event_group`` maps
          each flow event's rock pair to its storey (the engine's
          ``event_levels``); without it each flow event decides alone. Each
          facies keeps its value as its average. None: off.
        * ``noise_range_m`` — (horizontal, vertical) correlation lengths in
          metres of the ``poro_sd`` / ``log10_perm_sd`` noise, so its texture
          is the same on any grid. None keeps ``poro_noise_range`` cells.
        * ``facies_props`` entries may carry ``kvkh``, the facies' vertical to
          horizontal permeability ratio (``self.kvkh_mat``, written as PERMZ =
          kvkh x PERMX); facies without one keep the layer's ``kzkx``.
          ``fining_top_kvkh`` lowers it log-linearly inside the fining part of
          the fills that fine upward, to this value at their top (thin shales).
          Neither given: no ``kvkh_mat`` and PERMZ stays ``kzkx`` x PERMX.
        * ``levee_ntg_decay_m`` — levee fading: when ``facies_props[2]`` gives
          the levee's sub-cell sand fraction ``ntg`` and its sand beds
          ``bed_poro`` / ``bed_log10_perm``, each levee cell's fraction falls
          as exp(-d / this) with its plan distance d (m) from the channel belt
          of its own storey (every column holding CH or LA of the same level,
          from ``event_group``; all channels without it), scaled so the levee average stays
          ``ntg``; or, with ``ntg_crest`` (and ``ntg_floor``) instead of
          ``ntg``, it is ntg_crest x exp(-d / this), never below the floor. Its
          porosity and permeability are the arithmetic mix of the beds and the
          FF mud. The facies stay LV. None: off.

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
        props = {code: dict(values) for code, values in FACIES_PROPS.items()}   # never edit the module table
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
        poro_sd = np.zeros(self.facies.shape, dtype=np.float32)
        log_perm_sd = np.zeros(self.facies.shape, dtype=np.float32)
        for code, vals in props.items():
            mask = (self.facies == code)
            if mask.any():
                base_poro[mask] = vals["poro"]
                base_log_perm[mask] = vals["log10_perm"]
                poro_sd[mask] = vals.get("poro_sd", 0.0)
                log_perm_sd[mask] = vals.get("log10_perm_sd", 0.0)

        # Opt-in levee fading: sandier levee cells near the channel belt, muddier away from it.
        levee = props.get(2, {})
        faded_levee = False
        if levee_ntg_decay_m is not None and ("ntg" in levee or "ntg_crest" in levee) and (self.facies == 2).any():
            from scipy.ndimage import distance_transform_edt
            lv = self.facies == 2
            fill = (self.facies == 3) | (self.facies == 4)
            # Each levee cell is measured from the channels of its own storey (the level of the flow
            # event that built it, from ``event_group``), not from channels above or below it.
            storey = np.full(self.facies.shape, -1, dtype=np.int64)
            if event_group:
                idx = np.nonzero(lv | fill)
                storey[idx] = [event_group.get((a, b), -1)
                               for a, b in zip(poro_mult_field[idx], log_perm_offset_field[idx])]
            dist = np.zeros(self.facies.shape)
            for level in np.unique(storey[lv]):
                belt = (fill & ((storey == level) if level >= 0 else True)).any(axis=2)
                plan = (distance_transform_edt(~belt, sampling=(self.dx, self.dy)) if belt.any()
                        else np.zeros(belt.shape))
                own = lv & (storey == level)
                dist[own] = np.broadcast_to(plan[:, :, None], self.facies.shape)[own]
            fade = np.exp(-dist[lv] / float(levee_ntg_decay_m))
            if "ntg_crest" in levee:   # the fraction at the channel, down to a floor
                ntg = np.clip(float(levee["ntg_crest"]) * fade, float(levee.get("ntg_floor", 0.0)), 1.0)
            else:                      # the fraction's average kept
                ntg = np.clip(float(levee["ntg"]) * fade / fade.mean(), 0.0, 1.0)
            mud = props[-1]
            base_poro[lv] = ntg * float(levee["bed_poro"]) + (1.0 - ntg) * float(mud["poro"])
            base_log_perm[lv] = np.log10(ntg * 10.0 ** float(levee["bed_log10_perm"])
                                         + (1.0 - ntg) * 10.0 ** float(mud["log10_perm"]))
            faded_levee = True

        # Per-event Walker-1992 upward-fining ramp.
        if fining_perm_decades is not None:
            ramp = np.ones(self.facies.shape, dtype=np.float32)
        elif fining_amplitude is None:
            ramp = (0.7 + 0.6 * depth_norm).astype(np.float32)
        else:
            a = float(fining_amplitude)
            ramp = ((1.0 - a) + 2.0 * a * depth_norm).astype(np.float32)

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

        # Grain-size fining upward in the fills' upper parts (decades), per flow event;
        # an event is the set of cells sharing its (poro_mult, log_perm_offset) draw.
        fining = None
        if fining_perm_decades is not None:
            fill = (self.facies == 3) | (self.facies == 4)
            clean = float(fining_clean_fraction)
            fining = np.where(fill, float(fining_perm_decades) * np.clip(
                (1.0 - clean - depth_norm) / max(1.0 - clean, 1e-6), 0.0, 1.0), 0.0).astype(np.float32)
            if float(fining_probability) < 1.0 and fill.any():
                pairs = np.stack([poro_mult_field[fill], log_perm_offset_field[fill]], axis=1)
                unique, event = np.unique(pairs, axis=0, return_inverse=True)
                event = event.ravel()
                if event_group:   # one choice per storey; an unknown event is its own group
                    group = [event_group.get((np.float32(a), np.float32(b)), -1 - i)
                             for i, (a, b) in enumerate(unique)]
                    event = np.unique(group, return_inverse=True)[1].ravel()[event]
                fines = np.random.random(int(event.max()) + 1) < float(fining_probability)
                fining[fill] *= fines[event]
            poro_drop = 0.01 * float(fining_poro_per_decade) * fining
            if perm_poro_slope is not None:   # calibrated: each facies' value stays its average
                for code in (3, 4):
                    mask = self.facies == code
                    if mask.any():
                        poro_drop[mask] -= poro_drop[mask].mean()
            poro_mat = poro_mat - poro_drop

        # Mud cells (FF, FFCH) get base FACIES_PROPS values — no ramp,
        # no per-event mult, no per-realization shift (mud is not the
        # reservoir-quality control variable in this scheme).
        mud_mask = (self.facies == -1) | (self.facies == 0)
        if mud_mask.any():
            poro_mat[mud_mask] = base_poro[mud_mask]
            log_perm[mud_mask] = base_log_perm[mud_mask]

        # Cell-scale texture inside sand bodies: a correlated Gaussian
        # field, one draw per cube, so a channel is not a perfect ramp.
        if poro_noise_std > 0.0:
            noise = _correlated_noise(self.facies.shape, poro_noise_range)
            mult = np.clip(1.0 + float(poro_noise_std) * noise, 0.3, 1.7).astype(np.float32)
            sand_mask = self.facies >= 1
            poro_mat = np.where(sand_mask, poro_mat * mult, poro_mat)
            log_perm = np.where(sand_mask, log_perm + KC_SLOPE * np.log10(mult), log_perm)

        # Opt-in rock spread (per-facies poro_sd / log10_perm_sd), the porosity
        # cap and a porosity-permeability slope in decades per porosity unit.
        # With none of them given nothing is drawn and the cap is the 0.5 clip.
        sigma = None
        if noise_range_m is not None:
            horizontal, vertical = (float(v) / _RANGE_PER_SIGMA for v in noise_range_m)
            sigma = (horizontal / self.dx, horizontal / self.dy, vertical / self.dz)
        if poro_sd.any():
            poro_mat = poro_mat * np.exp(poro_sd * _correlated_noise(self.facies.shape, poro_noise_range, sigma))
        if perm_poro_slope is not None:
            # Calibrated rock: each facies' average porosity is its facies value (the
            # ramp, the cap and erosion otherwise shift the preserved cells' mean).
            sand_mask = self.facies >= 1
            for code in np.unique(self.facies):
                mask = self.facies == code
                value = (base_poro[mask].mean() if code == 2 and faded_levee   # a faded levee keeps its own mean
                         else np.float32(props[int(code)]["poro"]))
                target = float(value) * (float(poro_realization_mult) if code >= 1 else 1.0)
                mean = float(poro_mat[mask].mean())
                if mean > 0.0:
                    poro_mat[mask] = poro_mat[mask] * (target / mean)
        if poro_max is not None:
            knee = 0.9 * float(poro_max)
            span = float(poro_max) - knee
            poro_mat = np.where(poro_mat > knee, knee + span * np.tanh((poro_mat - knee) / span), poro_mat)
        if perm_poro_slope is not None:
            sand_mask = self.facies >= 1
            pivot = np.zeros(self.facies.shape, dtype=np.float32)
            for code in np.unique(self.facies[sand_mask]):
                mask = self.facies == code
                pivot[mask] = poro_mat[mask].mean()
            log_perm = np.where(sand_mask, base_log_perm + log_perm_realization_offset_f32
                                + log_perm_offset_field
                                + 100.0 * float(perm_poro_slope) * (poro_mat - pivot), log_perm)
        if fining is not None:
            # The slope already turns the porosity drop into part of the permeability drop.
            coupled = float(perm_poro_slope) * float(fining_poro_per_decade) if perm_poro_slope is not None else 0.0
            drop = fining * (1.0 - coupled)
            if perm_poro_slope is not None:   # calibrated: each facies' value stays its average
                for code in (3, 4):
                    mask = self.facies == code
                    if mask.any():
                        drop[mask] -= drop[mask].mean()
            log_perm = log_perm - drop
        if log_perm_sd.any():
            log_perm = log_perm + log_perm_sd * _correlated_noise(self.facies.shape, poro_noise_range, sigma)
        if perm_max is not None:
            knee = np.log10(float(perm_max)) - 0.5
            log_perm = np.where(log_perm > knee, knee + 0.5 * np.tanh((log_perm - knee) / 0.5), log_perm)

        # Inactive cells (FF=-1) get poro = base FF value; perm tracks.
        # Clip poro to a physical range to avoid float16 overflow / negatives.
        poro_mat = np.clip(poro_mat, 0.0, 0.5)
        perm_mat = (10.0 ** log_perm).astype(np.float32)

        self.poro_mat = poro_mat.astype(np.float32)
        self.perm_mat = perm_mat.astype(np.float32)

        # Opt-in vertical anisotropy per facies, thinner-bedded towards fining tops.
        self.kvkh_mat = None
        if any("kvkh" in vals for vals in props.values()) or fining_top_kvkh is not None:
            kvkh = np.full(self.facies.shape, float(self.kzkx), dtype=np.float32)
            for code, vals in props.items():
                if "kvkh" in vals:
                    kvkh[self.facies == code] = vals["kvkh"]
            if fining_top_kvkh is not None and fining is not None:
                g = fining / float(fining_perm_decades)          # 0 at the clean part's top, 1 at the fill's top
                kvkh = kvkh * (float(fining_top_kvkh) / kvkh) ** g
            self.kvkh_mat = kvkh.astype(np.float32)


    def create_geology(
        self,
        # ---- aggradation -----------------------------------------------
        nlevel: int = 8,
        level_z: list[float] | None = None,
        NTGtarget: float = 0.10,
        ntime: int | list[int] = 240,
        # ``True`` ⇒ ``ntime`` is interpreted per-level (counter resets
        # at every level); ``False`` ⇒ ``ntime`` is the total event cap
        # across all levels (Alluvsim default). Either way it is sized for
        # an 800 x 800 m grid; ``scale_ntime=True`` scales it by
        # sqrt(area / 800²) on larger grids (off by default). With
        # ``ntime_per_level`` a list gives one cap per level.
        ntime_per_level: bool = False,
        scale_ntime: bool = False,
        # Probability that a level continues the previous level's channel
        # path (organized stacking); 0 draws a fresh path every level.
        level_inherit: float = 0.0,
        # Deep, steep side of the channel and the wider levee on the outer bank
        # of each bend (where it migrates to); False keeps Alluvsim's inner bank.
        cutbank_outer: bool = False,
        # Neck cutoffs only for loops at least this many times longer than the
        # neck they close; 1 keeps Alluvsim's rule, which also shaves bends.
        cutoff_loop_ratio: float = 1.0,
        # Keep every channel flowing out of the volume: walk it on whenever its
        # downstream end drifts back inside (False keeps Alluvsim's dead ends).
        extend_to_boundary: bool = False,
        # Thalweg position at a tight bend (0.9 rivers, 0.8 submarine); each bend
        # gets its asymmetry from its own tightness. None: Alluvsim's rule.
        thalweg_max: float | None = None,
        # Unwrap the compass heading before smoothing it (False: Alluvsim's
        # curvature spikes wherever a channel heads due north).
        unwrap_azimuth: bool = False,
        # Thalweg by the migration model's upstream-weighted curvature, so the pool
        # lags the bend as the bank erosion does (False: local curvature).
        thalweg_lag: bool = False,
        # Metres every channel path is drawn beyond the grid at both ends (never
        # painted), so neither end pins the channel inside the grid: no belt
        # pinched at the entry, no straight end stretches, no dead ends from tail
        # cutoffs. 0: Alluvsim's paths, which start and stop at the edge.
        path_buffer: float = 0.0,
        # One continuous wall from channel to levee: levees start with the channel
        # wall's slope and the channel clears the space above it along its walls.
        # False: Alluvsim's rules (a slope break at the bank, vertical cuts).
        continuous_banks: bool = False,
        # Metres between a channel path's points, so the channel is the same on any
        # grid (the bend rules count points). None: one grid cell, as in Alluvsim.
        path_step: float | None = None,
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
        # Cell-scale porosity texture inside sand bodies (0 = smooth ramp)
        poro_noise_std: float = 0.0,
        poro_noise_range: float = 3.0,
        # Permeability per porosity unit for sands (None: K-C); soft porosity and permeability caps
        perm_poro_slope: float | None = None,
        poro_max: float | None = None,
        perm_max: float | None = None,
        # Fining-upward ramp 1 -+ this (None: Alluvsim's 0.3); per-event rock spreads
        fining_amplitude: float | None = None,
        event_poro_sd: float = 0.04,
        event_log_perm_sd: float = 0.12,
        # Permeability fining upward in channel fills and noise ranges in metres (see
        # _finalize_facies_table; None: off)
        fining_perm_decades: float | None = None,
        fining_clean_fraction: float = 0.5,
        fining_poro_per_decade: float = 0.0,
        fining_probability: float = 1.0,
        noise_range_m: tuple | None = None,
        # kv/kh at the top of the fills that fine upward (facies values: facies_props "kvkh")
        fining_top_kvkh: float | None = None,
        # Levee fading: decay length (m) of the levee cells' sand fraction away from the channel belt
        levee_ntg_decay_m: float | None = None,
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
            ntime_per_level=ntime_per_level, scale_ntime=scale_ntime,
            level_inherit=level_inherit, cutbank_outer=cutbank_outer,
            cutoff_loop_ratio=cutoff_loop_ratio, extend_to_boundary=extend_to_boundary,
            thalweg_max=thalweg_max, unwrap_azimuth=unwrap_azimuth, thalweg_lag=thalweg_lag,
            path_buffer=path_buffer, continuous_banks=continuous_banks, path_step=path_step,
            event_poro_sd=event_poro_sd, event_log_perm_sd=event_log_perm_sd,
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
            poro_noise_std=poro_noise_std,
            poro_noise_range=poro_noise_range,
            perm_poro_slope=perm_poro_slope,
            poro_max=poro_max,
            perm_max=perm_max,
            fining_amplitude=fining_amplitude,
            fining_perm_decades=fining_perm_decades,
            fining_clean_fraction=fining_clean_fraction,
            fining_poro_per_decade=fining_poro_per_decade,
            fining_probability=fining_probability,
            noise_range_m=noise_range_m,
            fining_top_kvkh=fining_top_kvkh,
            event_group=engine.event_levels,
            levee_ntg_decay_m=levee_ntg_decay_m,
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
