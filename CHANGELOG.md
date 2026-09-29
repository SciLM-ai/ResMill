# Changelog

## Unreleased

### Added

- `unwrap_azimuth=True` (`ChannelLayer`, and `DeltaLayer` through its
  fluvial passthrough): unwrap the channel's compass heading before
  smoothing it. Smoothing across the 360 -> 0 jump fakes a full turn
  wherever a channel heads due north: on a circle of radius 500 m the
  curvature there reads +5.4 deg/m instead of -0.115, and the tangents are
  wrong too. Curvature drives the migration (scaled by its largest bank
  velocity), the thalweg and levee sides and splay placement; channels head
  through north in 71 % (deepwater) and 82 % (meander) of migration steps,
  leaving zig-zag kinks there. Off by default: outputs are bit-identical to
  before.
- `thalweg_max` (`ChannelLayer`, and `DeltaLayer` through its fluvial
  passthrough): the thalweg position (fraction of the width from the inner
  bank) at a tight bend. Each bend then gets its asymmetry from its own
  tightness, the transverse bed slope A*H/R with A = 3 (Ikeda et al. 1981;
  Odgaard 1981), reaching `thalweg_max` once the radius is 1.5 channel
  widths; straight reaches and crossings stay symmetric. Alluvsim's rule,
  kept by default, scales every bend by the sharpest point of its channel
  (at most 0.75), so one kink leaves the other bends nearly symmetric.
  Observed: ~0.9 in river bends (Fisk 1952), ~0.75-0.8 in submarine bends
  (Jobe et al. 2010; Palm et al. 2021). Off by default: outputs are bit-
  identical to before.
- `extend_to_boundary=True` (`ChannelLayer`, and `DeltaLayer` through its
  fluvial passthrough): whenever migration or a cutoff leaves a channel's
  downstream end inside the grid, the channel is walked on (the pool's AR(2)
  walk, launched along its last heading) until it leaves the grid. Without
  it the free end drifts back into the volume and channels stop in the
  middle of it: 15 % of the channels of a deepwater complex and 32 % of
  meander_oxbow's. Leave it off for distributaries meant to end in the
  volume (delta mouth bars). Off by default: outputs are bit-identical to
  before.
- `cutoff_loop_ratio` (`ChannelLayer`, and `DeltaLayer` through its fluvial
  passthrough): a neck cutoff also needs the channel between the two points
  that meet to be at least this many times longer than the gap between them.
  Alluvsim's rule (1, the default, no such check) also removes gently curved
  stretches whose shortcut stays inside the channel, 2-3 per migration step
  in the presets, so bends are straightened every step and a collapsed path
  is occasionally replaced by a random new one; with 3 only real loops are
  cut off, however short. Default 1: outputs are bit-identical to before.
- `cutbank_outer=True` (`ChannelLayer`, and `DeltaLayer` through its fluvial
  passthrough): puts each bend's cut bank on the outer bank, the side the
  channel migrates to and where crevasse splays form: the channel's deepest
  point and steep side (the thalweg of the Deutsch-Wang cross-section) and
  the wider levee. Alluvsim's rules, kept by default, put both on the inner
  bank, while the port migrates bends outward (so they grow) and places
  splays on the outer bank. Off by default: outputs are bit-identical to
  before.
- `level_inherit` (`ChannelLayer`): the probability that a new level
  continues the previous level's channel path instead of drawing a fresh one
  from the pool. With 1 the levels stack into an organized complex; with
  `ntime` given as a list (one event cap per level, with
  `ntime_per_level=True`) many events in the first levels and few later give
  the laterally-migrating-then-aggrading stacking of submarine channel belts
  (Jobe et al. 2016; Covault et al. 2016). Both off by default: outputs are
  bit-identical to before.
- `scale_ntime=True` (`ChannelLayer`, and `DeltaLayer` through its fluvial
  passthrough): the event cap `ntime`, which the presets size for an
  800 x 800 m grid, is scaled by sqrt(area / 800²) on larger grids. A
  channel crosses the grid, so the sand one event adds grows with the
  grid's length. On a 4 x 3 km layer the braided and sheet presets otherwise
  stop at 28-60 % of their NTG target, the sheet presets with the top half
  of the layer empty; with the switch they reach it at every level, and the
  legacy delta keeps its design density. Off by default; the published
  dataset is unchanged.
- `ChannelLayer(n_sources=N)`: N entry points on the upstream edge, drawn
  per reservoir at least `source_spacing_min` of the edge apart; every
  channel enters at one of them, the draws rotating through the sources so
  every entry is represented; nothing else changes. Default 1 leaves the
  published channels bit-identical. `examples/dataset_generation/config_full_<preset>_v2.json`
  sample `n_sources` from {1, 1, 2, 3}.
- `DeltaLayer(bifurcate=True)`: a generation is a distributary tree, not an
  avulsion history. Discharge is split at every bifurcation (`split_frac_lo`
  to `split_frac_hi` to the new branch), width follows `Q ** width_exp` and
  depth `Q ** depth_exp` (Leopold-Maddock), a segment gives off at most
  `max_splits_per_branch` branches so splits cascade down the network,
  branches launch at `branch_angle_mean` +- `branch_angle_sd` degrees and
  ease back toward the regional slope, run at most
  `branch_length_scale * diagonal * sqrt(Q)` before ending in a mouth bar,
  and join another branch they run into (`merge_branches`); splits are Ys
  (`y_split`), land near a segment's upstream end when `split_pos_exp` > 1,
  branches ease back toward the regional slope by `branch_relax`, taper to
  `branch_taper` of their start width, and their mouth bars are sized by
  the tip's own width. A branch ends where its own lobe ends, at
  `front_radius * (1 + front_bulge * sqrt(Q))` (jittered) from the apex, so
  bigger channels prograde further and the shoreline is scalloped, never a
  circle; with `front_radius` > 1 the front lies outside the grid and
  channels run through to its edges. At every terminus the channel splits
  around its mouth bar into two short terminal channels with bars of their
  own; the front progrades with the generations (`progradation_fraction`);
  only terminal channels (share below `q_min`) stop by a width-scaled
  length; a bar sits at every bifurcation; with `n_trees` networks per
  generation the older ones are abandoned and mud-filled with `mFFCHprop`. Off by default; the published dataset is
  unchanged. `examples/dataset_generation/config_full_delta_v2.json` is the
  dataset config for it. `layer.tree_branches` lists every segment.
- `dome()` gains `aspect` and `azimuth` (elongated four-way closures);
  `examples/anticline_meander.py` and `examples/angular_unconformity.py`.

### Fixed

- Mouth bars (`paint_mouth_bars`) build downward from the channel top instead
  of straddling it, so no bar sand floats above a generation's channels.
- Fluvial streamlines (channel, delta) now enter the grid on its boundary.
  The entry was drawn on the upstream edge of the unrotated walk frame and
  the streamline rotated by `azimuth` about the grid centre at stamping, so
  for any azimuth off a multiple of 90 degrees channels, and the whole delta
  fan, started inside the grid with nothing feeding them, and entries that
  rotated out of the grid were wasted. Entries are slid along the mean flow
  onto the boundary and drawn across everything the grid spans perpendicular
  to the flow.
- The AR(2) walk is clipped where the channel lands, not where it is walked,
  so the grid's corners fill at every azimuth and net-to-gross no longer
  depends on azimuth.
- The walk's step cap (`ndis_cap`) is separate from the streamline node count
  (`ndis0`). The cap is a safety net sized from the grid diagonal, so
  channels on elongated grids no longer stop mid-domain; the node count keeps
  Alluvsim's density (two nodes per cell of the flow-direction span), since
  it sets every per-node rule (MEANDER_OXBOW net-to-gross fell from 0.45 to
  0.26 when it was tripled).
- `to_grdecl`: cells thinner than the ZCORN write precision are written
  inactive (`_MIN_THICKNESS` 5 mm), and `facies=True` is validated before
  the file is created.
- `plot_section` raises a clear error on a fully eroded model.
- `Reservoir` docstring: stacked arrays are copies; edit the layers for export.

## 0.2.0 (2026-08-31)

### Added

- `resmill.structure`: composable structural deformation fields
  (`anticline`, `syncline`, `dome`, `ramp`, `fault`, `surface`), vertical
  shifts in meters, positive down; plain callables, 2-D arrays, and
  scalars are accepted anywhere a `Structure` is.
- `to_grdecl()` (also as `Layer.to_grdecl` / `Reservoir.to_grdecl`):
  self-contained Petrel/Eclipse corner-point export (SPECGRID, COORD,
  ZCORN, ACTNUM, PORO, PERMX, PERMY, PERMZ, optional FACIES) with
  `top=`/`base=` conforming (drape or proportional squeeze),
  `erode_above=`/`erode_below=` truncation with ACTNUM, per-layer
  structure lists stitched with a younger-truncates-older rule (angular
  unconformities), and PERMZ from each layer's `kzkx`.
- `plot_section()`: true-depth cross-sections of the deformed grid.
- `to_pyvista()`: `pyvista.ExplicitStructuredGrid` for interactive 3-D QC
  (new `viz` extra).

### Changed

- **Breaking**: `Reservoir` now concatenates `poro_mat`/`perm_mat`/`active`
  with the deepest layer first, so stacked arrays follow the documented
  per-layer convention (k index increases upward) and stacked
  cross-sections from `plot_slices`/`plot_cube_slices` render the layer
  order correctly. Previously the top layer occupied global k=0..nz-1 and
  a stacked section drew the reservoir upside down. Code that indexed
  `Reservoir` arrays along k must be updated; `Reservoir.layers` and
  `Reservoir.zz` keep the listed top-to-bottom order.
- `__version__` and `pyproject.toml` are bumped together (0.1.3 -> 0.2.0);
  the 0.1.2/0.1.3 resync had already happened in 0.1.3.

## 0.1.3 (2026-08-30)

- Moved the repository to the SciLM-ai org, added the PyPI
  trusted-publishing workflow.

## 0.1.2

- LICENSE copyright set to the authors; first PyPI release line.
