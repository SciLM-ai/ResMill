# Changelog

## Unreleased

### Fixed

- `facies_props` given to one `ChannelLayer` or `DeltaLayer` no longer
  rewrite the module's `FACIES_PROPS`. The table was copied one level deep
  and its inner dicts updated in place, so every later layer built without
  `facies_props` in the same process got the first layer's values (one
  call with mud at 10^-5 mD left the default mud at 10^-5 mD). Layers that
  never pass `facies_props` are unchanged.

### Added

- `splay_step` (`ChannelLayer`, and `DeltaLayer` through its fluvial
  passthrough): metres between the points of a splay's walk. Alluvsim walks a
  splay one grid cell per step, draws as many random numbers as it takes steps,
  and paints a one-cell-wide thin sheet along the walk, below the lobe. So a
  splay followed the cell (roadmap item 13): on 12.5, 25 and 50 m cells the same
  settings gave x0.76, x1 and x1.41 of the splay volume for a lobe 5 channel
  widths long and 3 wide (six seeds), x0.63, x1 and x1.88 for one half a
  channel width wide, where the sheet is about 90 % of the splay at 25 m; and
  every later random draw, so the whole model, differed between grids. With a
  step in metres the walk, its draws and the lobe are the same on any grid, and
  the sheet is not painted: a splay is its lobe, and its volume is the lobe's
  analytic one (the integral of the half-width squared along it,
  `tests/test_channel.py`). On those three cell sizes the seed means then agree
  to 0.5 % for the wide lobe (x1.00, x1.00, x1.00) and 3 % for the narrow one
  (x0.97, x1, x1.00); between 20 m and 40 m cells (six seeds) the volume ratio
  is 1.01, 0.98 to 1.08 per seed, against 1.50, 0.99 to 2.10, without it. `None`,
  the default, keeps the cell and the sheet: outputs are bit-identical to before.
- `max_sinuosity` (`ChannelLayer`, and `DeltaLayer` through its fluvial
  passthrough): the upper clip of every drawn sinuosity, 1.9 by default
  (Alluvsim's, outputs bit-identical). Freely meandering reaches have 2.0, 2.34
  and 2.75 (P10, P50, P90 of 38 reaches, Finotello et al. 2020 data and
  Camporeale et al. 2005), above the clip. The walk's noise rule is linear in
  the sinuosity, so the realized sinuosity grows steeply above 2: with bends
  100 m / 6 apart (24 seeds) a walk asked for 2.0 has a median 1.9, 2.1 gives
  2.25, 2.2 gives 2.5 (P10 2.2, P90 3.0), 2.3 gives 2.9 and 2.5 gives 4.1. Raise
  the clip and ask for about 2.0 to 2.2.
- `path_step` (`ChannelLayer`, and `DeltaLayer` through its fluvial
  passthrough): metres between a channel path's points. Alluvsim spaces
  them one grid cell apart, and its bend rules count points: migration
  looks 30 points upstream and curvature is smoothed over 10. So the same
  river bends differently on another grid. A 76 m wide meandering river
  (seed 1, 1536 events) had loops on 10 m cells (bend radius 2 channel
  widths, 16 cutoffs), gentle bends on 25 m (3.8, 1) and was nearly
  straight on 50 m (9, none). With `path_step` the points are that many
  metres apart on any grid, and one seed gives one channel on every grid
  (splays still walk one cell per step). The spacing then sets the bend
  size, so give it as a fraction of the channel width. Default `None`
  (one grid cell): outputs are bit-identical to before.
- `continuous_banks=True` (`ChannelLayer`, and `DeltaLayer` through its
  fluvial passthrough): carry each channel's wall up into its levees.
  A levee's top starts at the bank with the wall's own mean slope on that
  side, the Deutsch-Wang cross-section's slope next to the bank (steep
  on the cut bank, gentle over the point bar). It rounds over into
  Alluvsim's crest (`LVheight`/e) and then follows Alluvsim's flank. The
  active channel clears the space above it along those walls, carried up
  to one channel width past the banks, instead of a vertical column.
  Alluvsim's rules, kept by default, give the levee a slope of its own,
  with the crest a sixth of `LVwidth` out. That leaves a slope break at
  the bank, and levees wider than about five channel widths are thinnest
  beside the channel. The channel also cuts older levees vertically.
  Observed: the channel wall rises continuously to the levee crest, and
  crests sit on the channel margin at every levee size (Pirmez & Flood
  1995; Jobe et al. 2020). Deepwater levees 1.5, 4 and 10 channel widths
  wide then plot inside the observed relief and far-field aggradation
  of 52 seismic sections. Off by default: outputs are bit-identical to
  before.
- `path_buffer` (`ChannelLayer`, and `DeltaLayer` through its fluvial
  passthrough): metres every fresh channel path is drawn beyond the grid at
  both ends. It starts that far upstream of its entry and goes on that far
  past the edge where it leaves, with path nodes added in proportion so
  their spacing stays the same; the parts outside are never painted.
  Alluvsim's paths start and stop at the edge, where they pin the channel:
  the first point never migrates and the last one only slides along the
  edge. So every position of the channel passes through one entry point,
  and its belt pinches there; the stretches next to both ends stay straight;
  and when a neck cutoff pairs the last point with one upstream, it deletes
  everything in between, the end included, so the channel stops inside the
  volume. In a deepwater complex (16 seeds) 8 % of channels ended inside;
  100 m inside the entry the belt was 0.58 of its mid-grid width; the first
  and last km had sinuosity 1.39 and 1.28 against 1.75 mid-grid. With 1 km
  no channel ends inside, the belt at the entry is 0.94 of mid-grid, and
  the first, middle and last km read 1.51, 1.65 and 1.52. Default 0:
  outputs are bit-identical to before.
- `thalweg_lag=True` (`ChannelLayer`, and `DeltaLayer` through its fluvial
  passthrough): place the thalweg (its side and strength) by the curvature
  the migration model feels, the same exponentially weighted mean over 30
  upstream nodes as the bank-velocity integral, instead of the local
  curvature. The migration responds to the curvature upstream, so just past
  an inflection the channel still erodes toward the previous bend's outer
  bank while the local-curvature thalweg has already switched banks: the
  point bar forms next to the pool. That was 25 % of all migration in a
  deepwater complex (74 % within one width of an inflection); with the lag
  it is 7 % (6 %). In nature the pool and the point bar lag the bend on
  opposite banks (Palm et al. 2021; Fisk 1952). Off by default: outputs are
  bit-identical to before.
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
- `compensation_scale` (`LobeLayer`): the weight of a column as the next stamp's centre is
  `exp(-deficit / (compensation_scale * dh_ave))`, the deficit being its height above the lowest
  column in metres, so a column one scale of stamp thicknesses higher is e times less likely.
  ResMill's weight, `(height above the lowest column in cells + 0.001) ** -m`, puts every stamp on the
  lowest column from `m` = 2 up, so `m` = 2, 10 and 50 give the same stack and the same compensation
  index kappa (Straub et al. 2009): 0.83 for about 350 stamps of 1.4 km (3 m peak), 0.86 for 90 stamps,
  0.93-0.95 for 15-19 stamps wider than the model, and the strength could not be varied. With the scale
  kappa falls smoothly as the scale grows (about 350 stamps, 2 seeds: 0.81-0.85 at 0.2 or less, 0.75 at
  0.5, 0.70 at 1, 0.66 at 2; random stacking is 0.55). Default `None` keeps the old weight: outputs are
  bit-identical to before.
- `facies_props` (`LobeLayer`): rock by facies for lobes, keyed by facies code as in `ChannelLayer`.
  Without it the sand is the top `ntg` share of the lobe structure, whose porosity and permeability are
  standardized over all cells before the mud is zeroed, so the sand's average moves with `ntg` (porosity
  0.292, 0.274 and 0.256 for a requested 0.25 at `ntg` 0.25, 0.5 and 0.85), the mud has no rock (the
  deck's floors) and PERMZ is one `kzkx` per layer. With it the sand keeps exactly `poro_ave`,
  `perm_ave`, `poro_std` and `perm_std` at any `ntg`; the mud (`-1`) has its own `poro`, `log10_perm`
  and optional `poro_sd` / `log10_perm_sd`; `kvkh` entries give each facies' vertical to horizontal
  ratio per cell (`kvkh_mat`, written as PERMZ). `{2: {"ntg_floor": f, "ntg_crest": c, "kvkh": k}}`
  also fades the sand: a cell's sub-cell sand fraction is an exponential of the rank of its lobe
  structure, `c` (default 1) in the axis and the base of a bed and `f` in the fringe and the top of a
  bed, with the mean (`ntg`) kept; porosity and permeability are the arithmetic mix of sand and mud
  and kv/kh runs log-linearly from `k` at `f` to the sand's at 1. Binary lobes leave a connected
  mud lattice 100-400 m wide around round sand cores in plan view, where Tanqua lobes fade from 85-100 %
  sand at the axis to 20-50 % at the fringe (Spychala et al. 2017); faded lobes have no cell below
  `f` of the sand's permeability. Facies are 3 (sand fraction of 0.5 or more) and 2 (thin-bedded
  fringe); `sand_fraction` holds the fraction. Default `None`: outputs are bit-identical to before.
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
