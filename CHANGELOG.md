# Changelog

## Unreleased

### Fixed

- `max_levels` (`DeltaLayer`, with `tree_ntg_stop=True`) in a zone no thicker than its trunk is deep grew a
  network at the first level only. The progradation follows each level's height above the first level's top,
  divided by the room above it (the zone's thickness minus the trunk's depth), which is nothing or negative in such a
  zone and was taken as 1e-9: every later level had a height of about -1e9, a front radius of nothing and a trunk
  cut to the apex, so the stop ran through all its levels, empty, and ended short of its target (3 x 3 km, 18 m
  trunk, aim 0.55: 0.06, 0.08, 0.12 and 0.18 in zones of 4, 6, 9 and 14 m, one level in 300 grew a network). The
  height is now held to 0-1 and the room is at least a cell: the same grid gives 0.62, 0.59, 0.58 and 0.56. A zone
  more than a cell thicker than its trunk is built as before; `progradation_fraction=0` and `max_levels=None` do not
  see the change.
- `facies_props` given to one `ChannelLayer` or `DeltaLayer` no longer
  rewrite the module's `FACIES_PROPS`. The table was copied one level deep
  and its inner dicts updated in place, so every later layer built without
  `facies_props` in the same process got the first layer's values (one
  call with mud at 10^-5 mD left the default mud at 10^-5 mD). Layers that
  never pass `facies_props` are unchanged.

### Added

- `fuse_prob` and `fuse_max_angle` (`ChannelLayer`): fusion of channels that meet. The engine keeps one active channel at a
  time, so a path drawn from the pool, spliced on by an avulsion inside or moved by migration into an older path of its own
  level cut straight through it and both went on, sand on sand, at 50-90 degrees; coeval channels merge at a confluence and
  a new avulsion channel that meets an older course usually reoccupies it. With `fuse_prob` > 0 a lineage (the migration
  history of one path between two births) is a reoccupier with that probability, drawn when it is born: at the first
  crossing at 15 degrees or more beyond three channel widths of its own length, in its first path and again after every
  migration, its path bends into the older heading over five channel widths and follows the older path to the edge of the
  model (`fuse_max_angle`, 70 degrees: a path meeting the older one more steeply is turned towards it first; a confluence
  joins at an acute angle). Only older paths of the same level count: a younger level cutting an older one is another
  generation and stays a crossing. The older paths are a raster of the level (the cell each finished lineage's paths crossed
  holds the path and segment), so a search costs a lookup of the cells the new path runs through
  (`resmill/layers/_fusion.py`); a build takes 5-12 % longer at field scale (means of 8 builds a table, 5.5 x 4 km, 25 m
  cells). The decision has a random stream of its own seeded from `seed`; a joined path's widths are drawn afresh, so the
  global stream moves on from the first join. On a 1.6 x 1.2 km model with two levels (`tests/test_fusion.py`, seeds 0-2)
  the crossings of two distinct channels of one level (the detector of `tests/channel_paths.py`: 20 degrees or more, three
  widths of each path on either side at least one width from the other's whole belt) are 70 without fusion, 8 at `fuse_prob`
  1 and 65 at 0.5 (half the lineages reoccupy, but the others cut across what they and their copies meet, and a copy
  inherits the crossings of the path it follows); the crossings between levels, 73, fall to 16, because the paths of a level
  bundle onto fewer trunks. At field scale (the sampler's draws of ResSimMill's fluvial tables, 8 seeds each, `fuse_prob` 1,
  5.5 x 4 km on 25 m cells, the same detector) the crossings of distinct channels of one level fall from 205 to 21
  (amalgamated channel belts), 32 to 7 (labyrinth), 24 to 2 (shoestring), 134 to 37 (proximal sheets), 41 to 20 (distal
  sheets) and stay 2 in the meander (too few to count); and so does the net-to-gross, from 0.52 to 0.45, 0.43 to 0.40, 0.19
  to 0.17, 0.58 to 0.42, 0.36 to 0.28 (0.42 to 0.39): a path that joins an older one lays no sand of its own where it
  follows it, so a table that switches fusion on needs its fill curve fitted with it. `fuse_prob` 0, the default, changes
  nothing: outputs bit-identical, nothing drawn.
- `salt_traps.salt_flank_trap(..., upturn_in_outline=False)`: with it the cut (`rim`) is read on the structure written, the
  upturn included, around the crest of the trap the salt leaves, instead of on the fold and its roughness. The ridge an upturn
  lifts along the wall can be most of the model's main trap and lie outside a cut of the pre-salt fold (53 % of it, with a
  folding zone of 500 m and the contact at 0.55 of the half-width); the default outline is what it was.
- `block_styles.tilted_blocks(..., rim=None)` and `rollover(..., rim=None)`: with a `rim` (m) the model also carries
  `BlockModel.outline`, the `(nx, ny)` bool map of the columns within `rim` of the trap the model measured
  (`labels["trap"]`), as `fold_trap` cuts a fold's: pass it to `to_grdecl(outline=)`. The trap is read on the planning
  grid, which may be coarser than the columns of `dx`, and each column is kept when the planning cell holding its centre
  is; `labels` gain `rim` and `outline_columns`. A `rim` and no trap to cut around raises `ValueError`. Nothing is drawn
  for it: without `rim` the model, its labels and every draw are those of before (eight models, four seeds of each
  style, identical byte for byte in structure, labels, faults and growth isochores).
- `to_grdecl(outline=)` and `structure.outline`: a simulation outline cut around a trap, as full field decks cut their
  models. `outline(trap, dx, dy, rim)` is the `(nx, ny)` bool map of the columns within `rim` m (centre to centre) of
  a trap's columns (the `mask` of `closure_stats`, or any other trap's); `to_grdecl(outline=map)` writes every cell of
  the other columns inactive (ACTNUM 0, every layer) and nothing else: ZCORN, the properties, FAULTS, the seal's
  multipliers and `report` are those of the whole map. That is deliberate: read as rock-free walls, the cut columns
  seal a trap inside the outline from a flood that starts at the map's edge (a four-way closure with a 500 m rim: the
  one block fills to its deepest column, 9.9 km2 and 83 m for a closure of 5 km2 and 59 m), so the fault blocks, the
  contacts and the labels are the whole map's, and the cut only says which cells are simulated. `outline=None`, the
  default, changes nothing: outputs are bit-identical to before. In Flow, a small model written whole and cut gave the
  same transmissibilities and fault NNCs among the cells kept, none across the cut and none new.
- `fold_traps.fold_trap(style, x_len, y_len, dx, top, thickness, closure, roughness=None, faults=None)`: a fold-style
  model from ResMill's own pieces, for the six fold styles of `fault_patterns` (`four_way`, `turtle`,
  `faulted_anticline`, `fold_belt`, `fault_bounded`, `low_relief`). The closure (`structure.closure`) plus its roughness
  (`structure.roughness`) is the structure, and the style's faults (`fold_faults`) are drawn on the fold without the
  roughness. Every value, seeds included, is an argument and nothing is drawn, so the model is a pure function of
  them; `roughness=None` and `faults=None` are a smooth fold without faults. Returns `{"kwargs": ..., "meta": ...}`:
  the keywords for `to_grdecl` (`structure`, `faults`) and the style, the asked area and height, the crest (a tilt moves
  it off the centre), the number of faults and their sets. New module, nothing existing changes.
  With `rim` (m) it also returns the simulation outline: `kwargs["outline"]` (the closure drawn, its ellipse, joined
  with the trap of the fold and with the one that came out with its roughness, each around the crest, and the crest's
  column, dilated by `rim` as `structure.outline` does) and `outline_columns` in `meta`; without `rim` nothing
  changes.
- `salt_traps.salt_flank_trap` and `subsalt_trap`: the two salt traps composed from `salt`'s pieces, as `fold_trap`
  composes the fold styles (every value, seeds included, is an argument; `{"kwargs": ..., "meta": ...}` back). A flank
  trap places the contact `wall_at` of the closure's half-width beyond the crest on the side away from the salt
  (`normal`, degrees), so the crest is cut off, centres a stock or wall (`body`: the keywords of `salt_body`, with
  `lobes`, `rough`, `hurst`, `overhang=(L, H)`) on it, and upturns the strata over a folding zone `width` wide: plain
  (`dip`, `thinning`, exponent held under `relief_cap`) or as a halokinetic sequence (`taper`, and the unconformity as
  the truncation `angle` through `truncation_cut`, or as the `loss` of the reservoir's thickness at the wall: a hook
  that pinches out needs a taper over 54 degrees, one that keeps the interval a `loss`). The contact's reference depth
  is the overhang's neck (`neck` of its height below the lifted beds' top) or `waist` m below the middle of the
  reservoir. Its faults are the `salt_flank` set, `radial_rate` per km of contact included. A subsalt trap puts the
  base of a salt sheet on the fold's trap, read on the rough surface at the shallowest column near its crest: `kind`
  `"truncated"` cuts the crest by `cut` of its relief, `"cover"` lies `margin` m above it; the base dips, is rugose
  (zero at the crest column) and carries the sheet's thickness. Both take `rim` (the outline cut of `fold_trap`) and
  `roughness`/`faults` dicts as `fold_trap` does. New module; `salt.sequence_dip` is the public name of the helper
  `_dip` that gave a sequence's dip, and `fold_traps.trap_outline` and `crest_cell` the shared parts of `fold_trap`'s
  outline (hashes of `fold_trap` for the six styles, with and without a rim, are as before).
- `to_grdecl(report=dict)` and `fault_seal.block_inputs`, `blocks_at`, `block_labels`: the fault blocks of a model for
  any fluid, without writing the grid again. `fault_blocks` needed the corner stack and the faces, which only
  `to_grdecl` holds, and its contacts depend on the fluid (`Capillary.delta_rho`), drawn after the model. `report` is
  filled with `fault_names`, `faults` (the seal's record of each fault, `[]` without a seal) and `block_inputs` (the
  tops' depth map, which columns hold rock, the cell edges a fault splits and the face records: a few arrays of map
  size); `blocks_at(inputs, dx, dy, capillary)` is `fault_blocks` on them and `block_labels(inputs)` gives every
  column its fault block. `fault_blocks` is their composition, `report=None` (the default) writes the same file:
  outputs are bit-identical to before.
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
- `drapes` (`ChannelLayer`): mud drapes on the bases of channel storeys
  (levels), as transmissibility multipliers on the faces between a
  channel-fill cell and the older-storey sand next to it, not as mud cells
  (a drape of 0.1-1.5 m is thinner than a cell). A drape covers part of an
  element's base (a median 60 %, under 5 % to over 90 % over 17 outcrops,
  Barton et al. 2010); in Barton's sector model recovery falls from 0.60
  without drapes to 0.53 at 60 % coverage and 0.20 at 90 % (his Fig. 2;
  Ruetten 2021: the same shape, at breakthrough). The multiplier is the
  thin barrier [1 + (t / h)(ks / kd - 1)]^-1 of a mud layer `t` thick
  (`h` the cell size, `ks` the harmonic mean permeability, `kd` the
  drape's, by default the FF mud's): it seals a vertical face and chokes a
  lateral face of a wide cell. `drapes` is a dict: `coverage`,
  `coverage_concentration` (k of the Beta each storey's share is drawn
  from; 0.71: continuous in a quarter of the storeys), `margin_bias`,
  `thickness`, `perm`, `hole_range_m` or `hole_range_widths`. Facies,
  porosity and permeability do not change, the drapes have a random stream
  of their own, and `to_grdecl` writes them as MULTX, MULTY and MULTZ
  (times a fault seal's; a later BOX / MULTX erases them where MULTIPLY
  'MULTX' keeps them). `sample_drapes` draws a reservoir's settings. A
  cell's storey is looked up once per flow event: same result, 8-10 times
  faster. Off by default: outputs are bit-identical to before.
- `distinct_events=True` (`ChannelLayer`): give every flow event its own
  (poro_mult, log_perm_offset) pair. Both are clipped at +-2 sd, so two
  events sometimes draw the same pair, and the engine's record (pair to
  level, how levee fading and `drapes` find a cell's storey) keeps the last
  one's level: in 80 anchor models 13 had cells of a wrong storey (0.45 %
  of the sand, 16 % in the worst). A repeated pair moves to the next
  free float32 poro_mult (1e-7 relative per repeat, no random number drawn). Compare runs
  made with the same setting. Off by default: outputs are bit-identical to
  before.
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
  kappa falls smoothly as the scale grows (about 350 stamps of 1.4 km and 3 m on 5.5 x 4 km, 3 seeds, an
  independent re-run: 0.82, 0.81, 0.79 and 0.76 at 0.03, 0.1, 0.2 and 0.4, 0.69 at 1, 0.65 at 3, 0.83 for ResMill's
  weight at m = 100; random stacking is 0.55; the first count of 0.81-0.85 at 0.2 or less was a little high). Default `None` keeps the old weight: outputs are
  bit-identical to before.
- `facies_props` (`LobeLayer`): rock by facies for lobes, keyed by facies code as in `ChannelLayer`.
  Without it the sand is the top `ntg` share of the lobe structure, whose porosity and permeability are
  standardized over all cells before the mud is zeroed, so the sand's average moves with `ntg` (porosity
  0.292, 0.274 and 0.256 for a requested 0.25 at `ntg` 0.25, 0.5 and 0.85), the mud has no rock (the
  deck's floors) and PERMZ is one `kzkx` per layer. With it the sand keeps exactly `poro_ave`,
  `perm_ave`, `poro_std` and `perm_std` at any `ntg`; the mud (`-1`) has its own `poro`, `log10_perm`
  and optional `poro_sd` / `log10_perm_sd`; `kvkh` entries give each facies' vertical to horizontal
  ratio per cell (`kvkh_mat`, written as PERMZ). Every cell is one facies with that facies' rock. Binary lobes
  leave a connected mud lattice 100-400 m wide around round sand cores in plan view, where Tanqua lobes
  fade from 85-100 % sand at the axis to 20-50 % at the fringe (Spychala et al. 2017): that fringe is the
  heterolithic facies of `interlobe_erosion`, below. Default `None`: outputs are bit-identical to before.
- `tree_ntg_stop=True` (`DeltaLayer`, with `bifurcate=True`): grow distributary networks in each
  generation until the layer holds its cumulative share of `NTGtarget` (sand cells, facies 1 or more,
  mouth bars included; generation g of `n_generations` asks for (g + 1) / `n_generations` of it, counted
  once over the layer's cells), `n_trees` being only the cap. The branching delta ignores `NTGtarget`,
  grows a fixed `n_trees` networks per generation and abandons every one but the last, mud-filling it
  with `mFFCHprop` (nothing at all is painted with the preset's 0), so at field scale it realizes 2-15 %
  sand whatever `n_trees` (1-12), `n_bifurcations` (16-128) or trunk width (70-450 m), and the mouth bars
  of the vanished networks float where they were. With the stop every network is stamped as sand, the
  older ones too, and each network's mouth bars are painted right after it so they count. A 48 x 48 x 12
  test delta ends at 0.053, 0.150 and 0.300 for targets of 0.05, 0.15 and 0.30; a 4 x 4 km, 36 m thick
  one at 0.501 for 0.5 (bars 4 x 1.5). The engine takes the stop as a hook, `after_tree(engine)`, called
  after each network: a True ends the level's networks and none is abandoned. Off by default: outputs
  are bit-identical.
- `interlobe_erosion` (`LobeLayer`, with `facies_props`): interlobe mud, continuous around each lobe at a low
  net-to-gross and eroded away at the axes at a high one, in cells that are one facies each. Each stamp is
  capped with mud `min((1 - f) t, M)` thick (`t` its thickness at the column, `f` the sand fraction of its margin,
  `facies_props[2]["ntg_floor"]`), so a lobe's sand fraction falls from `1 - M / t` where it is thick to `f` at its
  margin (Tanqua: axis 85-100 %, off-axis 50-85 %, fringe 20-50 %), and a younger stamp scours
  `interlobe_erosion` times its own thickness below its base, the mud in that reach, of whichever older
  cap, becoming its sand (sand on sand where it is thick, no cut at its margin; the deepest reach of any later
  stamp counts, so a sliver of a deposit between two stamps shields nothing). A cell is sand (3, half sand or more), the
  heterolithic fringe (2, a fifth to a half) or mud (-1), by what fills it, as channel, levee and floodplain cells are,
  with the rock and the kv/kh of its facies (the fringe has `poro`, `log10_perm`, spreads and `kvkh` of its own, 1-2 decades
  below the sand's in the Tanqua); nothing is mixed, so a cell's permeability does not depend on the cell's size. `M` is
  found by bisection so that the share of net cells (sand, and the fringe when its permeability is above 1 mD: the
  cells above 1 mD) is `ntg`, a warning and `interlobe["aim_missed"]` when the stack cannot go as low (cells as thick as
  the lobes, a fringe that is net). The mud of a cap half a cell thick or more fills cells; the mud in a sand cell, under
  half of it, is a thin cap and no cell: it is a vertical transmissibility barrier on the face of the cell nearest its
  middle, the thin-barrier factor of the mud drapes (`resmill.layers.drapes.thin_barrier`: the cap's thickness, the mud's
  permeability and the harmonic mean of the PERMZ of the two cells), written as MULTZ through the drapes' export path
  (`self.mult_z`), which multiplies into a fault seal. A cap of 0.3 cells of 1e-3 mD mud between sand of 300 mD gives a face of about 2e-5, and a
  stack of caps a set of compartments joined where the later lobes scoured the caps away. Facies 3, 2 and -1;
  `self.interlobe` holds the cap thickness, the share of contacts amalgamated, the realized `net_cells`, the mean sand
  share and the share of faces under a half. The option also clips the porosity decay of a stamp thinner than a
  cell or two at 1: a cell whose lower face lies below the stamp's base had a decay above 1, a ring of porosity above the
  design maximum 0.35 that shows as a small bright ring in a plan view (`clip_decay`, off in every other path).
  With cells of 0.3 m on lobes of 5 m, 40 x 30 columns, the cells above 1 mD are the 0.25, 0.5, 0.65 and 0.85 asked,
  at 1 m the last two, at 3 m none (cells as thick as the lobe). Default `None`: outputs are bit-identical.
- `max_levels=N` (`DeltaLayer`, with `tree_ntg_stop=True`): the layer is grown level by level instead of
  piling networks at `n_generations` levels. The stop of `tree_ntg_stop` with `n_trees` large puts 100 or
  more networks at each of a few levels, and since every plan view then holds the union of them (a
  network covers 3-6 % of a field-scale plan view) each is one solid fan with no interdistributary
  mud. With `max_levels`, `n_trees` networks (1) are grown at each of up to `max_levels` channel tops
  spread from the floor to the roof (both ends, the middle, the quarter points, ...: any `2**m + 1` of
  them are evenly spaced), until the layer holds `NTGtarget`; a plan view then shows the few networks
  that reach it as separate trees with mud between, and the net-to-gross lands at the target plus the
  last network's share. It replaces `n_generations` and `level_z`; the progradation follows each level's
  height. Default `None`: outputs are bit-identical.
- `mouth_bar_thickness_depths` (`DeltaLayer`, tree mode): every mouth bar's peak thickness in depths of its own
  tip's channel (a bar at a bifurcation: of the parent branch), whatever `mouth_bar_width_factor` makes its width. A bar
  is otherwise `mouth_bar_hw_ratio + mouth_bar_dw_ratio` of its half-width thick, so its thickness over the depth of
  its channel is that ratio times the channel's width per depth, and a tree's tips are narrower per depth than its
  trunk (a branch of discharge share q has width per depth `q**0.1` of the trunk's, then the taper): ratios worked out
  for the trunk (1.25 depths of an 8 m, 320 m trunk, Reynolds 1999's mouth bar over distributary channel) gave bars of
  0.74 tip depths (P10-P90 0.60-0.88, 102 tips at the anchors). The engine's `distal_tips` of a tree now carry the
  channel's depth as a sixth element. `None`, the default: outputs are bit-identical.
- `branch_levees=True` (`fluvial`, so `DeltaLayer` too; tree mode): `mLVwidth` and `mLVheight` are the
  trunk's, and a branch of discharge share q gets them times q**`width_exp` and q**`depth_exp`, its own
  width and depth ratios. Every branch otherwise has the trunk's levee, 60 m wide beside a 30 m
  distributary, and the levee sheets of a hundred overlapping networks fill the interdistributary area.
  Default `False`: outputs are bit-identical.
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
