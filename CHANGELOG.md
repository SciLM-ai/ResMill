# Changelog

## Unreleased

### Added

- `DeltaLayer(bifurcate=True)`: a generation is a distributary tree, not an
  avulsion history. Discharge is split at every bifurcation (`split_frac_lo`
  to `split_frac_hi` to the new branch), width follows `Q ** width_exp` and
  depth `Q ** depth_exp` (Leopold-Maddock), a segment gives off at most
  `max_splits_per_branch` branches so splits cascade down the network,
  branches launch at `branch_angle_mean` +- `branch_angle_sd` degrees and
  ease back toward the regional slope, run at most
  `branch_length_scale * diagonal * sqrt(Q)` before ending in a mouth bar,
  and join another branch they run into (`merge_branches`). `n_trees`
  networks per generation. Off by default; the published dataset is
  unchanged. `examples/dataset_generation/config_full_delta_v2.json` is the
  dataset config for it. `layer.tree_branches` lists every segment.
- `dome()` gains `aspect` and `azimuth` (elongated four-way closures);
  `examples/anticline_meander.py` and `examples/angular_unconformity.py`.

### Fixed

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
