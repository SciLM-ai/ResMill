# ResMill

Rule-based 3D geological reservoir modeling library.

## What This Project Does

ResMill generates synthetic 3D geological models using rule-based and stochastic methods, bends them into structural shapes (folds, faults, erosional unconformities) at export time, and writes Petrel/Eclipse corner-point grids (GRDECL). It targets subsurface reservoir modeling for oil & gas, groundwater, and carbon storage applications.

## Problem It Solves

Building realistic 3D geological models typically requires expensive commercial software and extensive manual input. ResMill provides a Python-native, pip-installable alternative that generates geologically plausible reservoir models programmatically and exports them into the commercial ecosystem (Petrel, Eclipse, tNavigator, OPM Flow, ResInsight all read the GRDECL output). Users define layer types and parameters; the library handles the physics-based geometry and property modeling.

## Supported Geology Types

- **Lobe layers** — Turbidite lobe deposition with compensational stacking, Bouma sequences, and upthinning
- **Gaussian layers** — Sequential Gaussian simulation (SGS) with spatial correlation for heterogeneous sand/shale distributions
- **Channel layers** — ALLUVSIM-ported fluvial engine (meandering, migration, avulsion, neck cutoffs, levees, crevasse splays); one `ChannelLayer` class plus importable presets (`PV_SHOESTRING`, `CB_JIGSAW`, `CB_LABYRINTH`, `SH_DISTAL`, `SH_PROXIMAL`, `MEANDER_OXBOW`)
- **Delta layers** — `DeltaLayer` drives the same engine with delta-tuned defaults (`DELTA_FAN`)

## Architecture

- `Layer` base class defines grid geometry (nx, ny, nz, dimensions, depth, dip, kzkx) plus node grids `X, Y` (nx+1, ny+1) and top/base depth surfaces `z1, z2`
- Each layer type inherits from `Layer` and implements `create_geology()` to populate 3D property arrays
- `Reservoir` stacks multiple layers vertically (listed top to bottom), validating compatibility; arrays concatenate deepest layer first so the global k index increases upward (0.2.0+)
- `resmill/structure.py`: composable `Structure` deformation fields (`anticline`, `syncline`, `dome`, `ramp`, `fault`, `surface`); plain callables, 2-D arrays, and scalars coerce automatically
- `resmill/export.py`: `to_grdecl()` (also methods on `Layer`/`Reservoir`) builds a corner-point grid at export time (geology itself stays in flat stratigraphic space) and writes a self-contained GRDECL; `to_pyvista()` returns a `pyvista.ExplicitStructuredGrid` (optional `viz` extra)
- `resmill/plotting.py`: index-space viewers (`plot_slices`, `plot_cube_slices`) plus `plot_section()`, the true-depth cross-section of the deformed grid
- All property outputs are numpy arrays shaped `(nx, ny, nz)`

## Key Commands

- Install: `pip install -e ".[dev]"` (add `viz` extra for pyvista)
- Test: `pytest tests/` (~3 min; the Numba fluvial engine JIT-compiles on first run; `test_alluvsim_parity.py` self-skips without the external Alluvsim checkout, and the xtgeo round-trip in `test_export.py` self-skips without xtgeo)
- Tutorial: `jupyter notebook notebooks/tutorial.ipynb` (section 6 covers structure & export)
- Worked structural examples (run from the repo root; `--show` opens an interactive 3-D corner-point view, needs pyvista): `python examples/anticline_meander.py` (four-way-closure trap with a meandering river), `python examples/angular_unconformity.py` (folded layers beveled flat under an undeformed cover)
- README figures: `python docs/make_readme_figures.py`

## Conventions

- Array ordering: `(nx, ny, nz)` with `meshgrid(..., indexing='ij')`
- Vertical: depth is positive down; within every layer the k index increases **upward** (k=0 is the layer base); `Reservoir` arrays keep that convention across the stack. The GRDECL exporter flips per layer to Eclipse order (K=1 at the top, counting down)
- Properties: `poro_mat` (porosity, 0-1), `perm_mat` (permeability, mD), `active` (0/1 sand mask — not simulation activity; the exporter's ACTNUM is all-active except structurally collapsed cells)
- Facies: every layer type exposes `facies` with the Alluvsim 6-class codes (-1=FF, 0=FFCH, 1=CS, 2=LV, 3=LA, 4=CH). Channel/delta use all six; lobe and Gaussian mark sand as 3 and background as -1. `LobeLayer.lobe_id` holds the per-lobe index (1..N). `Reservoir` has no `facies` attribute; assemble per layer (the exporter does)
- Physics parameters go in `create_geology()`, not `__init__()` — init is grid-only
- Structure is applied at export/plot time, never baked into property arrays; `Layer.dip` and per-layer `kzkx` (PERMZ = kzkx × PERMX) only take effect there
- Channel internals use Numba JIT and are prefixed with `_` (private)

## Parameter units — gotchas

- **Length is in meters, everywhere.** `x_len, y_len, z_len, top_depth, dx, dy, dz` are meters; the fluvial engine hardcodes `g = 9.8 m/s²`, so all `mCH*`, `mLV*`, `mCS*`, `mdistMigrate` (channel/delta), `dh_ave/dh_std/r_ave/r_std` (lobe), and `facies_filter/sand_filter` (gaussian) are also meters. Layers convert physical inputs to cell units internally via `self.dx/dy/dz`.
- **`perm_ave` and `perm_std` for `LobeLayer` and `GaussianLayer` are in log10(mD) space**, not linear mD. See the docstrings in `layers/lobe.py` and `layers/gaussian.py`. Typical sensible ranges:
  - `perm_ave`: `[0, 4]` → mean perm spans 1 to 10,000 mD
  - `perm_std`: `[0.1, 1.5]` → log10-std (factor-of-1.3 to factor-of-30 spread)
  - Passing linear-mD values (e.g. `perm_ave=500`) makes the internal `10**perm_mat` overflow float64 and the output cells saturate to the on-disk clip ceiling (60000 mD). `poro_ave` and `poro_std` stay in linear [0, 1] units.
- **Structure fields are vertical shifts in meters, positive down**: `anticline(amplitude=25)` lifts the crest 25 m (the field is -25 there). `azimuth` is degrees clockwise from +x (the lobe/channel convention). Put `fault` traces on grid lines (multiples of dx/dy) for clean vertical fault faces.
- **Trap vs fold**: `anticline()` is a cylindrical fold, open at both ends; a closed structural trap (four-way dip closure) is `dome(amplitude, radius, aspect, azimuth)` — `aspect > 1` elongates it into a doubly plunging anticline. See `examples/anticline_meander.py`.
- **Unconformities need no erosion surface**: pass `structure=` as a list with one entry per layer (top to bottom, `None` = undeformed). Layers deform independently and the exporter clamps older interfaces to any younger layer base that cuts them ("younger truncates older"), so folded layers under undeformed layers come out beveled flat at the contact, with eroded cells collapsed and written ACTNUM 0. `erode_above=`/`erode_below=` are only for present-day surfaces at the very top/base. See `examples/angular_unconformity.py`.
- `GaussianLayer` and `LobeLayer` zero out poro/perm in shale cells (`* active`); channel/delta give shale realistic low values from `FACIES_PROPS`. For simulation-ready exports of gaussian/lobe models use `to_grdecl(..., poro_floor=, perm_floor=)`.
