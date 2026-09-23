---
license: cc-by-4.0
language:
- en
size_categories:
- 1M<n<10M
task_categories:
- image-segmentation
- image-classification
- text-to-image
- other
pretty_name: Siliciclastic Reservoirs (1M Synthetic 3D Geological Cubes)
tags:
- geology
- earth-sciences
- reservoir
- synthetic
- voxel
- 3d
- flow-matching
- diffusion
- generative-modeling
- subsurface
- petroleum
- groundwater
- carbon-storage
- conditional-generation
configs:
- config_name: default
  data_files:
  - split: train
    path: splits/train.parquet
  - split: validation
    path: splits/validation.parquet
  - split: test
    path: splits/test.parquet
---

# SiliciclasticReservoirs

**1,000,000 synthetic 3D siliciclastic-reservoir geology cubes** generated from rule-based sedimentological simulations (turbidite lobes + 6 fluvial-channel architectures + a distributary-tree delta). Each cube is a `(64, 64, 32)` voxel window cut at a random position out of a `(128, 128, 64)` simulated volume, so every cube is a piece of a larger reservoir and its faces cut through sand bodies the way the boundaries of a modelled interval do.

Designed to train **conditional generative models** (flow matching, diffusion, etc.) of subsurface geology under interpretable physical conditioning.

## Quick stats

- **1,000,000 samples** across 8 reservoir architectures
- Cube shape: `(64, 64, 32)` voxels — `(x, y, z)`, `z` is depth (0 = base)
- 4 voxel arrays per sample: binary facies, 6-class facies, porosity, permeability
- Compact slim parquet (9–12 cols) and full reproducibility parquet (34–108 cols)
- Captions for text-conditioning experiments
- Every cube holds sand (realised net-to-gross > 0)
- Fully reproducible — every volume regenerable from `(seed, params)`, every window from its `crop_*` columns

## File structure

```
SiliciclasticReservoirs/
├── README.md                   ← this file
├── DATASHEET.md                ← Datasheet for Datasets (Gebru et al. 2018)
├── splits/                     ← deterministic 90/5/5 split, stratified by layer_type
│   ├── train.parquet           ← 900,000 rows
│   ├── validation.parquet      ←  50,000 rows
│   └── test.parquet            ←  50,000 rows
├── lobe/                       ← 200,000 samples, 792 shards
│   ├── shard_0000/
│   │   ├── facies.npy          (n, 64, 64, 32) int8 — binary 0/1
│   │   ├── facies_alluvsim.npy (n, 64, 64, 32) int8 — 6-class -1..4
│   │   ├── poro.npy            (n, 64, 64, 32) float16 — porosity in [0, 0.5]
│   │   ├── perm.npy            (n, 64, 64, 32) float16 — permeability in mD [0, 60000]
│   │   ├── params_slim.parquet ← training conditioning (use this)
│   │   └── params.parquet      ← full physics for reproducibility
│   ├── shard_0001/
│   └── …
├── channel_pv_shoestring/      ← 100,000 samples, 396 shards
├── channel_cb_labyrinth/       ← 100,000 samples, 396 shards
├── channel_cb_jigsaw/          ← 150,000 samples, 612 shards
├── channel_sh_distal/          ← 100,000 samples, 396 shards
├── channel_sh_proximal/        ← 100,000 samples, 396 shards
├── channel_meander_oxbow/      ← 100,000 samples, 432 shards
└── delta/                      ← 150,000 samples, 594 shards
```

Shards hold up to 256 samples (about 200 MB each); the last shards of a family are smaller.

## Layer types

| `layer_type` | architecture | samples |
|---|---|---|
| `lobe` | turbidite lobe (deep-water gravity-flow deposit) | 200,000 |
| `channel:PV_SHOESTRING` | paleo-valley shoestring sandstone | 100,000 |
| `channel:CB_JIGSAW` | channel-and-bar bodies, jigsaw connectivity | 150,000 |
| `channel:CB_LABYRINTH` | channel-and-bar bodies, labyrinthine connectivity | 100,000 |
| `channel:SH_DISTAL` | distal sheet-sandstone | 100,000 |
| `channel:SH_PROXIMAL` | proximal sheet-sandstone | 100,000 |
| `channel:MEANDER_OXBOW` | multi-storey meander-belt with oxbow plugs | 100,000 |
| `delta` | distributary-tree delta with mouth bars | 150,000 |

These follow Pyrcz & Deutsch's standard fluvial-architecture taxonomy ("User Guide to the Alluvsim Program", 2004).

## What varies between samples

Every family is a Sobol sweep (seed 42) of its architecture's parameters. Beyond the classic knobs (net-to-gross target, sinuosity, avulsion probabilities, mud-plug fraction, levee and splay geometry, lobe size and aspect), the sweep covers:

- **Channel size**: depth 3 to 16 m, log-uniform, with the width-to-depth ratio sampled per family (8 to 26), so channels span 2.5 to 41 cells wide and 3 to 16 cells thick in a 32-cell-tall cube.
- **Vertical stacking**: the number of channel levels follows a sampled aggradation ratio (level spacing over channel depth, 0.7 to 1.4), so stacked channels range from cutting into each other to leaving up to 0.4 depth of mud between them, and the simulated column is always spanned floor to roof.
- **Entry points**: one to three river entries on the upstream edge, and a sampled entry scatter from a nodal point (all channels through one spot) to entries spread over the whole edge.
- **Porosity texture**: a correlated Gaussian texture inside every sand body on top of the upward-fining ramp, with permeability following through a Kozeny-Carman coupling.
- **Delta**: distributary trees with discharge split at every bifurcation (width ∝ Q^0.5, depth ∝ Q^0.4), branches that may re-merge, mouth bars at every terminus and bifurcation, a discharge-dependent prograding front, and 4 to 21 stacked generations.

## Voxel arrays per sample

| file | dtype | shape | description |
|---|---|---|---|
| `facies.npy` | `int8` | `(N, 64, 64, 32)` | **binary facies**: 0 = mud, 1 = sand. Primary signal for generative training. |
| `facies_alluvsim.npy` | `int8` | `(N, 64, 64, 32)` | **6-class facies** preserving architectural sub-detail: -1 = FF (overbank fines), 0 = FFCH (mud plug), 1 = CS (crevasse splay), 2 = LV (levee), 3 = LA (lateral-accretion / point bar / mouth bar), 4 = CH (active channel). |
| `poro.npy` | `float16` | `(N, 64, 64, 32)` | porosity, range `[0, 0.5]` |
| `perm.npy` | `float16` | `(N, 64, 64, 32)` | permeability in millidarcies (mD), range `[0, 60000]`. **Use `log10(perm + 1e-3)` for ML** — perm spans 5+ decades. |

All four arrays are row-aligned within a shard (sample `i` of `facies` corresponds to row `i` of the parquets and to sample `i` of all other arrays).

## Slim parquet (`params_slim.parquet`) — the training conditioning

This is **what you train on** as conditioning. It carries the geological parameters that vary across samples.

### Universal columns (all 8 layer types)

| column | dtype | description |
|---|---|---|
| `layer_type` | str | one of the 8 strings above |
| `caption` | str | human-readable description, e.g. *"Paleo-valley shoestring reservoir with sinuosity 1.55, channel depth 9.7 m, …"*. Useful for text-conditioned variants. |
| `ntg` | float | realised net-to-gross of the stored cube, range `(0, 1]` |
| `poro_ave` | float | realised mean porosity over the cube's sand cells |
| `perm_ave` | float | realised mean of `log10(perm in mD)` over the cube's sand cells |
| `azimuth` | float | regional flow direction, compass-CW degrees `[0, 360)` |
| `width_cells` | float | characteristic horizontal extent in cells. Lobe = 2 × semi-minor axis. Channel/delta = full channel width. |
| `depth_cells` | float | characteristic vertical extent in cells. Lobe = thickness `dh_ave`. Channel/delta = `mCHdepth`. |

### Family-specific columns (NULL outside family)

| column | dtype | family | description |
|---|---|---|---|
| `asp` | float | lobe | ellipse aspect ratio (major/minor), `[1.0, 2.5]` |
| `mCHsinu` | float | channel + delta | sinuosity (path / straight-line), `[1.05, 1.85]` |
| `probAvulInside` | float | channel + delta | per-event in-belt avulsion probability, `[0, 0.8]` (recorded for the delta but not used by its tree mode) |
| `mFFCHprop` | float | channel + delta | abandoned-channel mud-plug fraction, `[0, 0.7]` |
| `trunk_length_fraction` | float | delta | proximal-trunk fraction, `[0.1, 0.5]` |

Total: 9 cols for lobe rows, 11 for channel rows, 12 for delta rows. Rows from different families share the universal columns and have NULL in non-applicable family columns.

## Full parquet (`params.parquet`) — for reproducibility

34 columns for lobe rows, 87 for channel rows, 108 for delta rows: every sampled physics parameter, the per-event statistical knob widths (`stdev*`), the engine event budgets (`ntime*`), the realised statistics, and the **window provenance**:

| column | description |
|---|---|
| `seed` | per-sample RNG seed of the simulated volume |
| `crop_x0`, `crop_y0`, `crop_z0` | origin of this cube inside its `(128, 128, 64)` volume; `crop_nx`, `crop_ny`, `crop_nz` = 64, 64, 32 |
| `crop_seed` | seed of the window draw (42) |
| `source_shard`, `source_row` | the rank shard and row of the volume in the generation output |

A volume is regenerated bit-for-bit from `(seed, params)` with the open-source engine (see its `REPRODUCIBILITY.md`), and the cube is `volume[crop_x0:crop_x0+64, crop_y0:crop_y0+64, crop_z0:crop_z0+32]`. The window origin is `numpy.random.default_rng([crop_seed, seed])`: x and y uniform over 0..64, z uniform over 1..31, redrawn from the same stream while the window has no sand.

## Splits

`splits/{train,validation,test}.parquet` define a deterministic **90 / 5 / 5** partition stratified by `layer_type` (each split has all 8 architectures in proportion). Generated with seed 42.

Each row is `(layer_type, shard_dir, sample_idx)` pointing into one of the data shards.

## Cell-physical-size note

Cube shape is identical (64×64×32) but the physical cell size differs by family:

| family | dx = dy | dz | physical extent |
|---|---|---|---|
| `lobe` | 100 m | 1 m | 6.4 km × 6.4 km × 32 m |
| `channel:*` and `delta` | 10 m | 1 m | 640 m × 640 m × 32 m |

For training in cell-units, this difference is hidden inside `width_cells` / `depth_cells`. The model can ignore physical scale and just train on the cube as-is.

## Loading example (PyTorch)

```python
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import torch
from torch.utils.data import IterableDataset

class ReservoirShardDataset(IterableDataset):
    """One worker walks an assigned subset of shards, yields (cube, params)."""
    def __init__(self, root: Path, split: str = "train"):
        self.root = Path(root)
        # Read split index
        self.index = pq.read_table(self.root / f"splits/{split}.parquet").to_pylist()

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        my_rows = (self.index if worker is None
                   else self.index[worker.id::worker.num_workers])
        cur_shard = None; cur_data = None
        for row in my_rows:
            shard_dir = self.root / row["shard_dir"]
            if shard_dir != cur_shard:
                cur_shard = shard_dir
                cur_data = {
                    "facies": np.load(shard_dir / "facies.npy", mmap_mode="r"),
                    "poro":   np.load(shard_dir / "poro.npy",   mmap_mode="r"),
                    "perm":   np.load(shard_dir / "perm.npy",   mmap_mode="r"),
                    "params": pq.read_table(
                        shard_dir / "params_slim.parquet").to_pylist(),
                }
            i = row["sample_idx"]
            yield {
                "facies": np.asarray(cur_data["facies"][i], dtype=np.int8),
                "poro":   np.asarray(cur_data["poro"][i],   dtype=np.float32),
                "perm":   np.asarray(cur_data["perm"][i],   dtype=np.float32),
                "params": cur_data["params"][i],
            }

ds = ReservoirShardDataset("/path/to/SiliciclasticReservoirs", split="train")
loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=4)
```

## Recommended preprocessing for flow-matching training

- **`facies` → `[0, 1]` float**: just `cube.astype(np.float32)`
- **`poro` → standardized**: `(poro - 0.15) / 0.10` keeps it roughly N(0, 1)-ish for sand cells
- **`perm` → log-perm**: `np.log10(np.maximum(perm, 1e-3))`, then maybe standardize
- **`layer_type`** → categorical embedding (8 classes). Use `layer_type.split(":")[0]` for family-level (3 classes) or full string for fine-grained (8 classes).
- **Other slim columns** → continuous embeddings, with `null` → learned sentinel or per-family mask.

## Geological caveats

- **Cubes are windows of larger volumes.** Sand bodies are cut by all six faces of a cube exactly as a modelled reservoir interval cuts through geology; a channel truncated at the base of a cube continues below it in the simulated volume. The window never contains the simulated floor or roof.
- **Fluvial channels split but do not re-merge**, except in the delta, whose distributary branches may rejoin. The braided / jigsaw look in CB_LABYRINTH / CB_JIGSAW comes from heavily overlapping channel stamps, not from explicit braid topology.
- **Per-cell poro/perm** follow a Walker-1992 upward-fining ramp per channel event, a per-event Kozeny-Carman-coupled draw, a per-realization "regional rock quality" multiplier, and a correlated Gaussian texture inside sand; they are not conditioned to any real reservoir.
- **FACIES_PROPS base values** are conservative midpoints from fluvial-reservoir literature (Pyrcz & Deutsch 2002, Allen 1965, Walker 1992), not from a specific analog.
- **Net-to-gross is realised, not prescribed.** Sheet and meander families reach about 0.7 to 0.9 of their sampled target in the simulated volume; the delta has no target. Condition on the realised `ntg` of the cube.

See `DATASHEET.md` for full documentation.

## Citation

```bibtex
@misc{SiliciclasticReservoirs_2026,
  author       = {Anonymous},
  title        = {{SiliciclasticReservoirs}: 1M Synthetic 3D Reservoir Geology Cubes for Conditional Generative Modeling},
  year         = {2026},
  publisher    = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/AnonymouScientist/SiliciclasticReservoirs}}
}
```

## License

[Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/). Use, share, modify, redistribute — just attribute the source.

## Contact / Issues

File issues at the dataset repository on HuggingFace.
