# Reproducibility Guide

How the **Siliciclastic Reservoirs** dataset (1,000,000 synthetic 3-D reservoir
volumes and one 64 x 64 x 32 training window per volume, hosted on HuggingFace)
is produced from this engine, end to end and deterministically.

The dataset is at
**[`AnonymouScientist/SiliciclasticReservoirs`](https://huggingface.co/datasets/AnonymouScientist/SiliciclasticReservoirs)** (CC-BY-4.0).

The configs are `examples/dataset_generation/config_full_<env>.json`, one per
environment; the Sobol master seed is 42. Every path below assumes the repo root,
the `resmill` environment (`pip install -e ".[dataset]"`) and `$SCRATCH` set.

---

## 1. Quickstart

```bash
git clone https://anonymous.4open.science/r/ResMill-7377
cd ResMill
pip install -e ".[dev,dataset]"
pytest tests/            # about a minute
```

One volume through the engine:

```python
import resmill as rm
from resmill.layers.channel import PV_SHOESTRING

layer = rm.ChannelLayer(nx=128, ny=128, nz=64, x_len=1280, y_len=1280, z_len=64, top_depth=0)
layer.create_geology(seed=42, **PV_SHOESTRING)
print(layer.facies.shape, layer.active.mean())
```

---

## 2. The dataset

| environment | config | volumes |
|---|---|---|
| lobes | `config_full_lobes.json` | 200,000 |
| pv_shoestring | `config_full_pv_shoestring.json` | 100,000 |
| cb_labyrinth | `config_full_cb_labyrinth.json` | 100,000 |
| cb_jigsaw | `config_full_cb_jigsaw.json` | 150,000 |
| sh_distal | `config_full_sh_distal.json` | 100,000 |
| sh_proximal | `config_full_sh_proximal.json` | 100,000 |
| meander_oxbow | `config_full_meander_oxbow.json` | 100,000 |
| delta | `config_full_delta.json` | 150,000 |

Every volume is 128 x 128 x 64 cells at dx = dy = 10 m and dz = 1 m (lobes
dx = dy = 100 m), stored whole. The training dataset is one random
64 x 64 x 32 window per volume. `examples/dataset_generation/vista/README.md`
describes the sampled design in detail: log-uniform channel depth 3 to 16 m,
the number of levels from a sampled aggradation ratio so every column is
spanned floor to roof, event budgets scaled to the box, one to three entry
points with a sampled entry scatter, correlated porosity texture, the
distributary-tree delta.

---

## 3. Producing it

### 3a. Generate the volumes

Eight independent Slurm jobs, one per environment, each running
`python -m resmill.dataset.cli examples/dataset_generation/config_full_<env>.json`
with one rank per core; rank r handles jobs r, r + world, r + 2 world, ... of the
shuffled Sobol job list and writes `shard_rXXXX_sNNNNNN` directories of
`shard_size` = 32 samples to the config's `output_dir`,
`$SCRATCH/resmill_dataset/<env>/`.

Launch scripts with measured costs: `examples/dataset_generation/vista/`
(TACC Vista, 33 gg node-hours for the whole run) and
`examples/dataset_generation/perlmutter/` (NERSC Perlmutter). Both READMEs
give the submission commands. Each script refuses to run on a checkout that
predates the configs it needs.

A rank shard holds

```
shard_r0000_s000000/
  facies.npy           (N, 128, 128, 64) int8     binary sand / not sand
  poro.npy             (N, 128, 128, 64) float16  porosity
  perm.npy             (N, 128, 128, 64) float16  permeability, mD
  facies_alluvsim.npy  (N, 128, 128, 64) int8     six-class facies (-1 FF, 0 FFCH, 1 CS, 2 LV, 3 LA, 4 CH)
  params.parquet       N rows: seed, layer_type, every sampled parameter, realised ntg / poro_ave / perm_ave, caption
  params_slim.parquet  N rows: the per-family whitelist of resmill/dataset/schemas.py
```

### 3b. Cut the training windows

```bash
cd examples/dataset_generation
python crop_windows.py --src $SCRATCH/resmill_dataset --dst $SCRATCH/resmill_dataset_win64 --workers 96 --verify
```

One 64 x 64 x 32 window per volume, into a second shard tree with the same
shard names and row order. The origin comes from
`numpy.random.default_rng([crop_seed, seed])` with `crop_seed` 42 and the
sample's own seed: x0 and y0 uniform over 0 to 64, z0 uniform over 1 to 31, so
the window never contains the engine's floor or roof. `ntg`, `poro_ave`,
`perm_ave` and the caption are recomputed on the window with the same code the
generator uses on a volume (`resmill.dataset.generate.realized_stats`,
`resmill.dataset.captions.caption_for`); everything else in the row is copied,
and `crop_x0, crop_y0, crop_z0, crop_nx, crop_ny, crop_nz, crop_seed,
source_shard, source_row` locate the window in its volume. `--verify` re-reads
random windows against the volumes.

### 3c. Combine the rank shards

```bash
python combine_shards.py --root $SCRATCH/resmill_dataset_win64 --group 8 --workers 64
```

Eight consecutive rank shards (lex-numeric order, rank 0 first) become one
`combined_shard_NNNN` of up to 256 samples under `<env>_combined/`, arrays and
parquets concatenated in that order. The script checks the total sample count
per environment and that sample 0 of `combined_shard_0000` equals sample 0 of
`shard_r0000_s000000`.

### 3d. Stage the HuggingFace layout

```bash
python stage_dataset.py --src $SCRATCH/resmill_dataset_win64 --dst $SCRATCH/SiliciclasticReservoirs
```

One directory per layer type (`lobe`, `channel_pv_shoestring`, ...,
`channel_meander_oxbow`, `delta`) holding `shard_NNNN` symlinks to the combined
shards, plus the dataset card and datasheet from
`examples/dataset_generation/dataset_card/`.

### 3e. Splits

```bash
python build_splits.py --root $SCRATCH/SiliciclasticReservoirs --out $SCRATCH/SiliciclasticReservoirs/splits --seed 42 --train-frac 0.90 --val-frac 0.05
```

`train.parquet`, `validation.parquet`, `test.parquet`, one row per sample with
`(layer_type, shard_dir, sample_idx)`, 90 / 5 / 5 stratified by layer type,
deterministic with the seed.

Then `hf upload AnonymouScientist/SiliciclasticReservoirs . --repo-type=dataset`
from the staged directory.

---

## 4. Reproducing a single sample bit-for-bit

Every row of `params.parquet` carries the seed and the full parameter set. To
regenerate a volume:

```python
import numpy as np, pyarrow.parquet as pq
import resmill as rm

row = pq.read_table("$SCRATCH/resmill_dataset/delta/shard_r0000_s000000/params.parquet").to_pylist()[3]

NOT_ENGINE = {"layer_type", "preset", "caption", "ntg", "requested_ntg", "poro_ave", "perm_ave",
              "r_ave_m", "r_ave_cells", "r_major_m", "r_major_cells", "dh_ave_m", "dh_ave_cells",
              "mCHdepth_m", "mCHdepth_cells", "mCHwidth_m", "mCHwidth_cells", "width_cells", "depth_cells"}
kwargs = {k: v for k, v in row.items() if k not in NOT_ENGINE and v is not None}
seed = int(kwargs.pop("seed"))

layer = rm.DeltaLayer(nx=128, ny=128, nz=64, x_len=1280.0, y_len=1280.0, z_len=64.0, top_depth=5000.0)
np.random.seed(seed)
layer.create_geology(**kwargs)
# layer.facies == facies_alluvsim.npy[3]; (layer.facies >= 1) == facies.npy[3];
# poro_mat / perm_mat match poro.npy / perm.npy after the float16 cast.
```

The grid arguments are the `grid` section of the environment's config. A
window is `volume[x0:x0+64, y0:y0+64, z0:z0+32]` with the `crop_*` columns of
its row, and `resmill.dataset.generate.realized_stats` gives its `ntg`,
`poro_ave` and `perm_ave`.

---

## 5. Validation / QA

```bash
python examples/dataset_generation/plot_dataset.py $SCRATCH/resmill_dataset/delta --workers 32 --limit 100
python examples/dataset_generation/plot_dataset_stats.py $SCRATCH/resmill_dataset/delta
```

The first writes facies, porosity and permeability pictures of the first 100
samples of a shard tree, the second per-environment histograms of NTG,
porosity, permeability and geometry plus the slim-column correlation heatmap.

---

## 6. Engine architecture (high level)

- **`resmill/layers/_fluvial.py`** — the fluvial engine: AR(2) streamline walks, several entry points, avulsion inside and outside, migration with neck cutoffs, level aggradation, per-event Kozeny-Carman draws, and the distributary-tree mode of the delta. Its Python-level loops are Numba kernels (`_movwinsmooth`, curvature, bank velocity).
- **`resmill/layers/_genchannel.py`** — Numba kernel painting one streamline's U-shape, with the fused nearest-node search and sub-node refinement; writes per-cell `depth_norm` for the upward-fining ramp.
- **`resmill/layers/_genabandoned.py`**, **`_calc_levee.py`**, **`_calc_lobe_splay.py`**, **`_make_cutoff.py`** — abandoned-channel mud plugs, levees, splays and mouth bars, neck-cutoff geometry.
- **`resmill/layers/channel.py`** — `ChannelLayer`, the presets, `FACIES_PROPS`, and the property assignment (facies base values, upward-fining ramp, per-event multipliers, per-reservoir multipliers, correlated porosity texture).
- **`resmill/layers/delta.py`** — `DeltaLayer`: `n_generations` independent single-level runs merged by facies rank, bottom generation anchored on the floor, mouth bars at the distal tips.
- **`resmill/layers/lobe.py`** — `LobeLayer`: stamped turbidite lobes with a correlated Gaussian porosity field.

The dataset pipeline:

- **`resmill/dataset/sampling.py`** — Sobol / LHS / grid / uniform job list with the `shared`, `jitter`, `fraction_of`, `linear_of`, `inverse_of` and `levels_from_ratio` specs.
- **`resmill/dataset/generate.py`** — `generate_sample(job, grid_cfg)` and `realized_stats`.
- **`resmill/dataset/io.py`** — `ShardWriter`, the per-rank shard writer (4 npy + 2 parquet, written atomically).
- **`resmill/dataset/cli.py`** — the Slurm rank-stripe entry point.
- **`resmill/dataset/schemas.py`**, **`captions.py`** — slim-column whitelist and caption templates per layer family.
- **`examples/dataset_generation/`** — `crop_windows.py`, `combine_shards.py`, `stage_dataset.py`, `build_splits.py`, the `vista/` and `perlmutter/` launch scripts, `run_dataset.py` (a multiprocessing driver for small runs without Slurm), and the plotting tools.

---

## 7. Determinism contract

1. Sobol draws: `qmc.Sobol(scramble=True, seed=section_seed)` with `section_seed = (master_seed + 1) * 10007 + lt_id`.
2. Job list shuffle: `np.random.default_rng(master_seed).permutation(total_n)`; per-sample seeds from `np.random.default_rng(section_seed)`.
3. Per-sample geometry: `np.random.seed(sample_seed)` before `create_geology`; Numba kernels are pure functions.
4. Windows: origin from `default_rng([crop_seed, sample_seed])`.
5. Combining, staging and splitting are order-preserving and seeded.

A re-run with the same master seed and configs produces the same 1,000,000
volumes in the same order, the same windows, the same combined shards and the
same splits.

---

## 8. Citation

```bibtex
@misc{siliciclastic_reservoirs_2026,
  author       = {Anonymous},
  title        = {{Siliciclastic Reservoirs}: 1M Synthetic 3D Reservoir Geology Cubes for Conditional Generative Modeling},
  year         = {2026},
  publisher    = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/AnonymouScientist/SiliciclasticReservoirs}}
}

@software{resmill_engine_2026,
  author       = {Anonymous},
  title        = {{ResMill}: Rule-Based Synthetic 3D Reservoir Geology Engine},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://anonymous.4open.science/r/ResMill-7377}}
}
```

The engine builds on the streamline-based fluvial architecture by Pyrcz & Deutsch:
- Pyrcz, M. J. (2003). *Stochastic Surface-based Modeling of Turbidite Lobes*. PhD dissertation, University of Alberta.
- Pyrcz, M. J., & Deutsch, C. V. (2002). *User Guide to the Alluvsim Program*. Centre for Computational Geostatistics.

---

## 9. License

- Engine code (this repository): MIT
- Dataset on HuggingFace: CC-BY-4.0

File issues at [`anonymous.4open.science/r/ResMill-7377`](https://anonymous.4open.science/r/ResMill-7377).
