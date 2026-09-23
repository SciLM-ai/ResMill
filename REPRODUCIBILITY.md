# Reproducibility Guide

Documents how the public **Siliciclastic Reservoirs** dataset (1,000,000 synthetic 3D reservoir cubes, hosted on HuggingFace) was produced from this engine. Every byte of the dataset can be regenerated bit-for-bit from this codebase plus the saved configs and seeds.

The dataset is at:
**[`AnonymouScientist/SiliciclasticReservoirs`](https://huggingface.co/datasets/AnonymouScientist/SiliciclasticReservoirs)** (CC-BY-4.0)

The configs that produced it live in `examples/dataset_generation/config_full_*.json`. The Sobol master seed is `42`. Pin the engine to git tag (TBD) for exact reproduction.

---

## 1. Quickstart

```bash
git clone https://anonymous.4open.science/r/ResMill-7377
cd ResMill
pip install -e ".[dev]"
```

Verify the engine works on a single sample:

```python
import resmill as rm
from resmill.layers.channel import PV_SHOESTRING

layer = rm.ChannelLayer(nx=64, ny=64, nz=32,
                        x_len=640, y_len=640, z_len=32, top_depth=0)
layer.create_geology(seed=42, **PV_SHOESTRING)

print(f"facies range: {layer.facies.min()}..{layer.facies.max()}")
print(f"poro range:   {layer.poro_mat.min():.3f}..{layer.poro_mat.max():.3f}")
print(f"perm range:   {layer.perm_mat.min():.1f}..{layer.perm_mat.max():.0f} mD")
print(f"realized NTG: {layer.active.mean():.3f}")
```

Expected output: a `(64, 64, 32)` PV-shoestring facies cube with non-zero NTG.

The 9 channel-layer pytest cases verify the engine end-to-end:
```bash
pytest tests/test_channel.py
```

---

## 2. Reproducing the published 1M dataset

Total: 184 node-hours on an HPC CPU partition + ~50 GB scratch I/O.

### 2a. Run the 8 SLURM jobs

`examples/dataset_generation/run_*.sh` are the production scripts. Each one:
- Loads the `resmill` conda env
- Runs `srun python -m resmill.dataset.cli config_full_<preset>.json`
- Each MPI rank handles its rank-stripe of the 1M-job list and writes shards to `$SCRATCH/.../<output_dir>/shard_r{rank}_s{shard_idx}/`

Submit in any order (independent jobs):
```bash
cd examples/dataset_generation
for sh in run_lobes.sh run_pv_shoestring.sh run_cb_labyrinth.sh \
          run_cb_jigsaw.sh run_sh_distal.sh run_sh_proximal.sh \
          run_meander_oxbow.sh run_delta.sh; do
    sbatch "$sh"
done
```

Per-preset breakdown (matches the published dataset):

| preset | nodes × walltime | NH (premium=2×) | samples |
|---|---|---|---|
| `run_lobes.sh` | 4 × 2h | 16 | 200,000 |
| `run_pv_shoestring.sh` | 4 × 2h | 16 | 100,000 |
| `run_cb_labyrinth.sh` | 8 × 2h | 32 | 100,000 |
| `run_cb_jigsaw.sh` | 8 × 3h | 48 | 150,000 |
| `run_sh_distal.sh` | 8 × 3h | 48 | 100,000 |
| `run_sh_proximal.sh` | 8 × 3h | 48 | 100,000 |
| `run_meander_oxbow.sh` | 8 × 4h | 64 | 100,000 |
| `run_delta.sh` | 16 × 10h | 192 (used ~33 in practice) | 150,000 |

Outputs land at `$SCRATCH/resmill_dataset/<output_dir>/` (configurable via the `output_dir` field of each `config_full_*.json`).

### 2b. Combine shards into the final 256-shard-per-preset layout

Rank-shards (512–2048 per preset, one per MPI rank) are too granular for typical ML pipelines. Combine them deterministically into 256 shards per preset:

```bash
python examples/dataset_generation/combine_shards.py \
    --root $SCRATCH/resmill_dataset \
    --target 256 \
    --workers 32
```

This concatenates contiguous groups of rank-shards in deterministic lex-numeric order (rank 0..N-1 → `combined_shard_0000`, etc.). Samples within each shard preserve original row order; npy arrays + parquets stay row-aligned. Output: `<input_preset>_combined/combined_shard_NNNN/`.

The script verifies after each preset that:
- Total combined sample count == total rank-shard sample count
- Sample 0 of `combined_shard_0000` matches sample 0 of rank `shard_r0000` for all 4 npy arrays + the slim parquet first row

### 2c. Generate the train/validation/test splits

```bash
python examples/dataset_generation/build_splits.py \
    --root $SCRATCH/SiliciclasticReservoirs \
    --out  $SCRATCH/SiliciclasticReservoirs/splits \
    --seed 42 --train-frac 0.90 --val-frac 0.05
```

Produces `train.parquet`, `validation.parquet`, `test.parquet` — one row per sample, columns `(layer_type, shard_dir, sample_idx)`. Stratified by `layer_type`, deterministic with the master seed.

### 2d. Stage the HuggingFace upload directory

```bash
# Symlinks to combined data + the staged READMEs + splits
DST=$SCRATCH/SiliciclasticReservoirs
SRC=$SCRATCH/resmill_dataset
declare -A MAP=(
  [lobe]=lobes_combined
  [channel_pv_shoestring]=channels_pv_shoestring_combined
  [channel_cb_labyrinth]=channels_cb_labyrinth_combined
  [channel_cb_jigsaw]=channels_cb_jigsaw_combined
  [channel_sh_distal]=channels_sh_distal_combined
  [channel_sh_proximal]=channels_sh_proximal_combined
  [channel_meander_oxbow]=channels_meander_oxbow_combined
  [delta]=delta_combined
)
for hf in "${!MAP[@]}"; do
    mkdir -p "$DST/$hf"
    for shard in "$SRC/${MAP[$hf]}"/combined_shard_*; do
        idx=$(basename "$shard" | sed 's/combined_shard_//')
        ln -sf "$shard" "$DST/$hf/shard_$idx"
    done
done
```

Then upload with `hf upload AnonymouScientist/SiliciclasticReservoirs . --repo-type=dataset`.

---

## 3. Reproducing a single sample bit-for-bit

Every sample's `params.parquet` row carries the seed and full physics parameters. To regenerate:

```python
import pyarrow.parquet as pq
import numpy as np
import resmill as rm

# Pick a sample from a shard
row = pq.read_table(
    "$SCRATCH/.../delta_combined/combined_shard_0000/params.parquet"
).to_pylist()[42]   # sample index 42

# Strip non-engine keys
ENGINE_IGNORE = {"layer_type", "preset", "caption", "ntg", "requested_ntg",
                 "poro_ave", "perm_ave",
                 "r_ave_m", "r_ave_cells", "r_major_m", "r_major_cells",
                 "dh_ave_m", "dh_ave_cells",
                 "mCHdepth_m", "mCHdepth_cells", "mCHwidth_m", "mCHwidth_cells",
                 "width_cells", "depth_cells"}
kwargs = {k: v for k, v in row.items() if k not in ENGINE_IGNORE and v is not None}
seed = int(kwargs.pop("seed"))

# Re-instantiate the layer (config grid kwargs are at examples/dataset_generation/config_full_<preset>.json)
nx, ny, nz, x_len, y_len, z_len = 80, 80, 50, 800.0, 800.0, 50.0  # delta grid
layer = rm.DeltaLayer(nx=nx, ny=ny, nz=nz,
                      x_len=x_len, y_len=y_len, z_len=z_len, top_depth=0)
np.random.seed(seed)
layer.create_geology(**kwargs)

# Crop the engine output the same way the dataset writer does:
# crop_spec from config grid: x: 8:-8, y: 8:-8, z: 9:-9
facies_cropped = layer.facies[8:-8, 8:-8, 9:-9]   # (64, 64, 32)
poro_cropped   = layer.poro_mat[8:-8, 8:-8, 9:-9]
perm_cropped   = layer.perm_mat[8:-8, 8:-8, 9:-9]
```

`facies_cropped` will match `facies.npy[42]` from that shard byte-for-byte (subject to float16 cast for poro/perm).

---

## 4. Validation / QA

Quick sanity check on any shard dir:

```bash
python examples/dataset_generation/plot_dataset.py \
    $SCRATCH/resmill_dataset/delta \
    --workers 32 --limit 100
```

Produces `facies_binary_pictures/`, `poro_pictures/`, `perm_pictures/`, `facies_alluvsim_pictures/` PNGs for the first 100 samples — useful for visual QA.

Per-layer-type summary stats:

```bash
python examples/dataset_generation/plot_dataset_stats.py \
    $SCRATCH/resmill_dataset/delta
```

Produces `stats_pictures/stats_delta.png` with NTG / poro / perm / geometry histograms + slim-column correlation heatmap.

For a full QA cycle on 100-sample subsets per preset (used during development):
```bash
bash examples/dataset_generation/replot_all_test_subsets.sh
```

---

## 5. Engine architecture (high level)

For developers who want to understand or extend the engine:

- **`resmill/layers/_fluvial.py`** — main fluvial-engine class. AR(2) walks (Pyrcz-Sun streamline model), avulsion-inside (anchored to `mCHazi`), neck cutoff, level aggradation, per-event K-C draws.
- **`resmill/layers/_genchannel.py`** — Numba-JIT kernel that paints one streamline's U-shape; writes per-cell `depth_norm` for upward-fining ramp.
- **`resmill/layers/_genabandoned.py`** — abandoned-channel mud plug (FFCH).
- **`resmill/layers/_calc_levee.py`** — natural-levee (LV) painter.
- **`resmill/layers/_calc_lobe_splay.py`** — crevasse-splay (CS) painters.
- **`resmill/layers/_make_cutoff.py`** — neck cutoff geometry.
- **`resmill/layers/channel.py`** — `ChannelLayer` (drives the engine, hosts `FACIES_PROPS`, applies per-event ramp + per-realization mults in `_finalize_facies_table`).
- **`resmill/layers/delta.py`** — `DeltaLayer` (subclass driving `n_generations` independent fluvial sims merged by max-facies takeover; per-cell aux fields propagate from the winning generation).
- **`resmill/layers/lobe.py`** — `LobeLayer` (separate non-fluvial implementation: stamped ellipsoidal turbidite lobes + Gaussian poro field).

The dataset pipeline:

- **`resmill/dataset/sampling.py`** — Sobol/LHS/grid/uniform JobList builder with shared/jitter/derived spec types.
- **`resmill/dataset/generate.py`** — `generate_sample(job, grid_cfg)` → `(facies, poro, perm, facies_alluvsim, meta)`.
- **`resmill/dataset/io.py`** — `ShardWriter` per-rank shard writer (writes 4 npy + 2 parquet atomically).
- **`resmill/dataset/cli.py`** — SLURM rank-stripe entry point.
- **`resmill/dataset/schemas.py`** — slim parquet column whitelist per layer family.
- **`resmill/dataset/captions.py`** — natural-language caption template per layer family.

---

## 6. Determinism contract

Every step is deterministic given the master seed:

1. `qmc.Sobol(scramble=True, seed=section_seed).random(N)` — bit-identical bytes per call.
2. `section_seed = (master_seed + 1) * 10007 + lt_id` — deterministic.
3. JobList shuffle: `np.random.default_rng(master_seed).permutation(total_n)`.
4. Per-sample seeds: `np.random.default_rng(section_seed).integers(...)`.
5. Per-sample geometry: `np.random.seed(sample_seed)` before each `create_geology` call.
6. Numba kernels: pure functions, no shared state.

A re-run with the same master seed produces the same 1,000,000 samples in the same order. The `combine_shards.py` and `build_splits.py` scripts are also deterministic.

---

## 7. Citation

If you use this dataset or engine, please cite:

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

## 8. License

- **Engine code (this repository)**: MIT (or pick whatever you prefer — propose Apache-2.0 if you want patent grant)
- **Dataset on HuggingFace**: CC-BY-4.0

---

## Issues / Contact

File issues at [`anonymous.4open.science/r/ResMill-7377`](https://anonymous.4open.science/r/ResMill-7377).


---

## 3. The v2 dataset (2026-09)

v2 is a redesign, not a re-run of v1: 128 x 128 x 64 volumes (dx = dy = 10 m,
dz = 1 m, lobes dx = 100 m) stored whole, with random 64 x 64 x 32 training
windows taken later; channel depth 3 to 16 m log-uniform; the number of levels
from a sampled aggradation ratio (level spacing over channel depth, 0.7 to 1.4)
so every column is spanned floor to roof; per-level event budgets scaled by the
plan area; several entry points and a sampled entry scatter; a distributary-tree
delta; correlated porosity texture inside sand; and the fixed engine (entries on
the grid boundary, walk clipped by the grid). The configs are
`examples/dataset_generation/config_full_<env>_v2.json`; the Sobol seed and the
sample counts are v1's. Launch scripts and measured costs: `vista/` (TACC Vista,
about 33 gg node-hours in total) and `perlmutter/` (NERSC). The original
`run_*.sh` in this directory are the Perlmutter scripts that produced v1 and are
kept for the record.
