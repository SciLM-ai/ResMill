# Generating the dataset on TACC Vista

Eight Slurm scripts, one per environment, for the `gg` partition (2 x 72 Grace
cores, 237 GB, no SMT), allocation `CHE23004`. Submit from a login node:

```bash
cd /work/08405/ilgar/vista/codes/ResMill_ls6 && git pull
cd examples/dataset_generation
for j in vista/run_*.sh; do sbatch "$j"; done
```

Every script refuses to run unless the checkout contains ResMill `27c48e4`
(the dataset configs and the aggradation-ratio sampler).

## What the configs describe

Eight `config_full_<env>.json` files, one per environment, sampled with Sobol
(seed 42) over the ranges in each file:

- **128 x 128 x 64 cells, stored whole.** dx = dy = 10 m, dz = 1 m (lobes keep
  dx = 100 m), so 1280 x 1280 x 64 m. The training dataset is one random
  64 x 64 x 32 window per volume, cut afterwards by `crop_windows.py`, with the
  window base between 1 and 31 so it never contains the engine's floor or roof,
  where the level ladder is anchored (bottom channel base on the floor, top
  channel top on the roof).
- **Channel depth 3 to 16 m, log-uniform.**
- **Levels from a sampled aggradation ratio.** `nlevel` (channels) and
  `n_generations` (delta) come from the sampler spec `levels_from_ratio`:
  r = level spacing / channel depth, uniform 0.7 to 1.4 per reservoir, and
  nlevel = round((64 - depth) / (r x depth)) + 1, so every column is spanned
  floor to roof with at most 0.4 depth of mud between levels (r > 1) or 0.3
  depth of erosion into the level below (r < 1). 4 to 25 levels result.
- **Event budgets scaled to the box.** `ntime` per level is the preset's budget
  for an 800 x 800 m box times 2.56, the plan-area ratio. `ntime` is a cap; a
  level stops once it has its NTG share. Delivered / target NTG on a 144-volume
  preview: PV 1.03, CB_LAB 0.93, CB_JIG 0.87, SH_DIST 0.90, SH_PROX 0.89,
  MEANDER 0.73, lobes 1.00. The tree delta has no NTG target (0.15 realised on
  average). A ratio above 1 caps the reachable NTG at 1 / r, because the
  per-level target is split evenly across levels.
- **Entry points.** `n_sources` in {1, 1, 2, 3} entry positions on the upstream
  edge; `probAvulOutside` sampled per preset; `stdevCHsource`, the across-flow
  scatter of every channel's entry point, log-uniform 5 to 300 m per reservoir
  (5 m is a nodal entry, 300 m spreads entries over the whole edge; with several
  sources each channel scatters around its own source).
- **Porosity texture.** `poro_noise_std` (0.05 to 0.15, relative) and
  `poro_noise_range` (2 to 8 cells laterally, a third of that vertically)
  multiply a correlated Gaussian field into every sand cell's porosity, with
  permeability following through the Kozeny-Carman slope, on top of the
  upward-fining ramp and the one-multiplier-per-event variability.
- **Delta.** Distributary-tree mode: discharge split at every bifurcation,
  width ~ Q^0.5, depth ~ Q^0.4, mouth bars, a prograding discharge-dependent
  front, `n_bifurcations` 8 to 32, the bottom generation's base on the floor.
- **Shards.** `shard_size` 32: each rank buffers one shard in memory before
  writing it, and a 128 x 128 x 64 sample is 6 MB, so 32 samples = 192 MB per
  rank (1000 would be 6 GB per rank and out-of-memory on every node).

Preview: `/scratch/08405/ilgar/resmill_preview144/preview144_v5_perm.pdf`
(18 volumes per environment through this CLI, xy slices and xz / yz sections).

## Cost (measured 2026-09-22 and 2026-09-23)

Per-volume times are the mean rank wall time of a 144-volume preview, 18 ranks
per environment through `resmill.dataset.cli`, cold Numba JIT included, with the
engine of `adf8c29` (its Numba nearest-node search, smoother, curvature and
migration kernels; MEANDER 56 s a volume). Walltimes carry a 1.3 x margin.
Lobes need 1.4 GB per rank at this size, so the lobe script runs 96 ranks per
node; the fluvial environments stay under 0.6 GB.

| job | volumes | s per volume | core-hours | node-hours | nodes x walltime |
|---|---|---|---|---|---|
| run_lobes.sh | 200,000 | 6 | 350 | 3.6 | 1 x 05:00 |
| run_pv_shoestring.sh | 100,000 | 5 | 136 | 0.9 | 1 x 01:30 |
| run_cb_labyrinth.sh | 100,000 | 13 | 353 | 2.4 | 1 x 03:30 |
| run_cb_jigsaw.sh | 150,000 | 16 | 662 | 4.6 | 2 x 03:00 |
| run_sh_distal.sh | 100,000 | 21 | 581 | 4.0 | 2 x 03:00 |
| run_sh_proximal.sh | 100,000 | 18 | 503 | 3.5 | 1 x 05:00 |
| run_meander_oxbow.sh | 100,000 | 56 | 1,553 | 10.8 | 3 x 05:00 |
| run_delta.sh | 150,000 | 11 | 471 | 3.3 | 1 x 04:30 |
| **total** | 1,000,000 | | **4,608** | **33** | |

The full run took 33 gg node-hours, 11 SU at the 1/3 charge factor: PV 0:24,
SH_DIST 1:52, CB_LAB 2:01, CB_JIG 2:08, delta 2:31, SH_PROX 3:18, lobes 3:35,
MEANDER 3:47 of wall time, 1,000,000 samples, no failures. Trade nodes for time
freely (halve `-N`, double `-t`).

The ranks run with `env -u LD_PRELOAD`: TACC's XALT preload brings its own
libcrypto into rank 0 and pyarrow then fails to import (OPENSSL_3.3.0 not
found), which silently loses rank 0's samples.

## After the eight jobs: windows, combined shards, staging, splits

The raw shards hold whole 128 x 128 x 64 volumes. The training dataset is one
random 64 x 64 x 32 window per volume; the raw volumes stay on scratch as the
context around each window. Four deterministic steps, all from the repo root
with the ResMill env active:

```bash
cd examples/dataset_generation
# 1. one window per volume (origin from crop_seed 42 and the sample seed; z0 in 1..31),
#    same shard layout, ntg / poro_ave / perm_ave / caption recomputed on the window
python crop_windows.py --src $SCRATCH/resmill_dataset --dst $SCRATCH/resmill_dataset_win64 --workers 96 --verify
# 2. 256 combined shards per family (about 390 to 780 samples each), counts verified
python combine_shards.py --root $SCRATCH/resmill_dataset_win64 --target 256 --workers 96
# 3. HuggingFace layout: <layer type>/shard_NNNN, hard-linked copies of the combined shards, plus the cards
python stage_dataset.py --src $SCRATCH/resmill_dataset_win64 --dst $SCRATCH/SiliciclasticReservoirs
# 4. 90 / 5 / 5 splits stratified by layer type, seed 42
python build_splits.py --root $SCRATCH/SiliciclasticReservoirs --out $SCRATCH/SiliciclasticReservoirs/splits --seed 42 --train-frac 0.90 --val-frac 0.05
```

Every window's parquet row carries `crop_x0, crop_y0, crop_z0, source_shard,
source_row`, so the full volume around it is
`$SCRATCH/resmill_dataset/<preset>/<source_shard>` row `source_row`.
Disk: raw volumes 5.8 TB, windows and their combined copy 0.7 TB each; the
staged directory adds nothing (hard links).

Output of the jobs goes to `$SCRATCH/resmill_dataset/<env>/` as set by each
config's `output_dir`.
