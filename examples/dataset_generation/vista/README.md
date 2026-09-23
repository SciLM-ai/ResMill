# Regenerating the dataset on TACC Vista (v2)

Eight Slurm scripts, one per environment, for the `gg` partition (2 x 72 Grace
cores, 237 GB, no SMT), allocation `CHE23004`. Submit from a login node:

```bash
cd /work/08405/ilgar/vista/codes/ResMill_ls6 && git pull
cd examples/dataset_generation
for j in vista/run_*.sh; do sbatch "$j"; done
```

Every script refuses to run unless the checkout contains ResMill `d459b4b`
(the v2 configs and the aggradation-ratio sampler).

## What v2 is

Eight `config_full_<env>_v2.json` files (ResMill `d459b4b` or later):

- **128 x 128 x 64 cells, stored whole.** dx = dy = 10 m, dz = 1 m (lobes keep
  dx = 100 m), so 1280 x 1280 x 64 m, no crop. Training takes random
  64 x 64 x 32 windows; keep the z offset between 1 and 31 so a window never
  contains the engine's floor or roof, where the level ladder is anchored
  (bottom channel base on the floor, top channel top on the roof).
- **Channel depth 3 to 16 m, log-uniform** (v1: 3 to 10 m uniform).
- **Levels from a sampled aggradation ratio.** `nlevel` (channels) and
  `n_generations` (delta) come from the new sampler spec `levels_from_ratio`:
  r = level spacing / channel depth, uniform 0.7 to 1.4 per reservoir, and
  nlevel = round((64 - depth) / (r x depth)) + 1, so every column is spanned
  floor to roof with at most 0.4 depth of mud between levels (r > 1) or 0.3
  depth of erosion into the level below (r < 1). 4 to 25 levels result.
- **Event budgets x 2.56** (the plan-area ratio to the 800 m box): `ntime`
  per level 30 -> 77 (PV), 75 -> 192 (CB_LAB), 50 -> 128 (CB_JIG, SH), 150 ->
  384 (MEANDER). `ntime` is a cap; a level stops once it has its NTG share.
  Delivered / target NTG on the 144-volume preview: PV 1.03, CB_LABYRINTH 0.93, CB_JIGSAW 0.87, SH_DISTAL 0.90, SH_PROXIMAL 0.89, MEANDER 0.73, lobes 1.00. The tree
  delta ignores `NTGtarget` (0.15 realised on average, 0.20 in the 640 m
  cube). A ratio above 1 caps the reachable NTG at 1 / r, because the
  per-level target is split evenly across levels.
- **Entry scatter sampled.** `stdevCHsource`, the across-flow scatter of every
  channel's entry point, is log-uniform 5 to 300 m per reservoir (v1: fixed
  80 m, 1 m for MEANDER): 5 m is a nodal entry, 300 m spreads entries over
  the whole edge. With several sources each channel scatters around its own
  source.
- **Porosity texture.** `poro_noise_std` (0.05 to 0.15, relative) and
  `poro_noise_range` (2 to 8 cells laterally, a third of that vertically)
  multiply a correlated Gaussian field into every sand cell's porosity, with
  permeability following through the Kozeny-Carman slope. v1 channel
  bodies were a smooth upward-fining ramp with one multiplier per event.
- **Delta floor.** The lowest generation's channel base now sits on the
  floor like the channels; before, its top sat at 1 m and only a one-cell
  sand sliver showed in slice 0 (hidden in v1 by the z-crop).
- Delta `n_bifurcations` 8 to 32 (5 to 20 in the 800 m box of the 64-cube
  trials, scaled by the 1.6 x longer edge).
- Also new against v1: n_sources {1,1,2,3}, sampled `probAvulOutside`, the
  tree delta. Lobes keep the v1 ranges.

Preview: `/scratch/08405/ilgar/resmill_preview144/preview144_v5_perm.pdf`
(18 volumes per environment through this CLI, xy slices and xz / yz sections).

## Cost (measured 2026-09-22)

Per-volume times are the mean rank wall time of the 144-volume preview, 18
ranks per environment through `resmill.dataset.cli`, cold Numba JIT included,
with the engine of `d459b4b`: its Numba nearest-node search, smoother, curvature
and migration kernels made the fluvial environments 5 to 8 x cheaper than the
`5bf93ad` engine with output bit-identical (MEANDER 427 -> 56 s a volume).
The earlier full-node check found linear scaling to 144 ranks; the walltimes
carry a 1.3 x margin on top. Lobes need 1.4 GB per rank at this size, so the
lobe script runs 96 ranks per node; the fluvial environments stay under 0.5 GB.

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

About 33 gg node-hours, about 11 SU at the 1/3 charge factor. Node counts
target roughly 4 h of wall time; trade nodes for time freely (halve `-N`,
double `-t`). MEANDER is the long pole; on 3 nodes it takes about 3.6 h.

The ranks run with `env -u LD_PRELOAD`: TACC's XALT preload brings its own
libcrypto into rank 0 and pyarrow then fails to import (OPENSSL_3.3.0 not
found), which silently loses rank 0's samples.

Each rank buffers one shard in memory before writing it, and a 128 x 128 x 64
sample is 6 MB, so the configs use `shard_size` 32 (192 MB per rank); v1's
1000 would need 6 GB per rank and OOM every node.

## After the eight jobs: windows, combined shards, staging, splits

The raw shards hold whole 128 x 128 x 64 volumes. The training dataset is one
random 64 x 64 x 32 window per volume; the raw volumes stay on scratch as the
context around each window. Four deterministic steps, all from the repo root
with the ResMill env active:

```bash
cd examples/dataset_generation
# 1. one window per volume (origin from crop_seed 42 and the sample seed; z0 in 1..31),
#    same shard layout, ntg / poro_ave / perm_ave / caption recomputed on the window
python crop_windows.py --src $SCRATCH/resmill_dataset_v2 --dst $SCRATCH/resmill_dataset_v2_win64 --workers 96 --verify
# 2. eight 32-sample rank shards -> one combined shard of up to 256 samples, counts verified
python combine_shards.py --root $SCRATCH/resmill_dataset_v2_win64 --group 8 --workers 64
# 3. HuggingFace layout: <layer type>/shard_NNNN symlinks to the combined shards
python stage_dataset.py --src $SCRATCH/resmill_dataset_v2_win64 --dst $SCRATCH/SiliciclasticReservoirs_v2
# 4. 90 / 5 / 5 splits stratified by layer type, seed 42
python build_splits.py --root $SCRATCH/SiliciclasticReservoirs_v2 --out $SCRATCH/SiliciclasticReservoirs_v2/splits --seed 42 --train-frac 0.90 --val-frac 0.05
```

Every window's parquet row carries `crop_x0, crop_y0, crop_z0, source_shard,
source_row`, so the full volume around it is
`$SCRATCH/resmill_dataset_v2/<preset>/<source_shard>` row `source_row`.
Disk: raw volumes 5.8 TB, windows and their combined copy 0.7 TB each.

Output goes to `$SCRATCH/resmill_dataset_v2/<env>/` as set by each config's
`output_dir`. A 128 x 128 x 64 volume is 8 x the cells of a 64-cube, so the
dataset is about 8 x the v1 size on disk.
