# Regenerating the dataset on TACC Vista (v3)

Eight Slurm scripts, one per environment, for the `gg` partition (2 x 72 Grace
cores, 237 GB, no SMT), allocation `CHE23004`. Submit from a login node:

```bash
cd /work/08405/ilgar/vista/codes/ResMill_ls6 && git pull
cd examples/dataset_generation
for j in vista/run_*.sh; do sbatch "$j"; done
```

Every script refuses to run unless the checkout contains ResMill `5bf93ad`
(the v3 configs and the aggradation-ratio sampler).

## What v3 is

Eight `config_full_<env>_v3.json` files (ResMill `5bf93ad` or later):

- **128 x 128 x 64 cells, stored whole.** dx = dy = 10 m, dz = 1 m (lobes keep
  dx = 100 m), so 1280 x 1280 x 64 m, no crop. Training takes random
  64 x 64 x 32 windows; keep the z offset between 1 and 31 so a window never
  contains the engine's floor or roof, where the level ladder is anchored
  (bottom channel base on the floor, top channel top on the roof).
- **Channel depth 3 to 16 m, log-uniform** (v2: 3 to 10 m uniform).
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
  channel's entry point, is log-uniform 5 to 300 m per reservoir (v2: fixed
  80 m, 1 m for MEANDER): 5 m is a nodal entry, 300 m spreads entries over
  the whole edge. With several sources each channel scatters around its own
  source.
- **Porosity texture.** `poro_noise_std` (0.05 to 0.15, relative) and
  `poro_noise_range` (2 to 8 cells laterally, a third of that vertically)
  multiply a correlated Gaussian field into every sand cell's porosity, with
  permeability following through the Kozeny-Carman slope. v1 and v2 channel
  bodies were a smooth upward-fining ramp with one multiplier per event.
- **Delta floor.** The lowest generation's channel base now sits on the
  floor like the channels; before, its top sat at 1 m and only a one-cell
  sand sliver showed in slice 0 (hidden in v1 by the z-crop).
- Delta `n_bifurcations` 8 to 32 (v2: 5 to 20, scaled by the 1.6 x longer edge).
  Everything else is the approved v2 setting (n_sources {1,1,2,3}, sampled
  `probAvulOutside`, tree delta, lobes v1 ranges).

Preview: `/scratch/08405/ilgar/resmill_preview144/preview144_v5_perm.pdf`
(18 volumes per environment through this CLI, xy slices and xz / yz sections).

## Cost (measured 2026-09-22)

Per-volume times are the mean rank wall time of the 144-volume preview, 18
ranks per environment through `resmill.dataset.cli`, cold Numba JIT included.
The earlier full-node check found linear scaling to 144 ranks; the walltimes
carry a 1.3 x margin on top. Lobes need 1.4 GB per rank at this size, so the
lobe script runs 96 ranks per node; the fluvial environments stay under 0.5 GB.

| job | volumes | s per volume | core-hours | node-hours | nodes x walltime |
|---|---|---|---|---|---|
| run_lobes.sh | 200,000 | 7 | 378 | 3.9 | 1 x 05:30 |
| run_pv_shoestring.sh | 100,000 | 14 | 389 | 2.7 | 1 x 04:00 |
| run_cb_labyrinth.sh | 100,000 | 71 | 1,972 | 13.7 | 4 x 04:30 |
| run_cb_jigsaw.sh | 150,000 | 86 | 3,596 | 25.0 | 7 x 05:00 |
| run_sh_distal.sh | 100,000 | 126 | 3,489 | 24.2 | 7 x 04:30 |
| run_sh_proximal.sh | 100,000 | 107 | 2,972 | 20.6 | 6 x 04:30 |
| run_meander_oxbow.sh | 100,000 | 427 | 11,856 | 82.3 | 21 x 05:30 |
| run_delta.sh | 150,000 | 19 | 779 | 5.4 | 2 x 04:00 |
| **total** | 1,000,000 | | **25,431** | **178** | |

About 178 gg node-hours, about 59 SU at the 1/3 charge factor. Node counts
target roughly 4 h of wall time; trade nodes for time freely (halve `-N`,
double `-t`). MEANDER is the long pole; on 21 nodes it takes about 4 h.

The ranks run with `env -u LD_PRELOAD`: TACC's XALT preload brings its own
libcrypto into rank 0 and pyarrow then fails to import (OPENSSL_3.3.0 not
found), which silently loses rank 0's samples.

## After the eight jobs

```bash
cd examples/dataset_generation
python combine_shards.py --root $SCRATCH/resmill_dataset_v3 --target 256 --workers 32
python build_splits.py --root $SCRATCH/SiliciclasticReservoirs_v3 --out $SCRATCH/SiliciclasticReservoirs_v3/splits --seed 42 --train-frac 0.90 --val-frac 0.05
```

Output goes to `$SCRATCH/resmill_dataset_v3/<env>/` as set by each config's
`output_dir`. A 128 x 128 x 64 volume is 8 x the cells of a 64-cube, so the
dataset is about 8 x the v1 size on disk.
