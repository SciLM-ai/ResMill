# Regenerating the dataset on NERSC Perlmutter (v3)

Same eight jobs as `../vista/`, with Perlmutter headers: `-C cpu`, `-q regular`
(premium is charged double), 128 ranks per node with `--cpus-per-task=2` as
the v1 run used, `--licenses=cfs,SCRATCH`, and `-A REPLACE_WITH_YOUR_ALLOCATION`
to fill in. Each script activates `$WORK/conda_envs/resmill`, the env the v1
scripts used; if yours lives elsewhere, edit that line. Every script refuses
to run unless the checkout contains ResMill `a8bfbfd`, so `git pull` first.

```bash
cd $WORK/codes/ResMill && git pull                     # or wherever the repo is
pip install -e ".[dataset]"                            # only if the env is new
cd examples/dataset_generation
for j in perlmutter/run_*.sh; do sbatch "$j"; done     # 8 independent jobs
```

## What v3 is

Eight `config_full_<env>_v3.json` files (ResMill `a8bfbfd` or later):

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
  Delivered / target NTG on the 144-volume preview: PV 1.01, CB_LAB 0.94,
  CB_JIG 0.89, SH_DIST 0.88, SH_PROX 0.91, MEANDER 0.72, lobes 1.00. The tree
  delta ignores `NTGtarget` (0.15 realised on average, 0.20 in the 640 m
  cube). A ratio above 1 caps the reachable NTG at 1 / r, because the
  per-level target is split evenly across levels.
- Delta `n_bifurcations` 8 to 32 (v2: 5 to 20, scaled by the 1.6 x longer edge).
  Everything else is the approved v2 setting (n_sources {1,1,2,3}, sampled
  `probAvulOutside`, tree delta, lobes v1 ranges).

Preview: `/scratch/08405/ilgar/resmill_preview144/preview144_v5_perm.pdf`
(18 volumes per environment through this CLI, xy slices and xz / yz sections).

## Cost

The Vista measurements (Grace cores) with a 1.3 x allowance for the Milan
cores and another 1.3 x in the walltime. Perlmutter nodes have 512 GB, so the
1.4 GB lobe ranks fit at 128 per node. Perlmutter has no XALT preload, so the
`env -u LD_PRELOAD` of the Vista scripts is not needed.

| job | volumes | s per volume | core-hours | node-hours | nodes x walltime |
|---|---|---|---|---|---|
| run_lobes.sh | 200,000 | 7 | 491 | 3.8 | 1 x 05:00 |
| run_pv_shoestring.sh | 100,000 | 13 | 459 | 3.6 | 1 x 05:00 |
| run_cb_labyrinth.sh | 100,000 | 62 | 2,246 | 17.5 | 5 x 05:00 |
| run_cb_jigsaw.sh | 150,000 | 85 | 4,610 | 36.0 | 10 x 05:00 |
| run_sh_distal.sh | 100,000 | 122 | 4,391 | 34.3 | 9 x 05:00 |
| run_sh_proximal.sh | 100,000 | 101 | 3,633 | 28.4 | 8 x 05:00 |
| run_meander_oxbow.sh | 100,000 | 422 | 15,221 | 118.9 | 30 x 05:30 |
| run_delta.sh | 150,000 | 18 | 975 | 7.6 | 2 x 05:00 |
| **total** | 1,000,000 | | **32,025** | **250** | |

About 250 Perlmutter node-hours at the regular QOS.

## After the eight jobs

```bash
cd examples/dataset_generation
python combine_shards.py --root $SCRATCH/resmill_dataset_v3 --target 256 --workers 32
python build_splits.py --root $SCRATCH/SiliciclasticReservoirs_v3 --out $SCRATCH/SiliciclasticReservoirs_v3/splits --seed 42 --train-frac 0.90 --val-frac 0.05
```

Output goes to `$SCRATCH/resmill_dataset_v3/<env>/` as set by each config's
`output_dir`. A 128 x 128 x 64 volume is 8 x the cells of a 64-cube, so the
dataset is about 8 x the v1 size on disk.
