# Regenerating the dataset on TACC Vista (gg partition)

Same configs, same Sobol seed 42, same job list as the published dataset
(REPRODUCIBILITY.md); only the engine changes, to ResMill `6d1a956`, which fixes
the three fluvial-walker defects listed in CHANGELOG.md. Every row keeps its
parameters and seed, so downstream selections by (shard, index) stay valid once
the shards are combined the same way.

Vista gg nodes have 144 Grace cores and no SMT, so each job runs 144
single-thread ranks per node (ResMill has no internal parallelism; one process
per core is the maximum throughput). Costs below are measured on this engine on
2026-09-21 with warm processes and the node fully loaded, rows drawn uniformly
from the published parameter tables (see per-environment ranges in the job
headers):

| job | samples | s/volume | core-hours | node-hours | nodes x walltime |
|---|---|---|---|---|---|
| run_lobes.sh | 200,000 | 1.8 | 100 | 0.7 | 1 x 01:30 |
| run_pv_shoestring.sh | 100,000 | 3.4 | 94 | 0.7 | 1 x 01:30 |
| run_cb_labyrinth.sh | 100,000 | 12.7 | 353 | 2.4 | 2 x 02:00 |
| run_cb_jigsaw.sh | 150,000 | 14.6 | 608 | 4.2 | 3 x 02:30 |
| run_sh_distal.sh | 100,000 | 18.8 | 522 | 3.6 | 3 x 02:00 |
| run_sh_proximal.sh | 100,000 | 19.7 | 547 | 3.8 | 3 x 02:00 |
| run_meander_oxbow.sh | 100,000 | 68.5 | 1,903 | 13.2 | 6 x 03:30 |
| run_delta.sh (tree delta, config v2) | 150,000 | ~3 | ~125 | ~0.9 | 1 x 01:30 |
| **total** | 1,000,000 | | ~4,250 | ~30 | 49 node-hours requested |

gg node-hours are charged at 1/3 of an SU each: about 30 node-hours of work
is about 10 SU (49 node-hours = 16 SU if every job ran to its walltime
limit; TACC charges actual run time). The delta row is the tree delta
(`config_full_delta_v2.json`, 2 to 4 s a cube); the old avulsion delta cost
89.6 s a cube, 3,733 core-hours.

## Steps

```bash
cd /work/08405/ilgar/vista/codes/ResMill_ls6 && git checkout 6d1a956
cd examples/dataset_generation
for j in vista/run_*.sh; do sbatch "$j"; done        # 8 independent jobs
# when all eight have finished (logs/ show every rank's summary line):
python combine_shards.py --root $SCRATCH/resmill_dataset --target 256 --workers 32
python build_splits.py --root $SCRATCH/SiliciclasticReservoirs_v2 --out $SCRATCH/SiliciclasticReservoirs_v2/splits --seed 42 --train-frac 0.90 --val-frac 0.05
```

Output goes to `$SCRATCH/resmill_dataset/<env>/shard_r*_s*/` as set in each
config's `output_dir` (rank shards: facies, poro, perm, params.parquet), then to
the combined 256-shard layout. Failed samples, if any, are listed in
`failures_r*.jsonl` per rank (the published run lost about 0.01%).

## Throughput scaling

Per-volume cost is flat as the node fills (lobe 1.8 s, PV 4.5 to 5.5 s,
CB_LABYRINTH 14.5 to 15.4 s at 36, 72, 108 and 140 concurrent processes), so
144 single-thread ranks per node is the maximum throughput; ResMill has no
internal parallelism to trade against it.

## The same counts at 128 x 128 x 64

Simulated at 144 x 144 x 82 and cropped by the same margins, same 10 m cells and
1 m layers (a 1.28 km x 1.28 km x 64 m box). Budgets scaled as the ResBench
field generator does: channel event budgets by area (x3.24), delta generations
by linear size (x1.8), channel levels by thickness (x1.64), lobe unchanged.
Realized sand fractions stay close to the small volumes (MEANDER 0.43 vs 0.43,
delta 0.38 vs 0.41, lobe 0.44 vs 0.48, CB_JIGSAW 0.42 vs 0.36, PV 0.21 vs 0.23);
tune the budgets before a production run. Six real rows per environment (ten
for lobe and PV), warm processes, node loaded:

| environment | samples | s/volume | core-hours | node-hours |
|---|---|---|---|---|
| lobes | 200,000 | 9.7 | 539 | 3.7 |
| pv_shoestring | 100,000 | 26.9 | 747 | 5.2 |
| cb_labyrinth | 100,000 | 183.1 | 5,086 | 35.3 |
| cb_jigsaw | 150,000 | 214.6 | 8,942 | 62.1 |
| sh_distal | 100,000 | 341.9 | 9,497 | 66.0 |
| sh_proximal | 100,000 | 237.1 | 6,586 | 45.7 |
| meander_oxbow | 100,000 | 756.3 | 21,008 | 145.9 |
| delta | 150,000 | 693.7 | 28,904 | 200.7 |
| **total** | 1,000,000 | | **81,310** | **565** |

About 565 gg node-hours, 188 SU at the 1/3 charge factor, roughly 10x the
dataset size; MEANDER_OXBOW and delta are 62% of it at either size.
