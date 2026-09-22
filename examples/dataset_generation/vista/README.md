# Regenerating the dataset on TACC Vista (gg partition)

Same configs, same Sobol seed 42, same job list as the published dataset
(REPRODUCIBILITY.md); only the engine changes, to ResMill `9c0bd15`, which fixes
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
| run_delta.sh | 150,000 | 89.6 | 3,733 | 25.9 | 10 x 04:00 |
| **total** | 1,000,000 | | 7,861 | 54.6 | 88 node-hours requested |

gg node-hours are charged at 1/3 of an SU each: 55 node-hours of work
is about 18 SU (88 node-hours = 29 SU if every job ran to
its walltime limit; TACC charges actual run time).

## Steps

```bash
cd /work/08405/ilgar/vista/codes/ResMill_ls6 && git checkout 9c0bd15
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
