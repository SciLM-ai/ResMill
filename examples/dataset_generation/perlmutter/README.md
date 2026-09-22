# Regenerating the v2 dataset on NERSC Perlmutter

Same jobs as `../vista/`, with Perlmutter headers: `-C cpu`, `-q regular`
(premium is charged double), 128 ranks per node with `--cpus-per-task=2` as
the v1 run used, `--licenses=cfs,SCRATCH`, and `-A REPLACE_WITH_YOUR_ALLOCATION`
to fill in. Each script activates `$WORK/conda_envs/resmill`, the env the v1
scripts used; if yours lives elsewhere, edit that line. Every script refuses
to run unless the checkout contains ResMill `3e71977` (final engine and v2
configs), so `git pull` first.

Costs are the Vista measurements (Grace cores) with a 1.3x allowance for the
Milan cores and another 1.3x in the walltime; MEANDER carries a further 10%
for the fresh belts of the v2 config and the delta uses the tree delta's cost:

| job | samples | core-hours | node-hours | nodes x walltime |
|---|---|---|---|---|
| run_lobes.sh | 200,000 | 130 | 1.0 | 1 x 02:00 |
| run_pv_shoestring.sh | 100,000 | 123 | 1.0 | 1 x 02:00 |
| run_cb_labyrinth.sh | 100,000 | 459 | 3.6 | 2 x 03:00 |
| run_cb_jigsaw.sh | 150,000 | 791 | 6.2 | 3 x 03:30 |
| run_sh_distal.sh | 100,000 | 679 | 5.3 | 3 x 03:00 |
| run_sh_proximal.sh | 100,000 | 711 | 5.6 | 3 x 03:00 |
| run_meander_oxbow.sh | 100,000 | 2,708 | 21.2 | 8 x 04:30 |
| run_delta.sh | 150,000 | 433 | 3.4 | 2 x 03:00 |
| **total** | 1,000,000 | **6,034** | **47** | |

About 47 Perlmutter node-hours at the regular QOS. Perlmutter has no XALT
preload, so the `env -u LD_PRELOAD` of the Vista scripts is not needed.

```bash
cd $WORK/codes/ResMill && git pull                     # or wherever the repo is
cd examples/dataset_generation
for j in perlmutter/run_*.sh; do sbatch "$j"; done     # 8 independent jobs
# when all eight have finished:
python combine_shards.py --root $SCRATCH/resmill_dataset --target 256 --workers 32
python build_splits.py --root $SCRATCH/SiliciclasticReservoirs_v2 --out $SCRATCH/SiliciclasticReservoirs_v2/splits --seed 42 --train-frac 0.90 --val-frac 0.05
```

Output goes to `$SCRATCH/resmill_dataset/<env>/` as set by each config's
`output_dir`, the same layout as v1.
