# Dataset generation

Everything under this directory produces the Siliciclastic Reservoirs dataset:
the eight `config_full_<env>.json` files, the Slurm launch scripts for TACC
Vista (`vista/`) and NERSC Perlmutter (`perlmutter/`), and the post-processing
tools. `REPRODUCIBILITY.md` at the repo root walks through the whole sequence;
`vista/README.md` describes the sampled design and the measured cost.

## What a run produces

`python -m resmill.dataset.cli CONFIG.json` under `srun` gives every rank a
stripe of the shuffled Sobol job list; each rank writes its own shards to the
config's `output_dir`, so ranks never coordinate:

```
output_dir/
  shard_r0000_s000000/
    facies.npy           (N, 128, 128, 64) int8     binary sand / not sand
    poro.npy             (N, 128, 128, 64) float16  porosity
    perm.npy             (N, 128, 128, 64) float16  permeability, mD
    facies_alluvsim.npy  (N, 128, 128, 64) int8     six-class facies
    params.parquet       N rows: seed, layer_type, every sampled parameter,
                         realised ntg / poro_ave / perm_ave, caption
    params_slim.parquet  N rows, the per-family column whitelist
  shard_r0000_s000001/
  ...
```

`layer_type` is `lobe`, `channel:<PRESET>` or `delta`. Samples whose parameter
combination raises inside the engine are skipped and logged to
`failures_r{rank}.jsonl`; the production run had none.

## Running

- **Production**: `vista/run_<env>.sh` and `perlmutter/run_<env>.sh`, one
  `sbatch` each, see their READMEs.
- **Without Slurm**: `python run_dataset.py CONFIG.json --workers N` shards the
  same job list across a multiprocessing pool. For a smoke test, copy a config
  and lower its `count`.
- **Memory**: a rank buffers one shard before writing it; at 128 x 128 x 64 a
  sample is 6 MB, so `shard_size` is 32 (192 MB per rank). Lobe ranks need
  1.4 GB each and run 96 per node on Vista's 237 GB nodes.

## After the jobs

```bash
python crop_windows.py  --src $SCRATCH/resmill_dataset --dst $SCRATCH/resmill_dataset_win64 --workers 96 --verify
python combine_shards.py --root $SCRATCH/resmill_dataset_win64 --target 256 --workers 96
python stage_dataset.py --src $SCRATCH/resmill_dataset_win64 --dst $SCRATCH/SiliciclasticReservoirs
python build_splits.py  --root $SCRATCH/SiliciclasticReservoirs --out $SCRATCH/SiliciclasticReservoirs/splits --seed 42 --train-frac 0.90 --val-frac 0.05
```

One random 64 x 64 x 32 window per volume (deterministic from `crop_seed` 42
and the sample seed, base between 1 and 31, redrawn while the window has no
sand), 256 combined shards per family, the HuggingFace layout of per-layer-type
`shard_NNNN` directories, and the 90 / 5 / 5 splits. Each window row carries `crop_x0, crop_y0, crop_z0,
source_shard, source_row`, so the raw volume around it is
`$SCRATCH/resmill_dataset/<env>/<source_shard>` row `source_row`.

## Config schema

One JSON per environment. Top-level keys:

- `output_dir`  — where shards are written (`$SCRATCH` is expanded)
- `seed`        — master seed of the Sobol stream, the shuffle and the per-sample seeds
- `shard_size`  — samples per rank shard
- `grid`        — grid geometry applied to every sample
- `layers`      — one section per layer type: `count`, `sampling`
                  (`sobol` / `lhs` / `grid` / `uniform`) and `params`

Per-parameter specs, documented in `resmill/dataset/sampling.py`:

- `{"range": [lo, hi]}`, optionally `"scale": "log"` or `"type": "int"`
- `{"choices": [...]}`, `{"value": v}`, `{"levels": N}` (grid sampling)
- `{"fraction_of": "X", "value": k}`, `{"linear_of": "X", "slope": a, "intercept": b}`,
  `{"inverse_of": "X", "scale": s, "min": m, "max": M}`
- `{"levels_from_ratio": "mCHdepth", "ratio": [lo, hi], "column": H, "min": m, "max": M}` —
  number of levels from a sampled aggradation ratio
- `"shared": "tag"` and `"jitter": j` tie parameters to one Sobol coordinate

## Reading the dataset

```python
import numpy as np, pyarrow.parquet as pq
from pathlib import Path

shards = sorted(Path("$SCRATCH/SiliciclasticReservoirs/channel_pv_shoestring").glob("shard_*"))
facies = np.load(shards[0] / "facies.npy", mmap_mode="r")          # (N, 64, 64, 32)
rows = pq.read_table(shards[0] / "params.parquet").to_pylist()
```

The split parquets list `(layer_type, shard_dir, sample_idx)` per sample.
