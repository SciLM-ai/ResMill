"""SLURM rank-stripe dataset generation CLI.

Usage::

    python -m resmill.dataset.cli path/to/config.json

Runs serially (``SLURM_NTASKS`` unset) on a login or interactive node,
or as N parallel workers under
``srun -n N --cpu-bind=cores python -m resmill.dataset.cli ...``.
Each rank handles job indices ``[rank, rank+world, rank+2*world, ...]``
of a shuffled global job list, so cheap and expensive samples are mixed
across ranks and per-rank runtimes balance statistically.
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

from .generate import generate_sample
from .io import ShardWriter
from .sampling import build_jobs


def _check_level_columns(cfg: dict) -> None:
    """A ``levels_from_ratio`` spec must quote the grid's ``z_len``."""
    z_len = float(cfg["grid"]["z_len"])
    for lname, lcfg in cfg["layers"].items():
        for pname, spec in lcfg["params"].items():
            if isinstance(spec, dict) and "levels_from_ratio" in spec:
                col = float(spec["column"])
                if abs(col - z_len) > 1e-6:
                    raise ValueError(
                        f"{lname}.{pname}: levels_from_ratio column {col} "
                        f"differs from grid z_len {z_len}"
                    )


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = json.load(f)

    cfg["output_dir"] = os.path.expandvars(os.path.expanduser(cfg["output_dir"]))

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))

    # build_jobs returns an already-shuffled JobList (compact numpy storage;
    # ~100 MB even for 10M samples). jobs[i] materialises one dict on demand.
    _check_level_columns(cfg)
    jobs = build_jobs(cfg["layers"], cfg["seed"])
    my_indices = list(range(rank, len(jobs), world))

    if rank == 0:
        print(
            f"[rank 0] total_jobs={len(jobs)}  world={world}  "
            f"jobs_per_rank~={len(my_indices)}",
            flush=True,
        )

    writer = ShardWriter(cfg["output_dir"], rank, cfg["shard_size"])
    failures_path = (
        Path(cfg["output_dir"]) / f"failures_r{rank:04d}.jsonl"
    )
    n_failed = 0
    t0 = time.perf_counter()
    for n_done, i in enumerate(my_indices, 1):
        job = jobs[i]
        try:
            facies, poro, perm, facies_alluvsim, meta = generate_sample(
                job, cfg["grid"]
            )
            writer.add(facies, poro, perm, facies_alluvsim, meta)
        except Exception as exc:
            # A small fraction of stochastic parameter combinations trigger
            # pre-existing bugs in the underlying layer code (e.g. spline
            # fits on degenerate channel geometries). Log and skip — a
            # 0.01% loss rate is acceptable at million-sample scale.
            n_failed += 1
            with open(failures_path, "a") as f:
                f.write(json.dumps({
                    "global_index": int(i),
                    "layer_type": job["layer_type"],
                    "seed": int(job["seed"]),
                    "params": job["params"],
                    "error": repr(exc)[:300],
                }) + "\n")
        if rank == 0 and n_done % 50 == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / n_done * (len(my_indices) - n_done)
            print(
                f"[rank 0] {n_done}/{len(my_indices)}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
                f"failed={n_failed}",
                flush=True,
            )
    writer.close()

    wall = time.perf_counter() - t0
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        f"[rank {rank:4d}] samples={len(my_indices)}  "
        f"ok={len(my_indices) - n_failed}  failed={n_failed}  "
        f"wall={wall:.1f}s  max_rss={max_rss_mb:.0f}MB",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m resmill.dataset.cli path/to/config.json",
              file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])

