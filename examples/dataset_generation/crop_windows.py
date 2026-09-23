"""Cut one random 64 x 64 x 32 training window out of every stored volume.

The generation jobs store whole 128 x 128 x 64 volumes (the raw rank shards).
This tool writes a second shard tree with the same shard layout, one window per
volume, so ``combine_shards.py`` / ``stage_dataset.py`` / ``build_splits.py``
run on it unchanged, and the raw volumes stay available as context.

Determinism contract:
  1. The window origin of a sample is drawn from
     ``numpy.random.default_rng([crop_seed, sample_seed])`` (``seed`` column of
     the sample): ``x0`` uniform on ``[0, nx - wx]``, ``y0`` on ``[0, ny - wy]``,
     ``z0`` uniform on ``[z_min, nz - wz - 1]`` (default ``z_min`` = 1 so the
     window never contains the engine's floor or roof, where the level ladder
     is anchored); the draw is repeated from the same stream while the window
     holds fewer than ``--min-sand-cells`` (default 1) sand cells, so no window
     is mud only. Same seeds -> same windows.
  2. Rank shard ``shard_rXXXX_sNNNNNN`` of a preset becomes the shard of the
     same name in the output, rows in the same order (one ``ShardWriter`` per
     rank with the raw shard size, so names and partial last shards match).
  3. Realised metadata is recomputed on the window with the same code
     ``resmill.dataset.generate`` runs on a volume (``realized_stats`` and
     ``caption_for``): ``ntg``, ``poro_ave``, ``perm_ave``, ``caption``.
     Everything else is copied. New columns:
     ``crop_x0, crop_y0, crop_z0, crop_nx, crop_ny, crop_nz, crop_seed,
     source_shard, source_row`` locate the window in its source volume.
  4. ``--verify`` re-reads random windows and checks them against the source
     volume and the recomputed ``ntg``; counts are checked for every shard.

Usage:
    python crop_windows.py --src $SCRATCH/resmill_dataset \\
        --dst $SCRATCH/resmill_dataset_win64 --workers 96 --verify
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from resmill.dataset.captions import caption_for
from resmill.dataset.generate import realized_stats
from resmill.dataset.io import ShardWriter

PRESETS = ["lobes", "pv_shoestring", "cb_labyrinth", "cb_jigsaw", "sh_distal",
           "sh_proximal", "meander_oxbow", "delta"]


def window_origin(crop_seed: int, sample_seed: int, shape, win, z_min: int, facies=None,
                  min_sand_cells: int = 1, max_tries: int = 64):
    """Window origin for one sample. Draws from ``default_rng([crop_seed, sample_seed])``
    and, when ``facies`` is given, redraws until the window holds at least
    ``min_sand_cells`` sand cells (a mud-only window has no porosity or
    permeability statistics), taking the last draw if ``max_tries`` fail."""
    nx, ny, nz = shape
    wx, wy, wz = win
    rng = np.random.default_rng([int(crop_seed), int(sample_seed)])
    for _ in range(max_tries):
        x0 = int(rng.integers(0, nx - wx + 1))
        y0 = int(rng.integers(0, ny - wy + 1))
        z0 = int(rng.integers(z_min, nz - wz))      # high is exclusive: z0 <= nz - wz - 1
        if facies is None or min_sand_cells <= 0:
            break
        if int(np.count_nonzero(facies[x0:x0 + wx, y0:y0 + wy, z0:z0 + wz])) >= min_sand_cells:
            break
    return x0, y0, z0


def realised_meta(facies: np.ndarray, poro: np.ndarray, perm: np.ndarray, meta: dict) -> dict:
    """Window row = volume row with ``ntg`` / ``poro_ave`` / ``perm_ave`` and the
    caption recomputed on the window by the same code ``generate_sample`` uses."""
    meta = dict(meta)
    meta.update(realized_stats(facies, poro, perm))
    meta["caption"] = caption_for(meta["layer_type"], meta)
    return meta


def _rank_shards(preset_dir: Path, rank: int) -> list[Path]:
    return sorted(preset_dir.glob(f"shard_r{rank:04d}_s*"))


def crop_rank(args):
    preset, rank, src_root, dst_root, win, z_min, crop_seed, min_sand = args
    src_dir = Path(src_root) / preset
    dst_dir = Path(dst_root) / preset
    shards = _rank_shards(src_dir, rank)
    if not shards:
        return preset, rank, 0, 0
    shard_size = int(np.load(shards[0] / "facies.npy", mmap_mode="r").shape[0])
    writer = ShardWriter(str(dst_dir), rank, shard_size)
    n_in = 0
    for shard in shards:
        # whole arrays, sequential reads: 13x faster on Lustre than memory-mapped window slices
        f = np.load(shard / "facies.npy")
        p = np.load(shard / "poro.npy")
        k = np.load(shard / "perm.npy")
        fa = np.load(shard / "facies_alluvsim.npy")
        rows = pq.read_table(shard / "params.parquet").to_pylist()
        if len(rows) != f.shape[0]:
            raise RuntimeError(f"{shard}: {len(rows)} parquet rows but {f.shape[0]} volumes")
        shape = f.shape[1:]
        for i, row in enumerate(rows):
            x0, y0, z0 = window_origin(crop_seed, row["seed"], shape, win, z_min, f[i], min_sand)
            sl = (i, slice(x0, x0 + win[0]), slice(y0, y0 + win[1]), slice(z0, z0 + win[2]))
            fc = np.ascontiguousarray(f[sl]); pc = np.ascontiguousarray(p[sl])
            kc = np.ascontiguousarray(k[sl]); fac = np.ascontiguousarray(fa[sl])
            meta = realised_meta(fc, pc, kc, row)
            meta.update(crop_x0=x0, crop_y0=y0, crop_z0=z0, crop_nx=win[0], crop_ny=win[1],
                        crop_nz=win[2], crop_seed=int(crop_seed), source_shard=shard.name,
                        source_row=i)
            writer.add(fc, pc, kc, fac, meta)
            n_in += 1
    writer.close()
    n_out = sum(int(np.load(s / "facies.npy", mmap_mode="r").shape[0]) for s in _rank_shards(dst_dir, rank))
    return preset, rank, n_in, n_out


def verify(src_root: Path, dst_root: Path, presets, per_preset: int, seed: int):
    rng = np.random.default_rng(seed)
    for preset in presets:
        shards = sorted((dst_root / preset).glob("shard_r*"))
        picks = rng.choice(len(shards), size=min(per_preset, len(shards)), replace=False)
        for j in picks:
            shard = shards[j]
            tab = pq.read_table(shard / "params.parquet").to_pylist()
            i = int(rng.integers(0, len(tab))); row = tab[i]
            win = np.load(shard / "facies.npy", mmap_mode="r")[i]
            src = np.load(src_root / preset / row["source_shard"] / "facies.npy", mmap_mode="r")[row["source_row"]]
            x0, y0, z0 = row["crop_x0"], row["crop_y0"], row["crop_z0"]
            ref = src[x0:x0 + row["crop_nx"], y0:y0 + row["crop_ny"], z0:z0 + row["crop_nz"]]
            if not np.array_equal(win, ref):
                raise RuntimeError(f"{shard} row {i}: window differs from its source volume")
            if row["ntg"] <= 0.0:
                raise RuntimeError(f"{shard} row {i}: mud-only window")
            if abs(row["ntg"] - float(win.mean())) > 1e-6:
                raise RuntimeError(f"{shard} row {i}: ntg {row['ntg']} != window sand fraction {win.mean()}")
            src_row = pq.read_table(src_root / preset / row["source_shard"] / "params.parquet").to_pylist()[row["source_row"]]
            if src_row["seed"] != row["seed"]:
                raise RuntimeError(f"{shard} row {i}: seed mismatch with source row")
        print(f"    verify {preset}: {len(picks)} random windows match their source volumes and ntg", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="root with the raw <preset>/shard_rXXXX_sNNNNNN dirs")
    ap.add_argument("--dst", required=True, help="output root, same layout")
    ap.add_argument("--window", type=int, nargs=3, default=(64, 64, 32), metavar=("NX", "NY", "NZ"))
    ap.add_argument("--z-min", type=int, default=1, help="lowest allowed window base (default 1: never the floor)")
    ap.add_argument("--crop-seed", type=int, default=42)
    ap.add_argument("--min-sand-cells", type=int, default=1,
                    help="redraw the origin until the window has at least this many sand cells (0 disables)")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--presets", nargs="+", default=PRESETS)
    ap.add_argument("--verify", action="store_true", help="re-read random windows against the source volumes")
    ap.add_argument("--verify-per-preset", type=int, default=8)
    args = ap.parse_args()

    src_root = Path(os.path.expandvars(os.path.expanduser(args.src)))
    dst_root = Path(os.path.expandvars(os.path.expanduser(args.dst)))
    win = tuple(int(v) for v in args.window)
    tasks = []
    for preset in args.presets:
        ranks = sorted({int(p.name.split("_")[1][1:]) for p in (src_root / preset).glob("shard_r*")})
        if (dst_root / preset).exists() and any((dst_root / preset).glob("shard_r*")):
            sys.exit(f"{dst_root / preset} already has shards; remove it first")
        tasks += [(preset, r, str(src_root), str(dst_root), win, args.z_min, args.crop_seed, args.min_sand_cells) for r in ranks]
    print(f"{len(tasks)} (preset, rank) tasks, window {win}, z0 in [{args.z_min}, nz-{win[2]}-1], crop_seed {args.crop_seed}", flush=True)
    t0 = time.perf_counter(); totals = {}
    with Pool(args.workers) as pool:
        for n, (preset, rank, n_in, n_out) in enumerate(pool.imap_unordered(crop_rank, tasks), 1):
            if n_in != n_out:
                raise RuntimeError(f"{preset} rank {rank}: {n_in} volumes in, {n_out} windows out")
            a, b = totals.get(preset, (0, 0)); totals[preset] = (a + n_in, b + 1)
            if n % 100 == 0 or n == len(tasks):
                print(f"  [{n}/{len(tasks)}] elapsed={time.perf_counter() - t0:.0f}s", flush=True)
    for preset in args.presets:
        n, r = totals.get(preset, (0, 0))
        print(f"  {preset:14s} {n:>8,} windows from {r} ranks", flush=True)
    print(f"  total {sum(v[0] for v in totals.values()):,} windows", flush=True)
    if args.verify:
        verify(src_root, dst_root, args.presets, args.verify_per_preset, args.crop_seed)


if __name__ == "__main__":
    main()
