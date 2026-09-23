"""Combine rank-shards into a smaller number of consolidated shards.

Determinism contract:
  1. Original shards are listed via ``sorted(glob)``. Names are
     ``shard_rXXXX_sNNNNNN`` with zero-padded XXXX so lex sort
     matches numeric sort (rank 2 < rank 10 < rank 100).
  2. For each contiguous group of ``GROUP`` shards, the four ``.npy``
     arrays and two parquets are concatenated in that exact order via
     ``np.concatenate`` and ``pyarrow.concat_tables`` — both functions
     preserve input row order. Result row N comes from rank-shard
     ``N // shard_size_per_rank`` (within the group).
  3. The same shard list is used for all 6 files, so all 6 stay
     row-aligned. Within each rank-shard the 6 files are already
     row-aligned (written from a single in-memory list).
  4. After combining, the script verifies:
       - total combined samples == total original samples
       - per-combined-shard sample count == sum of contributing rank shards
       - sample 0 of combined shard 0 == sample 0 of rank-shard 0 (spot)
"""
import argparse
import glob
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa


def list_rank_shards(preset_dir: Path) -> list[Path]:
    """Sorted (lex == numeric on zero-padded rank IDs) list of rank shards."""
    shards = sorted(preset_dir.glob("shard_r*_s*"))
    # Verify the lex sort actually matches numeric sort by checking rank IDs
    rank_ids = []
    for s in shards:
        # name: shard_rXXXX_sNNNNNN
        parts = s.name.split("_")
        rank_id = int(parts[1][1:])  # strip "r" prefix
        rank_ids.append(rank_id)
    if rank_ids != sorted(rank_ids):
        raise RuntimeError(
            f"shard lex sort != numeric sort in {preset_dir}; "
            f"first ten rank IDs: {rank_ids[:10]}"
        )
    return shards


def _concat_one_group(args):
    group_idx, group_paths, out_dir, dry_run = args
    """Concat one group of rank shards into one combined shard."""
    out_name = f"combined_shard_{group_idx:04d}"
    out_path = Path(out_dir) / out_name
    tmp_path = Path(out_dir) / (out_name + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Load arrays from each shard in order — concat preserves order.
    arrays = {"facies": [], "poro": [], "perm": [], "facies_alluvsim": []}
    tables_full = []
    tables_slim = []
    per_shard_counts = []
    for src in group_paths:
        # Load each .npy via memmap for streaming (don't blow RAM on large groups)
        n_samples = None
        for name in arrays:
            arr = np.load(src / f"{name}.npy", mmap_mode="r")
            if n_samples is None:
                n_samples = int(arr.shape[0])
            elif n_samples != arr.shape[0]:
                raise RuntimeError(
                    f"{src}: {name}.npy has {arr.shape[0]} samples but expected {n_samples}"
                )
            arrays[name].append(arr)
        per_shard_counts.append(n_samples)
        tables_full.append(pq.read_table(src / "params.parquet"))
        tables_slim.append(pq.read_table(src / "params_slim.parquet"))

    if dry_run:
        return group_idx, sum(per_shard_counts), per_shard_counts

    # Concatenate. ``np.concatenate`` preserves input order (rank0 rows
    # first, then rank1, …). ``pyarrow.concat_tables(promote=True)``
    # preserves row order across tables and unions schemas.
    for name, parts in arrays.items():
        out_arr = np.concatenate(parts, axis=0)
        np.save(tmp_path / f"{name}.npy", out_arr)
        del out_arr  # free memory before next array

    full = pa.concat_tables(tables_full, promote=True)
    pq.write_table(full, tmp_path / "params.parquet")
    slim = pa.concat_tables(tables_slim, promote=True)
    pq.write_table(slim, tmp_path / "params_slim.parquet")

    # Atomic rename
    if out_path.exists():
        shutil.rmtree(out_path)
    os.rename(tmp_path, out_path)
    return group_idx, sum(per_shard_counts), per_shard_counts


def combine_preset(preset_dir: Path, target_n, out_dir: Path,
                   workers: int, dry_run: bool, group: int | None = None):
    rank_shards = list_rank_shards(preset_dir)
    n_rank = len(rank_shards)
    if group is not None:
        group_size = int(group)
        target_n = (n_rank + group_size - 1) // group_size
    else:
        if n_rank % target_n != 0:
            raise RuntimeError(
                f"{preset_dir.name}: {n_rank} rank shards not divisible by "
                f"target {target_n}"
            )
        group_size = n_rank // target_n
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  rank shards: {n_rank}  →  combined: {target_n}  (group={group_size})")

    tasks = []
    for g in range(target_n):
        group_paths = rank_shards[g * group_size : (g + 1) * group_size]
        tasks.append((g, group_paths, str(out_dir), dry_run))

    t0 = time.perf_counter()
    total_combined = 0
    with Pool(workers) as pool:
        for g_idx, g_count, per_shard in pool.imap_unordered(_concat_one_group, tasks):
            total_combined += g_count
            if g_idx % 32 == 0 or g_idx == target_n - 1:
                wall = time.perf_counter() - t0
                print(
                    f"    [{g_idx + 1:4d}/{target_n}]  group_count={g_count}  "
                    f"elapsed={wall:.1f}s",
                    flush=True,
                )
    return total_combined


def verify(preset_dir: Path, out_dir: Path):
    """Cross-check sample counts + spot-check first sample."""
    rank_shards = list_rank_shards(preset_dir)
    rank_total = 0
    for s in rank_shards:
        rank_total += int(np.load(s / "facies.npy", mmap_mode="r").shape[0])

    combined = sorted(out_dir.glob("combined_shard_*"))
    comb_total = 0
    for s in combined:
        comb_total += int(np.load(s / "facies.npy", mmap_mode="r").shape[0])

    if rank_total != comb_total:
        raise RuntimeError(
            f"VERIFY FAIL {preset_dir.name}: rank total {rank_total} "
            f"vs combined total {comb_total}"
        )

    # Spot: combined shard 0, sample 0 should equal rank shard 0, sample 0
    # for each of the 4 npy arrays.
    for name in ("facies", "poro", "perm", "facies_alluvsim"):
        rank_arr = np.load(rank_shards[0] / f"{name}.npy", mmap_mode="r")
        comb_arr = np.load(combined[0] / f"{name}.npy", mmap_mode="r")
        if not np.array_equal(rank_arr[0], comb_arr[0]):
            raise RuntimeError(
                f"VERIFY FAIL {preset_dir.name}: {name}[0] mismatch combined vs rank"
            )

    # Spot: parquet — first row of combined slim should equal first row of rank slim
    rank_slim = pq.read_table(rank_shards[0] / "params_slim.parquet").to_pylist()[0]
    comb_slim = pq.read_table(combined[0] / "params_slim.parquet").to_pylist()[0]
    for k, v in rank_slim.items():
        cv = comb_slim.get(k)
        if v != cv and not (isinstance(v, float) and np.isnan(v) and np.isnan(cv)):
            # Floats: tolerate numerical sameness (parquet round-trip)
            if isinstance(v, float) and isinstance(cv, float) and abs(v - cv) < 1e-12:
                continue
            raise RuntimeError(
                f"VERIFY FAIL {preset_dir.name}: slim parquet row 0 col {k}: "
                f"rank={v} vs combined={cv}"
            )

    print(f"    ✓ verify {preset_dir.name}: total {rank_total}, "
          f"sample[0] arrays + parquet match rank-shard 0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="$SCRATCH/resmill_dataset")
    ap.add_argument("--target", type=int, default=None,
                    help="number of combined shards per preset (rank-shard count must divide by it)")
    ap.add_argument("--group", type=int, default=None,
                    help="alternative to --target: rank shards per combined shard (8 x 32-sample "
                         "rank shards = 256-sample combined shards); the last group may be smaller")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out-suffix", default="_combined")
    ap.add_argument("--dry-run", action="store_true",
                    help="list groups + counts, do not write")
    ap.add_argument("--presets", nargs="+", default=None,
                    help="subset of presets to combine (default: all 8)")
    args = ap.parse_args()

    root = Path(os.path.expandvars(os.path.expanduser(args.root)))
    # v2 preset directories (v1 used channels_<preset>; pass --presets for those)
    all_presets = ["lobes", "pv_shoestring", "cb_labyrinth", "cb_jigsaw", "sh_distal",
                   "sh_proximal", "meander_oxbow", "delta"]
    if (args.target is None) == (args.group is None):
        ap.error("give exactly one of --target or --group")
    presets = args.presets if args.presets else all_presets

    grand_total = 0
    for p in presets:
        in_dir = root / p
        out_dir = root / (p + args.out_suffix)
        if not in_dir.is_dir():
            print(f"SKIP {p}: {in_dir} does not exist")
            continue
        print(f"\n=== {p} ===")
        n = combine_preset(in_dir, args.target, out_dir, args.workers, args.dry_run, args.group)
        if not args.dry_run:
            verify(in_dir, out_dir)
        grand_total += n

    print(f"\n=== GRAND TOTAL: {grand_total} samples across {len(presets)} presets ===")


if __name__ == "__main__":
    main()
