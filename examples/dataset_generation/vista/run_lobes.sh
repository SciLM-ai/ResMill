#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH -t 05:00:00
#SBATCH -J resmill_lobes_v2
#SBATCH -A CHE23004
#SBATCH -o logs/%x-%j.out

# 200,000 lobes volumes of 128 x 128 x 64 from ../config_full_lobes_v2.json (v2: uncropped, depth 3-16 m,
# levels from the aggradation ratio), ResMill d459b4b or later, on TACC Vista gg nodes
# (2 x 72 Grace cores, no SMT: 96 single-thread ranks per node; lobes need 1.4 GB per rank at this size, so 96 ranks on a 237 GB node).
# Measured 2026-09-22 on the 144-volume preview with the node fully loaded: 6 s per volume on one
# core -> 350 core-hours = 3.6 node-hours at 96 ranks per node; on 1 node(s) about 3.6 h wall,
# walltime 05:00 gives a 1.3x margin. Each rank takes every N-th job of the shuffled Sobol job list.

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

REPO=/work/08405/ilgar/vista/codes/ResMill_ls6
PY=/work/08405/ilgar/vista/conda_libraries/resmill/bin/python
cd "$REPO" || exit 1
mkdir -p examples/dataset_generation/logs
git merge-base --is-ancestor d459b4b HEAD 2>/dev/null || { echo "this checkout does not contain ResMill d459b4b (v2 configs and the aggradation-ratio sampler); git pull"; exit 1; }

# env -u LD_PRELOAD: TACC's XALT preload brings its own libcrypto into rank 0 and pyarrow
# then fails to import (OPENSSL_3.3.0 not found); measured on Vista 2026-09-22, rank 0 lost.
srun --cpu-bind=cores env -u LD_PRELOAD "$PY" -m resmill.dataset.cli examples/dataset_generation/config_full_lobes_v2.json
