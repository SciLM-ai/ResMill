#!/bin/bash
#SBATCH -p gg
#SBATCH -N 21
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH -t 05:30:00
#SBATCH -J resmill_meander_oxbow_v3
#SBATCH -A CHE23004
#SBATCH -o logs/%x-%j.out

# 100,000 meander_oxbow volumes of 128 x 128 x 64 from ../config_full_meander_oxbow_v3.json (v3: uncropped, depth 3-16 m,
# levels from the aggradation ratio), ResMill 5bf93ad or later, on TACC Vista gg nodes
# (2 x 72 Grace cores, no SMT: 144 single-thread ranks per node).
# Measured 2026-09-22 on the 144-volume preview with the node fully loaded: 427 s per volume on one
# core -> 11,856 core-hours = 82.3 node-hours at 144 ranks per node; on 21 node(s) about 3.9 h wall,
# walltime 05:30 gives a 1.3x margin. Each rank takes every N-th job of the shuffled Sobol job list.

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

REPO=/work/08405/ilgar/vista/codes/ResMill_ls6
PY=/work/08405/ilgar/vista/conda_libraries/resmill/bin/python
cd "$REPO" || exit 1
mkdir -p examples/dataset_generation/logs
git merge-base --is-ancestor 5bf93ad HEAD 2>/dev/null || { echo "this checkout does not contain ResMill 5bf93ad (v3 configs and the aggradation-ratio sampler); git pull"; exit 1; }

# env -u LD_PRELOAD: TACC's XALT preload brings its own libcrypto into rank 0 and pyarrow
# then fails to import (OPENSSL_3.3.0 not found); measured on Vista 2026-09-22, rank 0 lost.
srun --cpu-bind=cores env -u LD_PRELOAD "$PY" -m resmill.dataset.cli examples/dataset_generation/config_full_meander_oxbow_v3.json
