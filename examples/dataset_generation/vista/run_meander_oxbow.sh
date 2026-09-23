#!/bin/bash
#SBATCH -p gg
#SBATCH -N 3
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH -t 05:00:00
#SBATCH -J resmill_meander_oxbow_v3
#SBATCH -A CHE23004
#SBATCH -o logs/%x-%j.out

# 100,000 meander_oxbow volumes of 128 x 128 x 64 from ../config_full_meander_oxbow_v3.json (v3: uncropped, depth 3-16 m,
# levels from the aggradation ratio), ResMill adf8c29 or later, on TACC Vista gg nodes
# (2 x 72 Grace cores, no SMT: 144 single-thread ranks per node).
# Measured 2026-09-22 on the 144-volume preview with the node fully loaded: 56 s per volume on one
# core -> 1,553 core-hours = 10.8 node-hours at 144 ranks per node; on 3 node(s) about 3.6 h wall,
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
git merge-base --is-ancestor adf8c29 HEAD 2>/dev/null || { echo "this checkout does not contain ResMill adf8c29 (v3 configs and the aggradation-ratio sampler); git pull"; exit 1; }

# env -u LD_PRELOAD: TACC's XALT preload brings its own libcrypto into rank 0 and pyarrow
# then fails to import (OPENSSL_3.3.0 not found); measured on Vista 2026-09-22, rank 0 lost.
srun --cpu-bind=cores env -u LD_PRELOAD "$PY" -m resmill.dataset.cli examples/dataset_generation/config_full_meander_oxbow_v3.json
