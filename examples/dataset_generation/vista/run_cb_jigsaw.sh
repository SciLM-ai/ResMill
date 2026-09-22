#!/bin/bash
#SBATCH -p gg
#SBATCH -N 3
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH -t 02:30:00
#SBATCH -J resmill_cb_jigsaw
#SBATCH -A CHE23004
#SBATCH -o logs/%x-%j.out

# 150,000 cb_jigsaw samples from ../config_full_cb_jigsaw.json, ResMill fdd8511, on TACC Vista gg
# nodes (2 x 72 Grace cores, no SMT: 144 single-thread ranks per node).
# Measured 2026-09-21 on warm processes with the node fully loaded, rows drawn
# from the published dataset's own parameter table: 14.6 s per volume on one
# core -> 608 core-hours = 4.2 node-hours; on 3 node(s) about
# 1.4 h wall, walltime 02:30:00 gives a 1.3x margin. Each rank takes
# every N-th job of the shuffled Sobol job list, so ranks finish together.

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

REPO=/work/08405/ilgar/vista/codes/ResMill_ls6
PY=/work/08405/ilgar/vista/conda_libraries/resmill/bin/python
cd "$REPO" || exit 1
mkdir -p examples/dataset_generation/logs
test "$(git rev-parse --short HEAD)" = fdd8511 || { echo "checkout ResMill fdd8511 first"; exit 1; }

# env -u LD_PRELOAD: TACC's XALT preload brings its own libcrypto into rank 0 and pyarrow
# then fails to import (OPENSSL_3.3.0 not found); measured on Vista 2026-09-22, rank 0 lost.
srun --cpu-bind=cores env -u LD_PRELOAD "$PY" -m resmill.dataset.cli examples/dataset_generation/config_full_cb_jigsaw.json
