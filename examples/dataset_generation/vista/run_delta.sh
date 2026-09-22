#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH -t 01:30:00
#SBATCH -J resmill_delta
#SBATCH -A CHE23004
#SBATCH -o logs/%x-%j.out

# 150,000 delta samples from ../config_full_delta_v2.json, ResMill b355621, on TACC Vista gg
# nodes (2 x 72 Grace cores, no SMT: 144 single-thread ranks per node).
# Tree delta (config v2): 2 to 4 s a cube instead of 90 to 110, measured 2026-09-22.
# The line below is the old avulsion delta's cost, kept for the record.
# Measured 2026-09-21 on warm processes with the node fully loaded, rows drawn
# from the published dataset's own parameter table: 89.6 s per volume on one
# core -> 3,733 core-hours = 25.9 node-hours; on 10 node(s) about
# 2.6 h wall, walltime 04:00:00 gives a 1.3x margin. Each rank takes
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
test "$(git rev-parse --short HEAD)" = b355621 || { echo "checkout ResMill b355621 first"; exit 1; }

# env -u LD_PRELOAD: TACC's XALT preload brings its own libcrypto into rank 0 and pyarrow
# then fails to import (OPENSSL_3.3.0 not found); measured on Vista 2026-09-22, rank 0 lost.
srun --cpu-bind=cores env -u LD_PRELOAD "$PY" -m resmill.dataset.cli examples/dataset_generation/config_full_delta_v2.json
