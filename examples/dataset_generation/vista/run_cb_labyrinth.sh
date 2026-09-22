#!/bin/bash
#SBATCH -p gg
#SBATCH -N 2
#SBATCH --ntasks-per-node=144
#SBATCH --cpus-per-task=1
#SBATCH -t 02:00:00
#SBATCH -J resmill_cb_labyrinth
#SBATCH -A CHE23004
#SBATCH -o logs/%x-%j.out

# 100,000 cb_labyrinth samples from ../config_full_cb_labyrinth.json, ResMill fdd8511, on TACC Vista gg
# nodes (2 x 72 Grace cores, no SMT: 144 single-thread ranks per node).
# Measured 2026-09-21 on warm processes with the node fully loaded, rows drawn
# from the published dataset's own parameter table: 12.7 s per volume on one
# core -> 353 core-hours = 2.4 node-hours; on 2 node(s) about
# 1.2 h wall, walltime 02:00:00 gives a 1.3x margin. Each rank takes
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

srun --cpu-bind=cores "$PY" -m resmill.dataset.cli examples/dataset_generation/config_full_cb_labyrinth.json
