#!/bin/bash
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 02:00:00
#SBATCH -J resmill_lobes_v2
#SBATCH --licenses=cfs,SCRATCH
#SBATCH -A REPLACE_WITH_YOUR_ALLOCATION
#SBATCH -o logs/%x-%j.out

# v2 dataset, 200,000 lobes samples from ../config_full_lobes.json (ResMill 3e71977 or later),
# on NERSC Perlmutter CPU nodes: 128 single-thread ranks per node as in v1.
# Cost from the Vista measurement of 2026-09-22 (1.8 s a cube on one Grace core,
# assumed 1.3x slower per core here): 130 core-hours = 1.0 node-hours,
# about 1.0 h on 1 node(s); walltime 02:00 leaves a 1.3x margin.

export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

module load conda
conda activate $WORK/conda_envs/resmill     # the env that has ResMill installed (pip install -e ".[dataset]")

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO" || exit 1
mkdir -p examples/dataset_generation/logs
git merge-base --is-ancestor 3e71977 HEAD 2>/dev/null || { echo "this checkout does not contain ResMill 3e71977 (final engine and v2 configs); git pull"; exit 1; }

srun --cpu-bind=cores python -m resmill.dataset.cli examples/dataset_generation/config_full_lobes.json
