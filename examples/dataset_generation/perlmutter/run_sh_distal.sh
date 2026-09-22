#!/bin/bash
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 9
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -t 05:00:00
#SBATCH -J resmill_sh_distal_v3
#SBATCH --licenses=cfs,SCRATCH
#SBATCH -A REPLACE_WITH_YOUR_ALLOCATION
#SBATCH -o logs/%x-%j.out

# 100,000 sh_distal volumes of 128 x 128 x 64 from ../config_full_sh_distal_v3.json (v3), ResMill a8bfbfd or later,
# on NERSC Perlmutter CPU nodes: 128 single-thread ranks per node as in v1.
# Cost from the Vista measurement of 2026-09-22 (122 s a volume on one Grace core,
# assumed 1.3x slower per core here): 4,391 core-hours = 34.3 node-hours,
# about 3.8 h on 9 node(s); walltime 05:00 leaves a 1.3x margin.

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
git merge-base --is-ancestor a8bfbfd HEAD 2>/dev/null || { echo "this checkout does not contain ResMill a8bfbfd (v3 configs and the aggradation-ratio sampler); git pull"; exit 1; }

srun --cpu-bind=cores python -m resmill.dataset.cli examples/dataset_generation/config_full_sh_distal_v3.json
