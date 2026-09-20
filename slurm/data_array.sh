#!/bin/bash
#SBATCH -J clin-jepa-data
#SBATCH -p <partition>
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --array=0-9
#SBATCH -o logs/%x_%A_%a.out
# usage: sbatch slurm/data_array.sh <step>     (step02_observations | step03_actions | step04_discretization | step06_trajectories)

set -a; source .env; set +a
if [ "$1" = "step06_trajectories" ]; then
    python -m clin_jepa.data.$1 --chunk_idx $SLURM_ARRAY_TASK_ID
else
    python -m clin_jepa.data.$1 --chunk_idx $SLURM_ARRAY_TASK_ID --total_chunks $SLURM_ARRAY_TASK_COUNT
fi
