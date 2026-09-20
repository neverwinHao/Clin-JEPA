#!/bin/bash
#SBATCH -J clin-jepa-train
#SBATCH -p <partition>
#SBATCH --gres=gpu:<gpu_type>:<n_gpus>
#SBATCH -c <n_cpus>
#SBATCH --mem=<memory>
#SBATCH -t <walltime>
#SBATCH -o logs/%x_%j.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
# usage: sbatch slurm/train.sh <module> <config>
#   e.g. sbatch slurm/train.sh curriculum configs/train/clin_jepa.yaml
#        sbatch slurm/train.sh train_predictor configs/train/predictor_vjepa2ac.yaml   (one GPU)

set -a; source .env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ "$1" = "train_predictor" ]; then
    python -m clin_jepa.training.$1 --config $2
    exit
fi
torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE --master_port=$((10000 + RANDOM % 40000)) \
    -m clin_jepa.training.$1 --config $2 &
LAUNCHER=$!
trap 'pkill -USR1 -P $LAUNCHER' USR1
while kill -0 $LAUNCHER 2>/dev/null; do wait $LAUNCHER; done
