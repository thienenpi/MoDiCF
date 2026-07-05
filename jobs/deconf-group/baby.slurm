#!/bin/bash
#SBATCH --job-name=raid-cf-baby-deconf-group
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue

# NEW METHOD: training-time counterfactual deconfounding on baby, with a per-user-group
# rank-based margin instead of a flat gamma. Users are clustered into "hobby" groups by
# embedding similarity; each group shares one gamma_{g,i} = max(0, score_{g,i} - y_{g,K+1}).
# Train + test on s = u.i - lambda_cf*gamma_{g(u),i}*sigm(y_i).
# Flat-gamma variant = jobs/deconf/baby.slurm. Baseline (MoDiCF) = jobs/reproduce/baby.slurm.
# Usage: sbatch jobs/deconf-group/baby.slurm [lambda_cf] [group_k]   (defaults: 1.0, 10)

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
LAMBDA="${1:-1.0}"
GROUP_K="${2:-10}"
exec > "logs/deconf-group-baby-l${LAMBDA}-k${GROUP_K}-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); lambda_cf=$LAMBDA; group_k=$GROUP_K; started $(date)"
nvidia-smi || true

python main.py \
    --dataset baby --load_dir ./checkpoint/baby/ \
    --debias_mode subtract --gamma_mode group --lambda_cf ${LAMBDA} --group_k ${GROUP_K} --from_scratch \
    --ckpt_dir ./checkpoint/baby/deconf-group-l${LAMBDA}-k${GROUP_K}/

echo "Finished $(date)"
