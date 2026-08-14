#!/bin/bash
#SBATCH --job-name=raid-cf-tiktok-deconf-group
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --array=0-9

# NEW METHOD: training-time counterfactual deconfounding on tiktok, with a per-user-group
# rank-based margin instead of a flat gamma. Users are clustered into "hobby" groups by
# embedding similarity; each group shares one gamma_{g,i} = max(0, score_{g,i} - y_{g,K+1}).
# Train + test on s = u.i - lambda_cf*gamma_{g(u),i}*sigm(y_i).
# Flat-gamma variant = jobs/deconf/tiktok.slurm. Compared against the officially published
# MoDiCF numbers (paper Table 2), NOT against a local reproduction.
# Repeated over 10 seeds (paper protocol, Sec 5.2.3) via a SLURM job array.
# Usage: sbatch jobs/deconf-group/tiktok.slurm [lambda_cf] [group_k]   (defaults: 1.0, 10)
# Aggregate with: python scripts/aggregate_seeds.py "logs/deconf-group-tiktok-l<lambda>-k<k>-seed*.out"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
LAMBDA="${1:-1.0}"
GROUP_K="${2:-10}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
exec > "logs/deconf-group-tiktok-l${LAMBDA}-k${GROUP_K}-seed${SEED}-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); lambda_cf=$LAMBDA; group_k=$GROUP_K; seed=$SEED; started $(date)"
nvidia-smi || true

python main.py \
    --alpha_1 0.7 --delta 0.3 --eta 0.6 --H 4 --lambda_1 0.06 \
    --dataset tiktok --load_dir ./checkpoint/tiktok/ --seed ${SEED} --data_seed 0 \
    --alpha_2 0.3 --anchor_rate 0.5 \
    --debias_mode subtract --gamma_mode group --lambda_cf ${LAMBDA} --group_k ${GROUP_K} --from_scratch \
    --ckpt_dir ./checkpoint/tiktok/deconf-group-l${LAMBDA}-k${GROUP_K}-seed${SEED}/

echo "Finished $(date)"
