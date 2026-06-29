#!/bin/bash
#SBATCH --job-name=raid-cf-tiktok-deconf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue

# NEW METHOD: training-time counterfactual deconfounding on tiktok (300 epochs from scratch).
# Train + test on the consistent debiased score s = u.i - gamma*sigm(y_i).
# Baseline (MoDiCF) = jobs/reproduce/tiktok.slurm.
# Usage: sbatch jobs/deconf/tiktok.slurm [gamma]   (gamma_train, default 0.5; sweep 0.1/0.3/0.5/1.0/2.0)

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
GAMMA="${1:-0.5}"
exec > "logs/deconf-tiktok-g${GAMMA}-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); gamma_train=$GAMMA; started $(date)"
nvidia-smi || true

python main.py \
    --alpha_1 0.7 --delta 0.3 --eta 0.6 --H 4 --lambda_1 0.06 \
    --dataset tiktok --load_dir ./checkpoint/tiktok/ \
    --alpha_2 0.3 --anchor_rate 0.5 \
    --debias_mode subtract --gamma_train ${GAMMA} --from_scratch \
    --ckpt_dir ./checkpoint/tiktok/deconf-g${GAMMA}/

echo "Finished $(date)"
