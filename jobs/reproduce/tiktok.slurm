#!/bin/bash
#SBATCH --job-name=modicf-tiktok-repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --requeue

# Reproduce author's tiktok results (fixed gamma). Resubmitting resumes from checkpoint.

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
exec > "logs/repro-tiktok-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); started $(date)"
nvidia-smi || true

# --- 3. run (checkpoints live in the working dir, fixed path => resumable) ---
python main.py \
    --alpha_1 0.7 --delta 0.3 --eta 0.6 --H 4 --lambda_1 0.06 \
    --dataset tiktok --load_dir ./checkpoint/tiktok/ \
    --gamma 20 --alpha_2 0.3 --anchor_rate 0.5 \
    --gamma_mode fixed \
    --ckpt_dir ./checkpoint/tiktok/reproduce/

echo "Finished $(date)"
