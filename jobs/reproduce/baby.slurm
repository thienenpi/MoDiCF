#!/bin/bash
#SBATCH --job-name=modicf-baby-repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --requeue

# Job 1: reproduce the author's results on the baby dataset (original fixed-gamma,
# i.e. eq. 17). Re-submitting this same script resumes from the last checkpoint
# instead of restarting from epoch 0.

# --- 1. resources are declared above ---

# --- 2. environment ---
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
exec > "logs/repro-baby-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); started $(date)"
nvidia-smi || true

# --- 3. run (checkpoints live in the working dir, fixed path => resumable) ---
python main.py \
    --dataset baby \
    --gamma_mode fixed \
    --ckpt_dir ./checkpoint/baby/reproduce/

echo "Finished $(date)"
