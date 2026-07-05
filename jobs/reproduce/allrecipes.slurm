#!/bin/bash
#SBATCH --job-name=modicf-allrecipes-repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --requeue

# Reproduce author's allrecipes results (fixed gamma). Resubmitting resumes from checkpoint.

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
exec > "logs/repro-allrecipes-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); started $(date)"
nvidia-smi || true

# --- 3. run (checkpoints live in the working dir, fixed path => resumable) ---
python main.py \
    --alpha_1 0.6 --delta 0.4 --eta 0.3 --gnn_embed_size 128 --H 8 --lambda_1 0.15 \
    --dataset allrecipes --load_dir ./checkpoint/allrecipes/ \
    --gamma 20 --alpha_2 0.5 --anchor_rate 0.75 \
    --ckpt_dir ./checkpoint/allrecipes/reproduce/

echo "Finished $(date)"
