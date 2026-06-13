#!/bin/bash
#SBATCH --job-name=raid-cf-tiktok-rank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue

# Rank-dependent counterfactual calibration on tiktok. Resubmitting resumes from checkpoint.

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
exec > "logs/rank-tiktok-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); started $(date)"
nvidia-smi || true

# --- 3. run (separate ckpt_dir from reproduce; fixed path => resumable) ---
# In rank mode --gamma is unused (replaced by the per-pair margin); lambda controls strength.
python main.py \
    --alpha_1 0.7 --delta 0.3 --eta 0.6 --H 4 --lambda_1 0.06 \
    --dataset tiktok --load_dir ./checkpoint/tiktok/ \
    --alpha_2 0.3 --anchor_rate 0.5 \
    --gamma_mode rank \
    --rank_k 20 \
    --lambda_cf 1.0 \
    --ckpt_dir ./checkpoint/tiktok/rank/

echo "Finished $(date)"
