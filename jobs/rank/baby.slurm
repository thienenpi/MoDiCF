#!/bin/bash
#SBATCH --job-name=raid-cf-baby-rank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue

# Rank-dependent counterfactual calibration on baby. Resubmitting resumes from checkpoint.

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
exec > "logs/rank-baby-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); started $(date)"
nvidia-smi || true

# --- 3. run (separate ckpt_dir from reproduce; fixed path => resumable) ---
# To sweep lambda at eval time, swap --lambda_cf for: --lambda_cf_list '[0.1, 0.5, 1.0, 2.0]'
python main.py \
    --dataset baby \
    --gamma_mode rank \
    --rank_k 20 \
    --lambda_cf 1.0 \
    --ckpt_dir ./checkpoint/baby/rank/

echo "Finished $(date)"
