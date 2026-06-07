#!/bin/bash
#SBATCH --job-name=modicf-baby-rank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/rank-baby-%j.out
#SBATCH --error=logs/rank-baby-%j.out

# Job 2: evaluate the new rank-dependent counterfactual calibration on baby
# (gamma_{u,i} = max(0, y_{u,i} - y_{u,K+1}), CountER-style margin).
# Re-submitting this same script resumes from the last checkpoint.

# --- 1. resources are declared above ---

# --- 2. environment ---
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); started $(date)"
nvidia-smi || true

# --- 3. run (separate ckpt_dir from the reproduce job; fixed path => resumable) ---
# To sweep lambda at eval time, swap --lambda_cf for:
#   --lambda_cf_list '[0.1, 0.5, 1.0, 2.0]'
python main.py \
    --dataset baby \
    --gamma_mode rank \
    --rank_k 20 \
    --lambda_cf 1.0 \
    --ckpt_dir ./checkpoint/baby/rank/

echo "Finished $(date)"
