#!/bin/bash
#SBATCH --job-name=raid-cf-baby-deconf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue

# NEW METHOD: training-time counterfactual deconfounding on baby (300 epochs from scratch).
# Train + test on the consistent debiased score s = u.i - gamma*sigm(y_i).
# Baseline (MoDiCF) = jobs/reproduce/baby.slurm.
# Usage: sbatch jobs/deconf/baby.slurm [gamma]   (gamma_train, default 0.5)

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
GAMMA="${1:-0.5}"
exec > "logs/deconf-baby-g${GAMMA}-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); gamma_train=$GAMMA; started $(date)"
nvidia-smi || true

python main.py \
    --dataset baby --load_dir ./checkpoint/baby/ \
    --debias_mode subtract --gamma_train ${GAMMA} --from_scratch \
    --ckpt_dir ./checkpoint/baby/deconf-g${GAMMA}/

echo "Finished $(date)"
