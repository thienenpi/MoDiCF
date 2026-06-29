#!/bin/bash
#SBATCH --job-name=raid-cf-allrecipes-deconf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue

# NEW METHOD: training-time counterfactual deconfounding on allrecipes (300 epochs from scratch).
# Train + test on the consistent debiased score s = u.i - gamma*sigm(y_i).
# Baseline (MoDiCF) = jobs/reproduce/allrecipes.slurm.
# Usage: sbatch jobs/deconf/allrecipes.slurm [gamma]   (gamma_train, default 0.5)

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
GAMMA="${1:-0.5}"
exec > "logs/deconf-allrecipes-g${GAMMA}-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); gamma_train=$GAMMA; started $(date)"
nvidia-smi || true

python main.py \
    --alpha_1 0.6 --delta 0.4 --eta 0.3 --gnn_embed_size 128 --H 8 --lambda_1 0.15 \
    --dataset allrecipes --load_dir ./checkpoint/allrecipes/ \
    --alpha_2 0.5 --anchor_rate 0.75 \
    --debias_mode subtract --gamma_train ${GAMMA} --from_scratch \
    --ckpt_dir ./checkpoint/allrecipes/deconf-g${GAMMA}/

echo "Finished $(date)"
