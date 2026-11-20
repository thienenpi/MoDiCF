#!/bin/bash
#SBATCH --job-name=raid-cf-tiktok
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/raid-cf-raid-cf-tiktok-l1.0-gk10-k10-b0.3-seed2-%j.out

# Expanded form of: sbatch --array=2 jobs/raid-cf/tiktok.slurm raid-cf
# variant=raid-cf, lambda_cf=1.0, group_k=10, retrieval_k=10, retrieval_beta=0.3, seed=2
# Usage: mkdir -p logs && sbatch jobs/raid-cf/tiktok-raid-cf-seed2.slurm

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); variant=raid-cf; lambda_cf=1.0; group_k=10; retrieval_k=10; beta=0.3; seed=2; started $(date)"
nvidia-smi || true

if [ -f "./checkpoint/tiktok/gmddc-pretrain-k10-b0.3-seed2/last_model.pth" ]; then
    echo "=== Stage 1 skipped: reusing ./checkpoint/tiktok/gmddc-pretrain-k10-b0.3-seed2"
else
    echo "=== Stage 1: pretraining completer"
    python main_pretrain.py \
        --dataset tiktok --seed 2 --variant raid-cf \
        --retrieval_k 10 --retrieval_beta 0.3 || exit 1

    # main_pretrain.py saves to ./checkpoint/tiktok/seed-2-<timestamp>/, so the folder
    # name is only known after it runs: read it back from the "Save path: " line in the log.
    SAVE_PATH=$(grep -m1 'Save path: ' "logs/raid-cf-raid-cf-tiktok-l1.0-gk10-k10-b0.3-seed2-${SLURM_JOB_ID}.out" | sed 's/.*Save path: //')
    if [ -n "$SAVE_PATH" ] && [ -d "$SAVE_PATH" ]; then
        ln -sfn "$(basename "$SAVE_PATH")" ./checkpoint/tiktok/gmddc-pretrain-k10-b0.3-seed2
        echo "Pretrained completer available at: ./checkpoint/tiktok/gmddc-pretrain-k10-b0.3-seed2 -> $SAVE_PATH"
    else
        echo "ERROR: could not resolve save path from stage 1" >&2
        exit 1
    fi
fi

if [ ! -f "./checkpoint/tiktok/gmddc-pretrain-k10-b0.3-seed2/last_model.pth" ]; then
    echo "ERROR: ./checkpoint/tiktok/gmddc-pretrain-k10-b0.3-seed2/last_model.pth still missing after stage 1" >&2
    exit 1
fi

mkdir -p ./checkpoint/tiktok/raid-cf-raid-cf-l1.0-gk10-k10-b0.3-seed2
cp -n ./checkpoint/tiktok/gmddc-pretrain-k10-b0.3-seed2/last_model.pth \
      ./checkpoint/tiktok/raid-cf-raid-cf-l1.0-gk10-k10-b0.3-seed2/

echo "=== Stage 2: joint training (raid-cf)"
python main.py \
    --alpha_1 0.7 --delta 0.3 --eta 0.6 --H 4 --lambda_1 0.06 \
    --dataset tiktok \
    --load_dir ./checkpoint/tiktok/raid-cf-raid-cf-l1.0-gk10-k10-b0.3-seed2/ \
    --load_model last_model.pth --seed 2 \
    --alpha_2 0.3 --anchor_rate 0.5 \
    --variant raid-cf --lambda_cf 1.0 --group_k 10 \
    --retrieval_k 10 --retrieval_beta 0.3 \
    --ckpt_dir ./checkpoint/tiktok/raid-cf-raid-cf-l1.0-gk10-k10-b0.3-seed2/

echo "Finished $(date)"
