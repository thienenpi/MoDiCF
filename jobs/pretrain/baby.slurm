#!/bin/bash
#SBATCH --job-name=gmddc-baby-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue

# Stage 1/2: pretrain the graph-retrieval-enhanced diffusion completer (GraphMSDiffusion)
# on baby. The author's published checkpoint cannot be reused as-is because this method
# adds per-modality fusion heads, so the completer must be pretrained here.
# Builds (and caches, under data/baby/) the top-K anchor index before training.
# Usage: sbatch jobs/pretrain/baby.slurm [retrieval_k] [retrieval_beta] [seed]  (defaults: 10, 0.3, 0)
# Feeds stage 2 = jobs/gmddc/baby.slurm via the checkpoint/baby/gmddc-pretrain-* symlink.

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
K="${1:-10}"
BETA="${2:-0.3}"
SEED="${3:-0}"
LOG="logs/gmddc-pretrain-baby-k${K}-b${BETA}-seed${SEED}-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); retrieval_k=$K; beta=$BETA; seed=$SEED; started $(date)"
nvidia-smi || true

python main_pretrain.py \
    --dataset baby --seed ${SEED} \
    --retrieval_k ${K} --retrieval_beta ${BETA}

# main_pretrain.py writes to a timestamped dir and prints "Save path: <dir>".
# Expose it under a stable name so stage 2 can pass it as --load_dir.
SAVE_PATH=$(grep -m1 '^Save path: ' "$LOG" | sed 's/^Save path: //')
STABLE="./checkpoint/baby/gmddc-pretrain-k${K}-b${BETA}-seed${SEED}"
if [ -n "$SAVE_PATH" ] && [ -d "$SAVE_PATH" ]; then
    ln -sfn "$(cd "$SAVE_PATH" && pwd)" "$STABLE"
    echo "Pretrained completer available at: $STABLE -> $SAVE_PATH"
else
    echo "WARNING: could not resolve save path; stage 2 needs --load_dir set manually" >&2
fi

echo "Finished $(date)"
