#!/bin/bash
#SBATCH --job-name=raid-cf-baby-deconf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --array=0-9

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
GAMMA="${1:-0.5}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
LOG="logs/deconf-baby-g${GAMMA}-seed${SEED}-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); gamma_train=$GAMMA; seed=$SEED; started $(date)"
nvidia-smi || true

PRETRAIN="./checkpoint/baby/pretrain-seed${SEED}"
RUN_DIR="./checkpoint/baby/deconf-g${GAMMA}-seed${SEED}"

if [ -f "${PRETRAIN}/last_model.pth" ]; then
    echo "=== Stage 1 skipped: reusing $PRETRAIN"
else
    echo "=== Stage 1: pretraining completer"
    python main_pretrain.py \
        --dataset baby --seed ${SEED} || exit 1

    SAVE_PATH=$(grep -m1 'Save path: ' "$LOG" | sed 's/.*Save path: //')
    if [ -n "$SAVE_PATH" ] && [ -d "$SAVE_PATH" ]; then
        ln -sfn "$(basename "$SAVE_PATH")" "$PRETRAIN"
        echo "Pretrained completer available at: $PRETRAIN -> $SAVE_PATH"
    else
        echo "ERROR: could not resolve save path from stage 1" >&2
        exit 1
    fi
fi

if [ ! -f "${PRETRAIN}/last_model.pth" ]; then
    echo "ERROR: ${PRETRAIN}/last_model.pth still missing after stage 1" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
cp -n "${PRETRAIN}/last_model.pth" "${RUN_DIR}/"

echo "=== Stage 2: joint MoDiCF training"
python main.py \
    --dataset baby --load_dir "${RUN_DIR}/" --load_model last_model.pth --seed ${SEED} \
    --debias_mode subtract --gamma_train ${GAMMA} \
    --ckpt_dir "${RUN_DIR}/"

echo "Finished $(date)"
