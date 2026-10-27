#!/bin/bash
#SBATCH --job-name=raid-cf-tiktok
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --array=0-9

# FULL FRAMEWORK (RAID-CF) on tiktok: graph-retrieval-enhanced diffusion completer
# (jobs/gmddc) + group-margin counterfactual debiasing (jobs/deconf-group), trained jointly.
# Stage 1 reuses the gmddc completer pretrain (same main_pretrain.py / GraphMSDiffusion);
# the deconf-group pretrain-seed* checkpoints are NOT usable: they lack the fusion heads.
# Repeated over 10 seeds (paper protocol, Sec 5.2.3) via a SLURM job array.
# Usage: sbatch jobs/raid-cf/tiktok.slurm [lambda_cf] [group_k] [retrieval_k] [retrieval_beta]
#        (defaults: 1.0, 10, 10, 0.3)
# Aggregate with: python scripts/aggregate_seeds.py "logs/raid-cf-tiktok-l<lambda>-gk<group_k>-k<K>-b<beta>-seed*.out"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
LAMBDA="${1:-1.0}"
GROUP_K="${2:-10}"
K="${3:-10}"
BETA="${4:-0.3}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
LOG="logs/raid-cf-tiktok-l${LAMBDA}-gk${GROUP_K}-k${K}-b${BETA}-seed${SEED}-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); lambda_cf=$LAMBDA; group_k=$GROUP_K; retrieval_k=$K; beta=$BETA; seed=$SEED; started $(date)"
nvidia-smi || true

# Same name as jobs/pretrain/tiktok.slurm, so a completer pretrained for the gmddc ablation is reused.
PRETRAIN="./checkpoint/tiktok/gmddc-pretrain-k${K}-b${BETA}-seed${SEED}"
# Own run dir: holds completed_data.pkl and the resume checkpoint, so this run can never
# resume from (or overwrite) an ablation run's state.
RUN_DIR="./checkpoint/tiktok/raid-cf-l${LAMBDA}-gk${GROUP_K}-k${K}-b${BETA}-seed${SEED}"

if [ -f "${PRETRAIN}/last_model.pth" ]; then
    echo "=== Stage 1 skipped: reusing $PRETRAIN"
else
    echo "=== Stage 1: pretraining graph-retrieval completer"
    python main_pretrain.py \
        --dataset tiktok --seed ${SEED} \
        --retrieval_k ${K} --retrieval_beta ${BETA} || exit 1

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

echo "=== Stage 2: joint RAID-CF training"
python main.py \
    --alpha_1 0.7 --delta 0.3 --eta 0.6 --H 4 --lambda_1 0.06 \
    --dataset tiktok --load_dir "${RUN_DIR}/" --load_model last_model.pth --seed ${SEED} \
    --alpha_2 0.3 --anchor_rate 0.5 \
    --retrieval_k ${K} --retrieval_beta ${BETA} \
    --debias_mode subtract --gamma_mode group --lambda_cf ${LAMBDA} --group_k ${GROUP_K} \
    --ckpt_dir "${RUN_DIR}/"

echo "Finished $(date)"
