#!/bin/bash
#SBATCH --job-name=raid-cf-baby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --array=0-9

# RAID-CF on baby, one variant per submission (passed straight through as --variant):
#   retrieval = graph-retrieval completer + MoDiCF multiply scoring   (ablation: module 1)
#   debias    = author's self-based completer + group-margin debias   (ablation: module 2)
#   raid-cf   = graph-retrieval completer + group-margin debias       (full framework)
# Stage 1 pretrains the completer unless a matching one exists: retrieval / raid-cf share
# the gmddc pretrain (GraphMSDiffusion), debias reuses pretrain-seed* (MSDiffusion).
# Repeated over 10 seeds (paper protocol, Sec 5.2.3) via a SLURM job array.
# Usage: sbatch jobs/raid-cf/baby.slurm <retrieval|debias|raid-cf> [lambda_cf] [group_k] [retrieval_k] [retrieval_beta]
#        (defaults: 1.0, 10, 10, 0.3)
# Aggregate with: python scripts/aggregate_seeds.py "logs/raid-cf-<variant>-baby-*-seed*.out"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
VARIANT="$1"
case "$VARIANT" in
    retrieval|debias|raid-cf) ;;
    *) echo "Usage: sbatch $0 <retrieval|debias|raid-cf> [lambda_cf] [group_k] [retrieval_k] [retrieval_beta]" >&2; exit 1 ;;
esac
LAMBDA="${2:-1.0}"
GROUP_K="${3:-10}"
K="${4:-10}"
BETA="${5:-0.3}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"

# Name each run only by the hyperparameters its variant actually uses.
case "$VARIANT" in
    retrieval) TAG="k${K}-b${BETA}" ;;
    debias)    TAG="l${LAMBDA}-gk${GROUP_K}" ;;
    raid-cf)   TAG="l${LAMBDA}-gk${GROUP_K}-k${K}-b${BETA}" ;;
esac
LOG="logs/raid-cf-${VARIANT}-baby-${TAG}-seed${SEED}-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); variant=$VARIANT; lambda_cf=$LAMBDA; group_k=$GROUP_K; retrieval_k=$K; beta=$BETA; seed=$SEED; started $(date)"
nvidia-smi || true

if [ "$VARIANT" = "debias" ]; then
    PRETRAIN="./checkpoint/baby/pretrain-seed${SEED}"
else
    PRETRAIN="./checkpoint/baby/gmddc-pretrain-k${K}-b${BETA}-seed${SEED}"
fi
# Own run dir: holds completed_data.pkl and the resume checkpoint, so no variant can
# resume from (or overwrite) another variant's or an old ablation's state.
RUN_DIR="./checkpoint/baby/raid-cf-${VARIANT}-${TAG}-seed${SEED}"

if [ -f "${PRETRAIN}/last_model.pth" ]; then
    echo "=== Stage 1 skipped: reusing $PRETRAIN"
else
    echo "=== Stage 1: pretraining completer"
    python main_pretrain.py \
        --dataset baby --seed ${SEED} --variant ${VARIANT} \
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

GAMMA_ARG=""

echo "=== Stage 2: joint training ($VARIANT)"
python main.py \
    --dataset baby --load_dir "${RUN_DIR}/" --load_model last_model.pth --seed ${SEED} \
    --variant ${VARIANT} --lambda_cf ${LAMBDA} --group_k ${GROUP_K} \
    --retrieval_k ${K} --retrieval_beta ${BETA} \
    --ckpt_dir "${RUN_DIR}/" ${GAMMA_ARG}

echo "Finished $(date)"
