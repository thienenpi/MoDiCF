#!/bin/bash
#SBATCH --job-name=gmddc-tiktok
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-9

# NEW METHOD, stage 2/2: joint MoDiCF training on tiktok where the missing modalities are
# completed by the graph-retrieval-enhanced diffusion completer instead of the author's
# self-based one. For each item/modality, K anchor items are retrieved from the item graph
# (mean cosine over co-observed modalities + beta * normalized co-interaction) and their
# target-modality encodings are gate-fused into the denoiser's condition.
# Requires stage 1 = jobs/pretrain/tiktok.slurm (supplies the pretrained fusion heads).
# Compared against the officially published MoDiCF numbers (paper Table 2), NOT against a
# local reproduction — the self-based completer is no longer reachable on this branch.
# Repeated over 10 seeds (paper protocol, Sec 5.2.3) via a SLURM job array.
# Usage: sbatch jobs/gmddc/tiktok.slurm [retrieval_k] [retrieval_beta]   (defaults: 10, 0.3)
# Aggregate with: python scripts/aggregate_seeds.py "logs/gmddc-tiktok-k<K>-b<beta>-seed*.out"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
K="${1:-10}"
BETA="${2:-0.3}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
exec > "logs/gmddc-tiktok-k${K}-b${BETA}-seed${SEED}-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); retrieval_k=$K; beta=$BETA; seed=$SEED; started $(date)"
nvidia-smi || true

# main.py caches completed_data.pkl inside --load_dir, and the completion depends on the
# seed-specific missing mask. Give each seed its own copy of the pretrained weights so the
# array members cannot read or overwrite each other's completion.
PRETRAIN="./checkpoint/tiktok/gmddc-pretrain-k${K}-b${BETA}-seed${SEED}"
RUN_DIR="./checkpoint/tiktok/gmddc-k${K}-b${BETA}-seed${SEED}"
# last_model.pth, not best_model.pth: the latter is selected by reconstruction MSE
# against the ground truth of the held-out cells, which this setting treats as
# unavailable. Stage 1 logs show train/eval loss nearly equal, so last costs nothing.
if [ ! -f "${PRETRAIN}/last_model.pth" ]; then
    echo "Missing ${PRETRAIN}/last_model.pth - run jobs/pretrain/tiktok.slurm first" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"
cp -n "${PRETRAIN}/last_model.pth" "${RUN_DIR}/"

# dataset-specific hyperparameters from the author's README
python main.py \
    --alpha_1 0.7 --delta 0.3 --eta 0.6 --H 4 --lambda_1 0.06 \
    --dataset tiktok --load_dir "${RUN_DIR}/" --load_model last_model.pth --seed ${SEED} \
    --gamma 20 --alpha_2 0.3 --anchor_rate 0.5 \
    --retrieval_k ${K} --retrieval_beta ${BETA}

echo "Finished $(date)"
