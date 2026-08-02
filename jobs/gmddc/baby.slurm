#!/bin/bash
#SBATCH --job-name=gmddc-baby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-9

# NEW METHOD, stage 2/2: joint MoDiCF training on baby where the missing modalities are
# completed by the graph-retrieval-enhanced diffusion completer instead of the author's
# self-based one. For each item/modality, K anchor items are retrieved from the item graph
# (mean cosine over co-observed modalities + beta * normalized co-interaction) and their
# target-modality encodings are gate-fused into the denoiser's condition.
# Requires stage 1 = jobs/pretrain/baby.slurm (supplies the pretrained fusion heads).
# Compared against the officially published MoDiCF numbers (paper Table 2), NOT against a
# local reproduction — the self-based completer is no longer reachable on this branch.
# Repeated over 10 seeds (paper protocol, Sec 5.2.3) via a SLURM job array.
# Usage: sbatch jobs/gmddc/baby.slurm [retrieval_k] [retrieval_beta]   (defaults: 10, 0.3)
# Aggregate with: python scripts/aggregate_seeds.py "logs/gmddc-baby-k<K>-b<beta>-seed*.out"

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
K="${1:-10}"
BETA="${2:-0.3}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
exec > "logs/gmddc-baby-k${K}-b${BETA}-seed${SEED}-${SLURM_JOB_ID}.out" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); retrieval_k=$K; beta=$BETA; seed=$SEED; started $(date)"
nvidia-smi || true

# main.py caches completed_data.pkl inside --load_dir, and the completion depends on the
# seed-specific missing mask. Give each seed its own copy of the pretrained weights so the
# array members cannot read or overwrite each other's completion.
PRETRAIN="./checkpoint/baby/gmddc-pretrain-k${K}-b${BETA}-seed0"
RUN_DIR="./checkpoint/baby/gmddc-k${K}-b${BETA}-seed${SEED}"
if [ ! -f "${PRETRAIN}/best_model.pth" ]; then
    echo "Missing ${PRETRAIN}/best_model.pth - run jobs/pretrain/baby.slurm first" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"
cp -n "${PRETRAIN}/best_model.pth" "${RUN_DIR}/"

python main.py \
    --dataset baby --load_dir "${RUN_DIR}/" --seed ${SEED} \
    --retrieval_k ${K} --retrieval_beta ${BETA}

echo "Finished $(date)"
