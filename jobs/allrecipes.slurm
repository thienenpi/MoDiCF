#!/bin/bash
#SBATCH --job-name=gmddc-allrecipes-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --array=0-9

# NEW METHOD, both stages in one job, on allrecipes.
#   Stage 1: pretrain the graph-retrieval-enhanced diffusion completer (GraphMSDiffusion).
#            The author's published checkpoint cannot be reused as-is because this method
#            adds per-modality fusion heads. Also builds (and caches, under data/allrecipes/)
#            the top-K anchor index.
#   Stage 2: joint MoDiCF training where the missing modalities are completed by that
#            completer instead of the author's self-based one. For each item/modality,
#            K anchor items are retrieved from the item graph (mean cosine over co-observed
#            modalities + beta * normalized co-interaction) and their target-modality
#            encodings are gate-fused into the denoiser's condition.
# One array task per seed, and the two stages of a seed stay in the same task: the missing
# mask is seed-dependent (data.py incomplete()), so a completer pretrained on seed s' mask
# must not feed seed s != s' - it would have seen the ground truth of cells seed s hides.
# Repeated over 10 seeds (paper protocol, Sec 5.2.3); compared against the officially
# published MoDiCF numbers (paper Table 2), NOT against a local reproduction - the
# self-based completer is no longer reachable on this branch.
# Usage: sbatch jobs/allrecipes.slurm [retrieval_k] [retrieval_beta]   (defaults: 10, 0.3)
# Aggregate with: python scripts/aggregate_seeds.py "logs/gmddc-allrecipes-k<K>-b<beta>-seed*.out"
# The split-stage variants (jobs/pretrain/, jobs/gmddc/) still exist and remain the cheaper
# route when tuning stage-2 hyperparameters against one fixed completer.

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs
K="${1:-10}"
BETA="${2:-0.3}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
# Same log name as the split stage 2, so aggregate_seeds.py needs no change: it keys the
# seed off "-seed<N>-" in the filename and only matches "Test_@[...]: Recall=[...]" lines,
# which stage 1 never emits.
LOG="logs/gmddc-allrecipes-k${K}-b${BETA}-seed${SEED}-${SLURM_JOB_ID}.out"
exec > "$LOG" 2>&1
module purge 2>/dev/null || true
source ~/.bashrc
conda activate thesis

echo "Job $SLURM_JOB_ID on $(hostname); retrieval_k=$K; beta=$BETA; seed=$SEED; started $(date)"
nvidia-smi || true

PRETRAIN="./checkpoint/allrecipes/gmddc-pretrain-k${K}-b${BETA}-seed${SEED}"
RUN_DIR="./checkpoint/allrecipes/gmddc-k${K}-b${BETA}-seed${SEED}"

# ---------------------------------------------------------------- stage 1: completer
# Skipped when the completer for this exact (dataset, seed, k, beta) is already on disk.
# This is what makes --requeue safe: a job preempted during stage 2 restarts this script
# from the top, and without the guard it would burn another full pretrain run.
if [ -f "${PRETRAIN}/last_model.pth" ]; then
    echo "=== Stage 1 skipped: reusing $PRETRAIN"
else
    echo "=== Stage 1: pretraining completer"
    python main_pretrain.py \
        --dataset allrecipes --seed ${SEED} \
        --retrieval_k ${K} --retrieval_beta ${BETA} || exit 1

    # main_pretrain.py writes to a timestamped dir and prints "Save path: <dir>".
    # Expose it under the stable name stage 2 reads, and keep it stable so a later
    # split-stage run (jobs/gmddc/allrecipes.slurm) can pick this completer up too.
    SAVE_PATH=$(grep -m1 'Save path: ' "$LOG" | sed 's/.*Save path: //')
    if [ -n "$SAVE_PATH" ] && [ -d "$SAVE_PATH" ]; then
        # relative target: both dirs share the same parent, so the symlink still
        # resolves after sync_checkpoints.sh rsyncs checkpoint/ to a different machine.
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

# ---------------------------------------------------------------- stage 2: recommender
# main.py caches completed_data.pkl inside --load_dir, and the completion depends on the
# seed-specific missing mask. Give each seed its own copy of the pretrained weights so the
# array members cannot read or overwrite each other's completion.
# last_model.pth, not best_model.pth: the latter is selected by reconstruction MSE
# against the ground truth of the held-out cells, which this setting treats as
# unavailable. Stage 1 logs show train/eval loss nearly equal, so last costs nothing.
mkdir -p "$RUN_DIR"
# -n so a requeued attempt keeps the completed_data.pkl this dir may already hold.
cp -n "${PRETRAIN}/last_model.pth" "${RUN_DIR}/"

echo "=== Stage 2: joint MoDiCF training"
# dataset-specific hyperparameters from the author's README
python main.py \
    --alpha_1 0.6 --delta 0.4 --eta 0.3 --gnn_embed_size 128 --H 8 --lambda_1 0.15 \
    --dataset allrecipes --load_dir "${RUN_DIR}/" --load_model last_model.pth --seed ${SEED} \
    --gamma 20 --alpha_2 0.5 --anchor_rate 0.75 \
    --retrieval_k ${K} --retrieval_beta ${BETA}

echo "Finished $(date)"
