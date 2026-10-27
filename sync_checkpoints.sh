#!/bin/bash
# Pull one dataset's trained weights / training state / logs / neighbor-index caches
# from the HCMUS server into this local repo, mirroring the SAME relative paths
# (checkpoint/<dataset>/, logs/*<dataset>*, data/<dataset>/retrieval_neighbors_*.npz).
# Safe to run repeatedly (e.g. via cron/watch) as a backup against server crash or
# disk loss.
set -euo pipefail

# --- config: edit REMOTE_HOST (and PORT if non-standard) ---
REMOTE_USER="${REMOTE_USER:-lhbac29}"
REMOTE_HOST="${REMOTE_HOST:-172.29.74.81}"          # e.g. the HCMUS ssh host/IP
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/media/lhbac29/raid-cf}"  # repo root on the server

# Local repo root = directory containing this script (keeps the same paths).
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$REMOTE_HOST" = "CHANGE_ME" ]; then
    echo "Set REMOTE_HOST first (edit the script or: REMOTE_HOST=<host> $0)" >&2
    exit 1
fi

# One dataset per run: keeps a single invocation short enough to babysit, and
# lets several run in parallel (one terminal each) when you do want everything.
DATASETS=(baby tiktok allrecipes)

usage() {
    echo "usage: $0 <dataset>" >&2
    echo "  dataset: ${DATASETS[*]} | all" >&2
    exit 1
}

[ $# -eq 1 ] || usage
DATASET="$1"

if [ "$DATASET" = all ]; then
    for d in "${DATASETS[@]}"; do "$0" "$d"; done
    exit 0
fi

valid=0
for d in "${DATASETS[@]}"; do [ "$d" = "$DATASET" ] && valid=1; done
[ "$valid" -eq 1 ] || { echo "unknown dataset: $DATASET" >&2; usage; }

# -a archive, -z compress, --partial resume interrupted transfers,
# --info=progress2 single overall progress bar. No --delete: never remove local
# backups just because they vanished on the server.
RSYNC_OPTS=(-az --partial --info=progress2 -e "ssh -p ${REMOTE_PORT}")

# checkpoint/ is one dir per dataset, so scope it to that dir. logs/ is flat with
# the dataset embedded in each filename, so filter by name instead.
for sub in "checkpoint/${DATASET}" logs; do
    src="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${sub}/"
    dst="${LOCAL_DIR}/${sub}/"
    filter=()
    [ "$sub" = logs ] && filter=(--include="*${DATASET}*" --exclude="*")

    if ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" "[ -d '${REMOTE_DIR}/${sub}' ]"; then
        echo ">> syncing ${sub}/ -> ${dst}"
        mkdir -p "$dst"
        rsync "${RSYNC_OPTS[@]}" "${filter[@]}" "$src" "$dst"
    else
        echo ">> skip ${sub}/ (not present on server)"
    fi
done

# The precomputed anchor index lives next to the features under data/<dataset>/ and is
# what pins down which neighbors a run actually used, so it is worth keeping. Pull only
# the .npz caches - never the multi-GB *_feat.npy files. Scoped to this dataset's dir,
# so no --prune-empty-dirs walk over the other datasets is needed.
sub="data/${DATASET}"
if ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" "[ -d '${REMOTE_DIR}/${sub}' ]"; then
    echo ">> syncing ${sub}/retrieval_neighbors_*.npz -> ${LOCAL_DIR}/${sub}/"
    mkdir -p "${LOCAL_DIR}/${sub}/"
    rsync "${RSYNC_OPTS[@]}" \
        --include='retrieval_neighbors_*.npz' --exclude='*' \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${sub}/" "${LOCAL_DIR}/${sub}/"
else
    echo ">> skip ${sub}/ (not present on server)"
fi

echo "Done ${DATASET} $(date)"
