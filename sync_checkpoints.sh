#!/bin/bash
# Pull trained weights / training state / logs from the HCMUS server into this
# local repo, mirroring the SAME relative paths (checkpoint/, logs/). Safe to run
# repeatedly (e.g. via cron/watch) as a backup against server crash or disk loss.
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

# -a archive, -z compress, --partial resume interrupted transfers,
# --info=progress2 single overall progress bar. No --delete: never remove local
# backups just because they vanished on the server.
RSYNC_OPTS=(-az --partial --info=progress2 -e "ssh -p ${REMOTE_PORT}")

for sub in checkpoint logs; do
    src="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${sub}/"
    dst="${LOCAL_DIR}/${sub}/"
    if ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" "[ -d '${REMOTE_DIR}/${sub}' ]"; then
        echo ">> syncing ${sub}/ -> ${dst}"
        rsync "${RSYNC_OPTS[@]}" "$src" "$dst"
    else
        echo ">> skip ${sub}/ (not present on server)"
    fi
done

echo "Done $(date)"
