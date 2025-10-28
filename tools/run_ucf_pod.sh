#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash /workspace/tools/run_ucf_pod.sh 1
#      -> runs ...remaining.shard1.list
#   bash /workspace/tools/run_ucf_pod.sh /workspace/lists/ucf_all_videos.remaining.shard3.list
#      -> runs the explicit list path

# --- Resolve shard/list argument ---
ARG="${1:-}"
if [[ -z "$ARG" ]]; then
  echo "Usage: $0 <shard_number|list_path>"
  exit 1
fi

if [[ "$ARG" =~ ^[0-9]+$ ]]; then
  LIST_FILE="/workspace/lists/ucf_all_videos.remaining.shard${ARG}.list"
else
  LIST_FILE="$ARG"
fi

if [[ ! -f "$LIST_FILE" ]]; then
  echo "❌ List not found: $LIST_FILE"
  exit 1
fi

# --- Optional knobs (can be overridden per pod) ---
export J="${J:-4}"                         # parallel jobs inside worker
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Env setup ---
export PATH=/workspace/miniconda/bin:$PATH
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate /workspace/envs/comotion

chmod +x /workspace/tools/worker_ucf.sh

echo "▶ Using GPU(s): ${CUDA_VISIBLE_DEVICES}"
echo "▶ Parallel jobs per pod (J): ${J}"
echo "▶ Processing list: ${LIST_FILE}"

# --- Run one shard ---
bash /workspace/tools/worker_ucf.sh "$LIST_FILE"

echo "✔ Done: ${LIST_FILE}"
