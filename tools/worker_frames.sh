#!/usr/bin/env bash
set -euo pipefail

# Args:
#   $1: list file containing relative FRAME paths like:
#         frames/<label>/<clip>/000001.jpg
#         frames/<label>/<clip>/000002.jpg
#   (You can generate this with 'find' shown below.)

LIST_FILE="${1:-}"
[ -f "$LIST_FILE" ] || { echo "❌  list not found: $LIST_FILE"; exit 1; }

IN_DIR="/workspace"                          # list entries are relative to this
OUT_DIR="/workspace/results/comotion_frames" # per-frame .pt output root
PY="/workspace/envs/comotion/bin/python"     # your CoMotion venv Python
JOBS="${J:-8}"

LOG_DIR="/workspace/logs"; mkdir -p "$LOG_DIR"
JOBLOG="$LOG_DIR/$(basename "$LIST_FILE").log"

# lightweight deps (idempotent)
command -v parallel >/dev/null 2>&1 || {
  echo "Installing GNU parallel …"
  apt-get update -qq && apt-get install -y parallel
}
python -m pip install --quiet --no-deps click tqdm >/dev/null

export PY IN_DIR OUT_DIR

parallel --joblog "$JOBLOG" --line-buffer -j "$JOBS" '
  rel={};                                  # frames/<label>/<clip>/000123.jpg
  img="$IN_DIR/$rel"
  # map frames/<label>/<clip>/000123.jpg → results/comotion_frames/<label>/<clip>/000123.pt
  sub="${rel#frames/}"                     # <label>/<clip>/000123.jpg
  out_dir="$OUT_DIR/$(dirname "$sub")"
  out_pt="$out_dir/$(basename "${sub%.jpg}.pt")"

  [ -f "$out_pt" ] && exit 0
  mkdir -p "$out_dir"

  # Single-image path → demo.py runs run_detection() and writes a .pt with confidences
  "$PY" /workspace/ml-comotion/demo.py \
        -i "$img" \
        -o "$out_dir" \
        --skip-visualization
' < "$LIST_FILE"

echo "✔ Finished frames list: $LIST_FILE"
