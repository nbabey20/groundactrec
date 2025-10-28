#!/usr/bin/env bash
set -euo pipefail

# Args:
#   $1: list file with video paths *relative to /workspace*, e.g.:
#       datasets/ucf101/UCF-101/PommelHorse/v_PommelHorse_g03_c02.avi
#       datasets/occlusion_extracted/58/vid0_0_7.avi

LIST_FILE="${1:-}"
[ -f "$LIST_FILE" ] || { echo "❌  list not found: $LIST_FILE"; exit 1; }

IN_DIR="/workspace"
OUT_DIR="/workspace/results/comotion_ucf"   # <— new output root
PY="/workspace/envs/comotion/bin/python"
JOBS="${J:-4}"                              # tune for your GPU/pod

LOG_DIR="/workspace/logs"; mkdir -p "$LOG_DIR"
JOBLOG="$LOG_DIR/$(basename "$LIST_FILE").log"

# deps (idempotent)
command -v parallel >/dev/null 2>&1 || {
  echo "Installing GNU parallel …"
  apt-get update -qq && apt-get install -y parallel
}
python -m pip install --quiet --no-deps click tqdm >/dev/null

export PY IN_DIR OUT_DIR

parallel --joblog "$JOBLOG" --line-buffer -j "$JOBS" '
  rel={};                             # e.g. datasets/ucf101/UCF-101/Apply.../v_...avi
  vid="$IN_DIR/$rel"

  # Mirror the *datasets/* subtree under results/comotion_ucf/
  # datasets/.../file.avi -> results/comotion_ucf/.../file.pt
  sub="${rel#datasets/}"              # ucf101/UCF-101/... or occlusion_extracted/...
  out_dir="$OUT_DIR/$(dirname "$sub")"
  out_pt="$OUT_DIR/${sub%.avi}.pt"

  # Already processed? -> skip
  [ -f "$out_pt" ] && exit 0
  mkdir -p "$out_dir"

  # Call your CoMotion demo (video mode -> writes .pt and .txt)
  "$PY" /workspace/ml-comotion/demo.py \
        -i "$vid" \
        -o "$out_dir" \
        --skip-visualization
' < "$LIST_FILE"

echo "✔ Finished $LIST_FILE"
