#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0.  INPUTS
###############################################################################
#  • first CLI argument  → explicit list file
#  • or $CLIP_LIST env   → fallback (lets you `export CLIP_LIST=…` once)
#  • otherwise bail out
LIST_FILE="${1:-${CLIP_LIST:-}}"
[ -f "$LIST_FILE" ] || { echo "❌  List file not found: $LIST_FILE"; exit 1; }

###############################################################################
# 1.  PATHS / PARAMS
###############################################################################
IN_DIR="/workspace"                       # list paths are relative to this
OUT_DIR="/workspace/results/comotion"
PY="/workspace/envs/comotion/bin/python"  # the venv you used on the A100 pod
JOBS="${J:-8}"                            # default: 8 parallel jobs

LOG_DIR="/workspace/logs"; mkdir -p "$LOG_DIR"
JOBLOG="$LOG_DIR/$(basename "$LIST_FILE").log"   # one log per list

###############################################################################
# 2.  LIGHT RUNTIME DEPS  (idempotent – will be skipped if already present)
###############################################################################
command -v parallel >/dev/null 2>&1 || {
  echo "Installing GNU parallel …"
  apt-get update -qq && apt-get install -y parallel
}

python -m pip install --quiet --no-deps \
       click tqdm einops timm yacs opencv-python >/dev/null

###############################################################################
# 3.  MAIN PROCESSING LOOP
###############################################################################
export PY IN_DIR OUT_DIR                   # visible inside GNU-parallel

parallel --joblog "$JOBLOG" --line-buffer -j "$JOBS" '
  rel={};                                   # line from the list, e.g. cropped/…
  vid="$IN_DIR/$rel"                        # absolute path to video

  # strip the leading “cropped/” so results mirror the dataset structure
  subdir="${rel#cropped/}"
  out="$OUT_DIR/${subdir%.mp4}.pt"

  # already processed? → skip
  [ -f "$out" ] && exit 0

  mkdir -p "$(dirname "$out")"

  "$PY" /workspace/ml-comotion/demo.py \
        -i "$vid" \
        -o "$(dirname "$out")" \
        --skip-visualization
'  < "$LIST_FILE"

echo "Finished $LIST_FILE"

