#!/usr/bin/env python3
# crop_inhard.py  – split one 1280×720 mosaic into 3× 640×360 clips
import subprocess, sys
from pathlib import Path

# ----------------------------------------------------------------------
src      = Path(sys.argv[1]).expanduser().resolve()     # input clip
dst_root = Path(sys.argv[2]).expanduser().resolve()     # e.g.  workspace/cropped
assert src.is_file(), f"Input not found: {src}"

# Make an output sub‑directory that mirrors the clip’s parent folder
#  “Put down measuring rod/P03_R03_…mp4”  -->  workspace/cropped/Put down measuring rod/
out_dir = dst_root / src.parent.name                   # src.parent.name == class folder
out_dir.mkdir(parents=True, exist_ok=True)

name = src.stem                                         # P03_R03_0029.72_0030.68
out_top   = out_dir / f"{name}_top.mp4"
out_left  = out_dir / f"{name}_left.mp4"
out_right = out_dir / f"{name}_right.mp4"

# ----------------------------------------------------------------------
cmd = [
    "ffmpeg", "-loglevel", "error", "-y", "-i", str(src),
    "-filter_complex",
    "[0:v]split=3[v0][v1][v2];"
    "[v0]crop=iw/2:ih/2:0:0[vtop];"
    "[v1]crop=iw/2:ih/2:iw/2:0[vleft];"
    "[v2]crop=iw/2:ih/2:iw/2:ih/2[vright]",
    "-map", "[vtop]",   str(out_top),   "-an",
    "-map", "[vleft]",  str(out_left),  "-an",
    "-map", "[vright]", str(out_right), "-an"
]
subprocess.check_call(cmd)
print(f"✓  Cropped clips written to {out_dir}")
