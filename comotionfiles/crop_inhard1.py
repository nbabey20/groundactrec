#!/usr/bin/env python3
# crop_inhard.py — split 1280×720 mosaic into 3×640×360: top, left, right
import subprocess, sys
from pathlib import Path

src      = Path(sys.argv[1]).expanduser().resolve()   # input clip
dst_root = Path(sys.argv[2]).expanduser().resolve()   # e.g., /workspace/cropped
assert src.is_file(), f"Input not found: {src}"

out_dir = dst_root / src.parent.name                  # mirror class folder
out_dir.mkdir(parents=True, exist_ok=True)

name = src.stem                                       # e.g., P03_R03_0029.72_0030.68
out_top   = out_dir / f"{name}_top.mp4"
out_left  = out_dir / f"{name}_left.mp4"
out_right = out_dir / f"{name}_right.mp4"

# Resume-safe: skip if all three outputs already exist
if out_top.exists() and out_left.exists() and out_right.exists():
    print(f"✓ already done: {name}")
    sys.exit(0)

cmd = [
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
    "-i", str(src),
    "-filter_complex",
        "[0:v]split=3[v0][v1][v2];"
        "[v0]crop=iw/2:ih/2:0:0[vtop];"
        "[v1]crop=iw/2:ih/2:iw/2:0[vleft];"
        "[v2]crop=iw/2:ih/2:iw/2:ih/2[vright]",
    # TOP
    "-map", "[vtop]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_top),
    # LEFT
    "-map", "[vleft]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_left),
    # RIGHT
    "-map", "[vright]", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_right),
]

subprocess.check_call(cmd)
print(f"✓ Cropped clips written to {out_dir}")
