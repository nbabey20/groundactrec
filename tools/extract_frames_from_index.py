#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path
import pandas as pd

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print("❌", " ".join(cmd))
        print(p.stdout)
    return p.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, type=Path,
                    help="CSV with columns at least: clip,label. (baseline_skel_val_bg.csv)")
    ap.add_argument("--video-root", default="/workspace/cropped", type=Path,
                    help="Where <label>/<clip>.mp4 lives.")
    ap.add_argument("--frames-root", default="/workspace/frames", type=Path,
                    help="Where to write <label>/<clip>/*.jpg")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract even if frames already exist.")
    ap.add_argument("--fps", type=str, default=None,
                    help="Optional: override FPS (e.g., 15). If omitted, keep native FPS.")
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    assert {"clip","label"}.issubset(df.columns), "Index must have clip,label"

    for _, r in df.iterrows():
        label = str(r["label"])
        clip  = str(r["clip"])
        vid   = args.video_root / label / f"{clip}.mp4"
        outdir = args.frames_root / label / clip
        outdir.mkdir(parents=True, exist_ok=True)

        # Skip if frames already there (heuristic: at least 2 jpgs)
        if not args.overwrite and any(outdir.glob("*.jpg")):
            continue

        if not vid.exists():
            print(f"⚠️  missing video: {vid}")
            continue

        # ffmpeg: extract full-quality JPEGs, preserve frames (no VFR drop)
        cmd = ["ffmpeg", "-y", "-i", str(vid), "-vsync", "0"]
        if args.fps: cmd += ["-r", str(args.fps)]
        cmd += [str(outdir / "%06d.jpg")]
        rc = run(cmd)
        if rc == 0:
            print(f"✔ extracted frames → {outdir}")

if __name__ == "__main__":
    main()
