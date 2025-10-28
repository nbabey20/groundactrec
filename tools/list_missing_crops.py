#!/usr/bin/env python3
"""
List all originals that don't have all three cropped views (left/right/top)
under the matching action folder.

Usage:
  python tools/list_missing_crops.py \
    --orig-root /workspace/datasets/inhard/InHARD/Segmented/RGBSegmented \
    --crop-root /workspace/cropped \
    --out-dir   /workspace/datasets/inhard/indices

Then crop the missing ones:
  parallel -j 8 --arg-sep ' ' 'python /workspace/crop_inhard.py {1} /workspace/cropped' \
    :::: /workspace/datasets/inhard/indices/missing_crops.txt
"""
from pathlib import Path
import argparse, csv, re

RE_ORIG = re.compile(r"^(P\d{2}_R\d{2})_(\d+\.\d+?)_(\d+\.\d+?)\.mp4$")  # P04_R02_0444.80_0450.32.mp4
VIEWS = ("left","right","top")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-root", required=True, type=Path)
    ap.add_argument("--crop-root", required=True, type=Path)
    ap.add_argument("--out-dir",   required=True, type=Path)
    ap.add_argument("--strict", action="store_true",
                    help="Strictly require outputs under the *same* label folder (default True).")
    args = ap.parse_args()

    orig_root = args.orig_root.resolve()
    crop_root = args.crop_root.resolve()
    out_dir   = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    missing_txt  = out_dir / "missing_crops.txt"
    missing_csv  = out_dir / "missing_crops_detail.csv"

    total = done = 0
    skipped_badname = 0
    rows = []

    with open(missing_txt, "w") as f_txt, open(missing_csv, "w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=[
            "original_path","label","basename","has_left","has_right","has_top","missing_views"
        ])
        writer.writeheader()

        # walk originals by label/action folder
        for label_dir in sorted(orig_root.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for mp4 in sorted(label_dir.glob("*.mp4")):
                total += 1
                base = mp4.stem  # e.g., P04_R02_0444.80_0450.32
                if not RE_ORIG.match(mp4.name):
                    skipped_badname += 1
                    continue

                # destination folder mirrors label
                out_dir_label = crop_root / label
                need = {v: (out_dir_label / f"{base}_{v}.mp4") for v in VIEWS}
                has  = {v: need[v].exists() for v in VIEWS}
                missing = [v for v in VIEWS if not has[v]]

                if missing:
                    f_txt.write(str(mp4) + "\n")
                    writer.writerow({
                        "original_path": str(mp4),
                        "label": label,
                        "basename": base,
                        "has_left": int(has["left"]),
                        "has_right": int(has["right"]),
                        "has_top": int(has["top"]),
                        "missing_views": ",".join(missing),
                    })
                else:
                    done += 1

    print("=== Missing-crops scan ===")
    print(f"Originals scanned : {total}")
    print(f"Fully cropped     : {done}")
    print(f"Need cropping     : {total - done}")
    if skipped_badname:
        print(f"Skipped non-matching filenames: {skipped_badname}")
    print(f"Wrote:\n  {missing_txt}\n  {missing_csv}")

if __name__ == "__main__":
    main()
