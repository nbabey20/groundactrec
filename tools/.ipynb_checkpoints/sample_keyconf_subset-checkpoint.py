# sample_keyconf_subset.py
#!/usr/bin/env python3
import argparse, random
from pathlib import Path
import pandas as pd
import numpy as np
from glob import glob
import re

def norm_label(lbl: str) -> str:
    s = str(lbl).strip()
    s = re.sub(r'^\s*\[[^\]]+\]\s*', '', s)  # drop "[OP###]"
    s = re.sub(r'\s+', ' ', s)
    return s

def find_video(cropped_root: Path, label_raw: str, clip: str):
    cand_labels = [norm_label(label_raw)]
    if cand_labels[0] != label_raw:
        cand_labels.append(label_raw)

    for lab in cand_labels:
        for cam in ("left","right","top"):
            p = cropped_root / lab / f"{clip}_{cam}.mp4"
            if p.exists():
                return p, cam, lab
    # fallback: anywhere under cropped_root
    m = sorted(glob(str(cropped_root / "**" / f"{clip}_*.mp4"), recursive=True))
    if m:
        p = Path(m[0]); cam = p.stem.split("_")[-1]; lab = p.parent.name
        return p, cam, lab
    return None, "", ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--cropped_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cropped_root = Path(args.cropped_root)
    df = pd.read_csv(args.index_csv)
    if "clip" not in df.columns or "label" not in df.columns:
        raise SystemExit("index_csv must have columns: clip,label")

    rng = np.random.default_rng(args.seed)
    # sample WITHOUT replacement
    take = min(args.n, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    sdf = df.iloc[idx].copy().reset_index(drop=True)

    rows = []
    found = 0
    for _, r in sdf.iterrows():
        clip  = str(r["clip"]).strip()
        label = str(r["label"]).strip()
        vp, cam, lab_used = find_video(cropped_root, label, clip)
        rows.append({
            "clip": clip,
            "label": label,
            "label_fs": lab_used,
            "camera": cam,
            "video_path": str(vp) if vp else "",
        })
        if vp: found += 1

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote pilot list -> {args.out_csv} | videos resolved {found}/{len(out)}")

if __name__ == "__main__":
    main()
