#!/usr/bin/env python3
import argparse, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CONF_CANDIDATES = [
    "conf", "scores", "kps_conf", "kp_confs",
    "keypoints_2d_conf", "joints2d_conf", "joints_conf"
]
KPS2D_CANDIDATES = ["keypoints_2d", "kps2d", "joints2d", "kp_2d", "pose2d"]
KPS3D_CANDIDATES = ["keypoints_3d", "kps3d", "joints3d", "kp_3d", "pose3d"]

def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)

def _extract_conf_per_person(d: Dict) -> List[np.ndarray]:
    people = []
    # explicit confidence arrays
    for k in CONF_CANDIDATES:
        if k in d:
            v = _np(d[k])
            if v.ndim == 2:      # [P,J]
                return [v[p] for p in range(v.shape[0])]
            if v.ndim == 1:      # [J]
                return [v]
            if v.ndim == 3 and v.shape[-1] == 1:  # [P,J,1]
                return [v[p,:,0] for p in range(v.shape[0])]
    # infer from keypoints last channel
    for k in (KPS2D_CANDIDATES + KPS3D_CANDIDATES):
        if k in d:
            v = _np(d[k])
            if v.ndim == 3 and v.shape[-1] >= 3:   # [P,J,C]
                return [v[p,:,2] for p in range(v.shape[0])]
            if v.ndim == 2 and v.shape[-1] >= 3:   # [J,C]
                return [v[:,2]]
    return []

def _pick_primary(conf_per_person: List[np.ndarray]) -> int:
    if not conf_per_person: return -1
    means = [float(np.nanmean(c)) if c.size else -1.0 for c in conf_per_person]
    return int(np.argmax(means))

def _summ_frame(c1p: np.ndarray, low_thr: float) -> Dict[str, float]:
    c = c1p.astype(np.float32)
    c = c[~np.isnan(c)]
    if c.size == 0:
        return dict(mean_conf=np.nan, frac_low=np.nan, num_joints=0)
    num_low = float((c < low_thr).sum())
    return dict(mean_conf=float(np.mean(c)),
                frac_low=float(num_low/len(c)),
                num_joints=int(len(c)))

def _collect_pts(dir_path: Path) -> List[Path]:
    return sorted(dir_path.glob("*.pt"))

def process_clip(frames_dir: Path, low_thr: float, frame_low_frac: float):
    per_frame = []
    for pt in _collect_pts(frames_dir):
        try:
            d = torch.load(pt, map_location="cpu", weights_only=False)
        except Exception:
            continue
        confs = _extract_conf_per_person(d)
        if not confs: continue
        c1p = confs[_pick_primary(confs)]
        per_frame.append(_summ_frame(c1p, low_thr))

    if not per_frame:
        return dict(n_frames=0, mean_conf=np.nan, pct_low_frames=np.nan,
                    mean_frac_low_joints=np.nan)

    n = len(per_frame)
    mean_conf = float(np.nanmean([s["mean_conf"] for s in per_frame]))
    mean_frac_low_joints = float(np.nanmean([s["frac_low"] for s in per_frame]))
    pct_low_frames = 100.0 * float(np.mean([s["frac_low"] >= frame_low_frac
                                            for s in per_frame if not math.isnan(s["frac_low"])]))
    return dict(n_frames=n, mean_conf=mean_conf,
                pct_low_frames=pct_low_frames,
                mean_frac_low_joints=mean_frac_low_joints)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True,
                    help="CSV with columns clip,label (validation set).")
    ap.add_argument("--frames-root", type=Path, required=True,
                    help="Root of per-frame .pt files: <root>/<label>/<clip>/*.pt")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--low-thr", type=float, default=0.25,
                    help="Joint confidence below this is considered low.")
    ap.add_argument("--frame-low-frac", type=float, default=0.30,
                    help="Frame is low if >= this fraction of joints are low.")
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    assert {"clip","label"}.issubset(df.columns)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Aggregating"):
        label, clip = str(r["label"]), str(r["clip"])
        clip_dir = args.frames_root / label / clip
        m = process_clip(clip_dir, args.low_thr, args.frame_low_frac)
        rows.append(dict(label=label, clip=clip, frames_dir=str(clip_dir),
                         low_thr=args.low_thr, frame_low_frac=args.frame_low_frac, **m))
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"✔ wrote {args.out_csv}  (rows={len(out)})")

if __name__ == "__main__":
    main()
